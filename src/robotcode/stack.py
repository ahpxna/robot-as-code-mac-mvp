"""The onboard robot software. It talks to the robot only through a RobotInterface.

Every control tick (50 Hz) the stack reads the sensors, updates localisation,
the obstacle costmap and the safety monitor, then advances the active skill.
Skills are generators: they issue commands, `yield` to wait for the next tick
and `return` a result -- the cooperative equivalent of ROS 2 action servers.
"""
from __future__ import annotations

from typing import Callable, Generator

import numpy as np

from .description import has_gripper
from .hal import LaserScan, RobotInterface, RobotState
from .kinematics import Kinematics
from .localization import Localizer
from .manipulation import Arm, Gripper
from .mapping import PriorMap
from .math_utils import to_map
from .navigation import DiffDriveController, Navigator
from .perception import WorldModel, make_detector
from .safety import SafetyMonitor


class RobotStack:
    def __init__(self, hal: RobotInterface, spec: dict, prior: PriorMap, catalog: dict, *,
                 initial_pose=None, log: Callable[[str], None] | None = None, detector: str | None = None):
        self.hal, self.spec, self.prior, self.catalog = hal, spec, prior, catalog
        self.dt = 1.0 / spec.get("control", {}).get("rate_hz", 50)
        self._log = log or (lambda msg: None)
        self.kin = Kinematics(spec)
        self.localizer = Localizer(prior, spec, prior.start_pose if initial_pose is None else initial_pose)
        self.drive = DiffDriveController(hal, spec)
        self.nav = Navigator(self)
        self.arm = Arm(self) if spec.get("arm") else None
        self.gripper = Gripper(self) if has_gripper(spec) else None
        self.detector = make_detector(self, detector) if "head_camera" in spec["sensors"] else None
        self.world = WorldModel(prior, catalog)
        self.safety = SafetyMonitor(self)
        self.state: RobotState | None = None
        self.scan: LaserScan | None = None
        self._active: Generator | None = None
        self._active_name = ""
        self.result = None
        self.log_lines: list[tuple[float, str]] = []
        self._grip_lost_since: float | None = None

    # ---- convenience views -----------------------------------------------------
    @property
    def time(self) -> float:
        return self.hal.time()

    @property
    def pose(self) -> np.ndarray:
        return self.localizer.pose

    @property
    def capabilities(self) -> set[str]:
        caps = {"navigate"}
        if self.detector:
            caps.add("perceive")
        if self.arm and self.gripper:
            caps.add("manipulate")
        return caps

    def grip_lost(self) -> str | None:
        """If the robot believes it holds something but the finger sensors disagree, say so (once)."""
        if self.world.held is None or self.gripper is None or self.gripper.check().holding:
            self._grip_lost_since = None
            return None
        # Debounce: contact forces flicker for a few ms during impacts.
        self._grip_lost_since = self._grip_lost_since if self._grip_lost_since is not None else self.time
        if self.time - self._grip_lost_since < 0.3:
            return None
        obj = self.world.objects[self.world.held]
        xy = to_map(self.pose, self.arm.gripper_position()[None, :2])[0]
        self.world.mark_released([xy[0], xy[1], 0.0], None, self.time)     # probably on the floor nearby
        # It is now on the floor, below the LiDAR plane: keep the base away from it.
        self.nav.costmap.add_obstacle(xy, 0.15, self.time)
        self.log(f"GRIP LOST: the {obj.label} slipped out of the gripper")
        return f"dropped the {obj.label} (finger force sensors lost contact)"

    def log(self, message: str) -> None:
        self.log_lines.append((self.time, message))
        self._log(f"[t={self.time:6.2f}s] {message}")

    # ---- skill scheduling ------------------------------------------------------------
    def start(self, name: str, generator: Generator) -> None:
        if self._active is not None:
            self._active.close()
        self._active, self._active_name, self.result = generator, name, None

    @property
    def busy(self) -> bool:
        return self._active is not None

    def cancel(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None
        self.drive.halt()

    def tick(self) -> None:
        self.state = self.hal.read_state()
        self.localizer.predict(self.state)
        scan = self.hal.read_scan()
        if scan is not None and (self.scan is None or scan.stamp > self.scan.stamp):
            self.scan = scan
            self.localizer.correct(scan)
            self.nav.update(self.time)
        self.safety.check()
        if self._active is None:
            self.drive.command(0.0, 0.0, self.dt)
            return
        try:
            next(self._active)
        except StopIteration as stop:
            self.result = stop.value
            self._active = None
            self.drive.halt()
