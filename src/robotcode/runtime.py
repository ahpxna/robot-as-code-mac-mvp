"""Simulation runtime: physics loop + robot software + an independent referee.

    scene (task, seed) -> MuJoCo model -> SimRobot (HAL) -> RobotStack (onboard software)

`GroundTruth` is the referee: it reads simulator state to score an episode
(collisions, where objects really ended up, localisation error). Nothing in the
robot stack can see it.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from .hal import SimRobot
from .scene import Scene, build_scene
from .stack import RobotStack


class GroundTruth:
    def __init__(self, model, data, scene: Scene):
        import mujoco

        self._mj = mujoco
        self.model, self.data, self.scene = model, data, scene
        base = model.body("base").id
        self.robot_bodies = {b for b in range(model.nbody) if self._root(b) == base}
        self.object_bodies = {model.body(o.name).id: o.name for o in scene.objects}
        self.gripper_bodies = {model.body(n).id for n in ("gripper_link", "finger_left", "finger_right")
                               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
        self.floor = model.geom("floor").id
        self.collisions: list[dict] = []
        self._in_contact: dict[int, float] = {}
        self.path_length = 0.0
        self._last_xy = self.robot_pose()[:2]
        self.loc_errors: list[float] = []
        self.object_disturbances: list[dict] = []

    def _root(self, body: int) -> int:
        while self.model.body_parentid[body] != 0:
            body = self.model.body_parentid[body]
        return body

    def robot_pose(self) -> np.ndarray:
        b = self.data.body("base")
        rot = b.xmat.reshape(3, 3)
        return np.array([b.xpos[0], b.xpos[1], math.atan2(rot[1, 0], rot[0, 0])])

    def object_position(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def object_upright(self, name: str) -> float:
        """cos(tilt) of the object's z axis; 1 = upright."""
        return float(self.data.body(name).xmat.reshape(3, 3)[2, 2])

    def after_step(self) -> None:
        m, d = self.model, self.data
        now = float(d.time)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for robot, other, geom in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
                if robot in self.robot_bodies and other in self.object_bodies and robot not in self.gripper_bodies:
                    name = self.object_bodies[other]
                    if now - self._in_contact.get(-other, -1.0) > 0.5:
                        self.object_disturbances.append({"t": round(now, 2), "object": name,
                                                         "robot_part": m.body(robot).name})
                    self._in_contact[-other] = now
                if robot in self.robot_bodies and other == 0 and geom != self.floor:
                    last = self._in_contact.get(geom, -1.0)
                    if now - last > 0.5:
                        self.collisions.append({"t": round(now, 2), "with": m.geom(geom).name,
                                                "robot_part": m.body(robot).name})
                    self._in_contact[geom] = now
        xy = self.robot_pose()[:2]
        self.path_length += float(np.linalg.norm(xy - self._last_xy))
        self._last_xy = xy


class Simulation:
    def __init__(self, task: dict, *, seed: int = 0, verbose: bool = True, viewer: bool = False,
                 realtime: bool = False, log: Callable[[str], None] | None = None, detector: str | None = None):
        import mujoco

        self._mj = mujoco
        self.task = task
        self.scene = build_scene(task, seed=seed)
        self.model, self.data = self.scene.compile()
        mujoco.mj_forward(self.model, self.data)
        self.hal = SimRobot(self.model, self.data, self.scene.robot_spec, seed=seed)
        self.verbose = verbose
        printer = log or ((lambda msg: print(msg, flush=True)) if verbose else None)
        # The robot is told its nominal start pose (like an AMCL initial pose); the scene jitters the
        # true pose and localisation has to find the difference.
        nominal = task.get("robot_start", {}).get("pose", self.scene.prior.start_pose)
        self.stack = RobotStack(self.hal, self.scene.robot_spec, self.scene.prior, self.scene.catalog,
                                initial_pose=nominal, log=printer, detector=detector)
        self.n_sub = max(1, round(self.stack.dt / self.model.opt.timestep))
        self.truth = GroundTruth(self.model, self.data, self.scene)
        self.realtime = realtime
        self.viewer = None
        self._wall_start = None
        self._sim_start = 0.0
        self._last_sync = 0.0
        if viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data, show_left_ui=False,
                                                   show_right_ui=False)
            self._configure_camera()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def _configure_camera(self) -> None:
        cam = self.viewer.cam
        cam.lookat[:] = [4.0, 3.4, 0.0]
        cam.distance, cam.azimuth, cam.elevation = 10.5, -90.0, -62.0

    def step(self) -> bool:
        """Advance one control period. Returns False if the viewer was closed."""
        mj = self._mj
        for _ in range(self.n_sub):
            mj.mj_step(self.model, self.data)
            self.hal.after_physics_step(self.model.opt.timestep)
            self.truth.after_step()
        self.stack.tick()
        if int(self.time * 5) != int((self.time - self.stack.dt) * 5):
            err = self.truth.robot_pose()[:2] - self.stack.pose[:2]
            self.truth.loc_errors.append(float(np.hypot(*err)))
        if self.viewer is not None:
            return self._sync_viewer()
        return True

    def _sync_viewer(self) -> bool:
        if not self.viewer.is_running():
            return False
        now = time.perf_counter()
        if self._wall_start is None:
            self._wall_start, self._sim_start = now, self.time
        if now - self._last_sync > 1 / 30:
            with self.viewer.lock():
                self._draw_overlays()
            self.viewer.sync()
            self._last_sync = now
        if self.realtime:
            lag = (self.time - self._sim_start) - (time.perf_counter() - self._wall_start)
            if lag > 0:
                time.sleep(min(lag, 0.05))
        return True

    def _draw_overlays(self) -> None:
        """Planned path (blue dots) and remembered objects (yellow rings) in the 3D view."""
        mj = self._mj
        scn = self.viewer.user_scn
        scn.ngeom = 0

        def add(kind, size, pos, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            mj.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, dtype=float),
                            np.asarray(pos, dtype=float), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
            scn.ngeom += 1

        path = self.stack.nav.path
        if path is not None:
            for p in path[::2]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], [p[0], p[1], 0.03], [0.2, 0.5, 1.0, 0.8])
        for obj in self.stack.world.objects.values():
            if not obj.held:
                add(mj.mjtGeom.mjGEOM_CYLINDER, [0.07, 0.003, 0], [obj.position[0], obj.position[1], obj.position[2] + 0.13],
                    [1.0, 0.85, 0.1, 0.7])
        pose = self.stack.pose
        add(mj.mjtGeom.mjGEOM_ARROW, [0.03, 0.03, 0.35], [pose[0], pose[1], 1.6], [0.1, 0.9, 0.3, 0.8])

    def run(self, name: str, generator, *, timeout: float = 300.0, on_tick: Callable | None = None):
        """Run a skill/mission generator to completion (or timeout). Returns its result."""
        self.stack.start(name, generator)
        deadline = self.time + timeout
        while self.stack.busy:
            if self.time > deadline:
                self.stack.cancel()
                return None
            if not self.step():
                self.stack.cancel()
                return None
            if on_tick:
                on_tick(self)
        return self.stack.result

    def idle(self, seconds: float) -> bool:
        end = self.time + seconds
        while self.time < end:
            if not self.step():
                return False
        return True

    def close(self) -> None:
        self.hal.close()
        if self.viewer is not None:
            self.viewer.close()
