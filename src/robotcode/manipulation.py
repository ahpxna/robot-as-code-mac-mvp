"""Arm and gripper control.

Arm: joint-space trajectories with per-joint speed limits (the servo drives
track the moving setpoint), Cartesian targets via IK on the robot description,
and a closed-loop correction from the measured joint positions (servo sag under
load). Gripper: force-limited closing; the grasp is judged from the finger
force sensors and the encoder, never from the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import clamp


@dataclass
class GraspCheck:
    holding: bool
    width: float                # distance between the finger pads (m)
    forces: tuple[float, float]


class Arm:
    def __init__(self, ctx):
        self.ctx = ctx
        arm = ctx.spec["arm"]
        self.joints = ctx.kin.arm_joints
        self.max_speed = {j: arm["joints"][j]["max_speed"] for j in self.joints}
        self.stow_pose = {j: float(arm.get("stow", {}).get(j, 0.0)) for j in self.joints}
        self.carry_pose = {j: float(arm.get("carry", arm.get("stow", {})).get(j, 0.0)) for j in self.joints}
        self.setpoint: dict[str, float] | None = None

    def measured(self) -> dict[str, float]:
        return dict(self.ctx.state.joints)

    def gripper_position(self, q: dict[str, float] | None = None) -> np.ndarray:
        """Grasp-site position in the base frame (forward kinematics of measured joints)."""
        return self.ctx.kin.site("grasp_site", q or self.measured())[0]

    def move_joints(self, targets: dict[str, float], *, speed_scale: float = 1.0, tol: float = 0.005,
                    timeout: float = 12.0):
        """Generator: ramp the setpoints to `targets`; returns True once the joints have settled."""
        ctx = self.ctx
        if self.setpoint is None:
            self.setpoint = self.measured()
        goal = ctx.kin.clamp({**self.setpoint, **targets})
        start = ctx.time
        reached_at = None
        while ctx.time - start < timeout:
            for j in self.joints:
                step = self.max_speed[j] * speed_scale * ctx.dt
                self.setpoint[j] += clamp(goal[j] - self.setpoint[j], -step, step)
            ctx.hal.command_joints(self.setpoint)
            q = self.measured()
            if all(abs(self.setpoint[j] - goal[j]) < 1e-9 for j in self.joints):
                reached_at = reached_at if reached_at is not None else ctx.time
                err = max(abs(q[j] - goal[j]) for j in targets)
                if err < tol or ctx.time - reached_at > 1.5:
                    return err < 3 * tol
            yield
        return False

    def move_to(self, target_base: np.ndarray, *, order: tuple[str, ...] = (), speed_scale: float = 1.0,
                tol: float = 0.006):
        """Generator: put the grasp site at `target_base` (base frame). Joints listed in `order` move
        one after another (collision-aware sequencing), the rest together at the end.
        Returns the final Cartesian error (m), or None if the target is out of reach."""
        q_goal, err = self.ctx.kin.ik(target_base, self.measured())
        if err > 0.01:
            return None
        for j in order:
            yield from self.move_joints({j: q_goal[j]}, speed_scale=speed_scale)
        yield from self.move_joints(q_goal, speed_scale=speed_scale)
        # Closed loop: the lift sags a few mm under load; correct from measured kinematics.
        for _ in range(2):
            error = np.asarray(target_base) - self.gripper_position()
            if np.linalg.norm(error) < tol:
                break
            corrected, _ = self.ctx.kin.ik(self.gripper_position(self.setpoint) + error, self.setpoint)
            yield from self.move_joints(corrected, speed_scale=0.5)
        return float(np.linalg.norm(np.asarray(target_base) - self.gripper_position()))

    def stow(self, holding: bool = False):
        """Generator: retract first, then move vertically, so nothing sweeps across a table."""
        pose = self.carry_pose if holding else self.stow_pose
        retract = {j: pose[j] for j in self.joints if j != "lift"}
        yield from self.move_joints(retract)
        return (yield from self.move_joints(pose))


class Gripper:
    def __init__(self, ctx):
        self.ctx = ctx
        g = ctx.spec["gripper"]
        self.max_opening = g["max_opening"]
        self.travel = g["max_opening"] / 2 - 0.002
        self.threshold = g.get("contact_threshold_n", 0.8)

    def width(self) -> float:
        return self.max_opening - 2.0 * self.ctx.state.gripper_position

    def check(self) -> GraspCheck:
        st = self.ctx.state
        forces = st.finger_forces
        holding = min(forces) > self.threshold and st.gripper_position < self.travel - 0.004
        return GraspCheck(holding, self.width(), forces)

    def open(self, width: float | None = None, timeout: float = 1.5):
        """Generator: open fully, or only to `width` (narrower sweep in clutter)."""
        ctx = self.ctx
        target = 0.0 if width is None else float(np.clip((self.max_opening - width) / 2, 0.0, self.travel))
        ctx.hal.command_gripper(target)
        start = ctx.time
        while ctx.time - start < timeout and abs(ctx.state.gripper_position - target) > 0.002:
            yield
        return abs(ctx.state.gripper_position - target) <= 0.004

    def close(self, timeout: float = 2.5):
        """Generator: close until the fingers stop moving; returns a GraspCheck."""
        ctx = self.ctx
        ctx.hal.command_gripper(self.travel)
        start = ctx.time
        last, still_since = ctx.state.gripper_position, None
        while ctx.time - start < timeout:
            yield
            pos = ctx.state.gripper_position
            if abs(pos - last) < 2e-4 and ctx.time - start > 0.3:
                still_since = still_since if still_since is not None else ctx.time
                if ctx.time - still_since > 0.25:
                    break
            else:
                still_since = None
            last = pos
        return self.check()
