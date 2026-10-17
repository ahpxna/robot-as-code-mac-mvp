"""Legged locomotion: leg geometry and inverse kinematics, postures, and the gait controller.

A leg is coxa (yaw at the body) -> femur (pitch) -> tibia (pitch), as on most hexapod/octopod
robots. Postures are declared in the robot YAML as *foot positions*; joint angles always come
from the IK here, so the same code drives any leg dimensions or number of legs.

GaitController exposes the same interface as the wheeled DiffDriveController (command(v, w),
halt(), max_v, ...), so navigation, docking and every skill work unchanged on legs: the body
twist the planner asks for is turned into foot trajectories (alternating tetrapod by default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .math_utils import clamp

SEGMENTS = ("coxa", "femur", "tibia")


@dataclass
class Leg:
    name: str
    mount: np.ndarray           # (x, y, z) of the coxa joint in the base frame
    heading: float              # rad, direction the leg points at coxa = 0
    coxa: float
    femur: float
    tibia: float
    foot_radius: float
    limits: dict                # joint name -> (lo, hi) rad

    @property
    def joints(self) -> tuple[str, str, str]:
        return tuple(f"{self.name}_{s}" for s in SEGMENTS)

    def fk(self, q) -> np.ndarray:
        """Foot centre in the base frame for joint angles (coxa, femur, tibia)."""
        q1, q2, q3 = q
        r = self.coxa + self.femur * math.cos(q2) + self.tibia * math.cos(q2 + q3)
        z = self.femur * math.sin(q2) + self.tibia * math.sin(q2 + q3)
        a = self.heading + q1
        return self.mount + np.array([r * math.cos(a), r * math.sin(a), z])

    def ik(self, foot) -> tuple[tuple[float, float, float], bool]:
        """Joint angles that put the foot centre at `foot` (base frame), knee up.
        Returns (angles clamped to the joint limits, reachable)."""
        d = np.asarray(foot, dtype=float) - self.mount
        c, s = math.cos(self.heading), math.sin(self.heading)
        x, y = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        q1 = math.atan2(y, x)
        r = math.hypot(x, y) - self.coxa
        z = d[2]
        cos_knee = (r * r + z * z - self.femur ** 2 - self.tibia ** 2) / (2 * self.femur * self.tibia)
        reachable = -1.0 <= cos_knee <= 1.0
        q3 = -math.acos(clamp(cos_knee, -1.0, 1.0))
        q2 = math.atan2(z, r) - math.atan2(self.tibia * math.sin(q3), self.femur + self.tibia * math.cos(q3))
        q = (q1, q2, q3)
        clamped = tuple(clamp(v, *self.limits[j]) for v, j in zip(q, self.joints))
        reachable = reachable and all(abs(a - b) < 1e-6 for a, b in zip(q, clamped))
        return clamped, reachable


def is_legged(spec: dict) -> bool:
    return spec["base"].get("kind") == "legged"


def leg_mounts(spec: dict) -> dict[str, tuple[float, float, float]]:
    legs = spec["base"]["legs"]
    mounts = {name: tuple(v) for name, v in legs["mounts"].items()}
    if legs.get("mirror"):
        for name, (x, y, h) in list(mounts.items()):
            if name.startswith("L"):
                mounts["R" + name[1:]] = (x, -y, -h)
    return mounts


def leg_joint_names(spec: dict) -> list[str]:
    if not is_legged(spec):
        return []
    return [f"{leg}_{s}" for leg in leg_mounts(spec) for s in SEGMENTS]


def build_legs(spec: dict) -> list[Leg]:
    lg = spec["base"]["legs"]
    out = []
    for name, (x, y, heading) in leg_mounts(spec).items():
        limits = {f"{name}_{s}": tuple(math.radians(v) for v in lg["joints"][s]["range"]) for s in SEGMENTS}
        out.append(Leg(name, np.array([x, y, 0.0]), math.radians(heading), lg["coxa"]["length"],
                       lg["femur"]["length"], lg["tibia"]["length"], lg["foot"]["radius"], limits))
    return out


def posture_feet(spec: dict, legs: list[Leg], posture: str) -> tuple[dict[str, np.ndarray], float]:
    """Foot targets (base frame) and body height of a named posture."""
    p = spec["base"]["postures"][posture]
    height = p["body_height"]
    offsets = p.get("heading_offset", {})
    feet = {}
    for leg in legs:
        key = leg.name if leg.name in offsets else "L" + leg.name[1:]
        offset = math.radians(offsets.get(key, 0.0)) * (-1 if leg.name.startswith("R") else 1)
        a = leg.heading + offset
        reach = p["foot_reach"]
        xy = leg.mount[:2] + reach * np.array([math.cos(a), math.sin(a)])
        z = -(height - leg.foot_radius) + p.get("foot_lift", 0.0)
        feet[leg.name] = np.array([xy[0], xy[1], z])
    return feet, height


def posture_joints(spec: dict, posture: str) -> dict[str, float]:
    legs = build_legs(spec)
    feet, _ = posture_feet(spec, legs, posture)
    out = {}
    for leg in legs:
        q, ok = leg.ik(feet[leg.name])
        if not ok:
            raise ValueError(f"posture {posture!r}: foot of leg {leg.name} is out of reach")
        out.update(dict(zip(leg.joints, q)))
    return out


class GaitController:
    """Body twist (v, w) -> foot trajectories -> joint setpoints, with speed and acceleration limits.

    Stance feet slide backwards under the body (so the body moves forward); swing feet lift in an
    arc to a touchdown point half a stride ahead of neutral (Raibert-style placement). When the
    command goes to zero the robot finishes its step and stands on all legs.
    """

    WATCHDOG_S = 0.3

    def __init__(self, hal, spec: dict):
        self.hal = hal
        self.spec = spec
        base = spec["base"]
        self.legs = build_legs(spec)
        self.by_name = {leg.name: leg for leg in self.legs}
        gait = base["gait"]
        self.groups = [list(g) for g in gait["groups"]]
        self.period = gait["period"]
        self.step_height = gait["step_height"]
        lim = base["limits"]
        self.max_v, self.max_w = lim["max_linear_speed"], lim["max_angular_speed"]
        self.max_dv, self.max_dw = lim["max_linear_accel"], lim["max_angular_accel"]
        self._limits = (self.max_v, self.max_w, self.max_dv, self.max_dw)
        self.neutral, self.body_height = posture_feet(spec, self.legs, "stand")
        self.feet = {k: v.copy() for k, v in self.neutral.items()}
        self.v = self.w = 0.0
        self._target = (0.0, 0.0)
        self._last_cmd = -1.0
        self._no_forward_until = -1.0
        self.phase = 0.0
        self.stepping = False
        self.posture = base.get("initial_posture", "stand")
        # "walk": the gait owns the legs; "manual": a skill (e.g. folding) commands the joints directly.
        self.mode = "walk" if self.posture == "stand" else "manual"
        self._liftoff: dict[str, np.ndarray] = {}
        self._swinging = -1

    # ---- the drive interface shared with wheeled robots ----------------------------------
    def set_carrying(self, carrying: bool) -> None:
        v, w, dv, dw = self._limits
        self.max_v, self.max_w, self.max_dv, self.max_dw = (0.7 * v, 0.6 * w, dv, dw) if carrying else (v, w, dv, dw)

    def inhibit_forward(self, until: float) -> None:
        self._no_forward_until = until

    def command(self, v: float, w: float, dt: float) -> None:
        self._target = (v, w)
        self._last_cmd = self.hal.time()

    def halt(self) -> None:
        self._target = (0.0, 0.0)
        self.v = self.w = 0.0

    @property
    def standing(self) -> bool:
        """All feet down at their neutral stand positions."""
        return not self.stepping and self.mode == "walk"

    # ---- called by the stack every control tick ------------------------------------------
    def tick(self, dt: float) -> None:
        if self.mode != "walk":
            return
        now = self.hal.time()
        v, w = self._target if now - self._last_cmd <= self.WATCHDOG_S else (0.0, 0.0)
        if now < self._no_forward_until:
            v = min(v, 0.0)
        v = clamp(v, -self.max_v, self.max_v)
        w = clamp(w, -self.max_w, self.max_w)
        self.v += clamp(v - self.v, -self.max_dv * dt, self.max_dv * dt)
        self.w += clamp(w - self.w, -self.max_dw * dt, self.max_dw * dt)
        moving = abs(self.v) > 1e-3 or abs(self.w) > 1e-3
        settled = all(np.linalg.norm(self.feet[k] - self.neutral[k]) < 0.004 for k in self.feet)
        if moving or not settled:
            self.stepping = True
        if self.stepping:
            self._step(dt, moving, settled)
        self.hal.command_joints(self.joint_targets())

    def _step(self, dt: float, moving: bool, settled: bool) -> None:
        half = self.period / 2
        previous = self.phase
        self.phase = (self.phase + dt / self.period) % 1.0
        swinging = 0 if self.phase < 0.5 else 1
        if swinging != self._swinging or previous > self.phase:
            if not moving and settled and self._swinging != -1:
                self.stepping, self._swinging, self.phase = False, -1, 0.0
                return
            self._swinging = swinging
            self._liftoff = {k: self.feet[k].copy() for k in self.groups[swinging]}
        s = ((self.phase % 0.5) / 0.5)
        stride_rot = self.w * half / 2
        c, sn = math.cos(-self.w * dt), math.sin(-self.w * dt)
        for name, foot in self.feet.items():
            if name in self._liftoff:
                # Swing: arc to half a stride ahead of neutral.
                n = self.neutral[name]
                cr, sr = math.cos(stride_rot), math.sin(stride_rot)
                touchdown = np.array([cr * n[0] - sr * n[1] + self.v * half / 2, sr * n[0] + cr * n[1], n[2]])
                start = self._liftoff[name]
                blend = s * s * (3 - 2 * s)
                foot[:2] = start[:2] + (touchdown[:2] - start[:2]) * blend
                foot[2] = n[2] + self.step_height * math.sin(math.pi * s)
            else:
                # Stance: the foot stays on the ground while the body moves over it.
                x, y = foot[0], foot[1]
                foot[0], foot[1] = c * x - sn * y - self.v * dt, sn * x + c * y
                foot[2] = self.neutral[name][2]

    def joint_targets(self) -> dict[str, float]:
        out = {}
        for leg in self.legs:
            q, _ = leg.ik(self.feet[leg.name])
            out.update(dict(zip(leg.joints, q)))
        return out

    def reset_to(self, posture: str = "stand") -> None:
        """Resynchronise the gait after a skill moved the legs itself (e.g. after unfolding)."""
        self.neutral, self.body_height = posture_feet(self.spec, self.legs, posture)
        self.feet = {k: v.copy() for k, v in self.neutral.items()}
        self.stepping, self._swinging, self.phase = False, -1, 0.0
        self.v = self.w = 0.0


# ---------------------------------------------------------------------------
# Transformations between postures (e.g. walking <-> folded "logo")
# ---------------------------------------------------------------------------
def _move_feet(ctx, legs: list[Leg], start: dict, end: dict, duration: float, lift: float = 0.0, moving=None):
    """Generator: Cartesian foot interpolation (smoothstep) with an optional lift arc, via leg IK."""
    moving = set(moving or start)
    t0 = ctx.time
    while True:
        s = min(1.0, (ctx.time - t0) / duration)
        b = s * s * (3 - 2 * s)
        targets = {}
        for leg in legs:
            a, z = start[leg.name], end[leg.name]
            p = a + (z - a) * b
            if leg.name in moving and lift:
                p = p + np.array([0.0, 0.0, lift * math.sin(math.pi * s)])
            q, _ = leg.ik(p)
            targets.update(dict(zip(leg.joints, q)))
        ctx.hal.command_joints(targets)
        if s >= 1.0:
            return
        yield


def transform(ctx, target: str):
    """Generator: fold into the compact "logo" posture (target="fold") or unfold and stand up
    (target="stand"). The body is lowered onto the floor with the feet planted, then each tetrapod
    group lifts and tucks its feet in turn (and the reverse). Returns (ok, message)."""
    drive = ctx.drive
    spec = ctx.spec
    if drive.posture == target:
        return True, f"already in the {target} posture"
    legs = drive.legs
    stand, h_stand = posture_feet(spec, legs, "stand")
    folded, h_fold = posture_feet(spec, legs, "fold")
    # Crouched = stand footprint, body at fold height (feet move up in the base frame as the body lowers).
    crouch = {k: np.array([v[0], v[1], -(h_fold - drive.by_name[k].foot_radius)]) for k, v in stand.items()}
    if target == "fold":
        drive.halt()
        t0 = ctx.time
        while drive.stepping and ctx.time - t0 < 3.0:
            yield
        drive.mode = "manual"
        yield from _move_feet(ctx, legs, stand, crouch, 1.5)
        current = dict(crouch)
        for group in drive.groups:
            nxt = {k: (folded[k] if k in group else current[k]) for k in current}
            yield from _move_feet(ctx, legs, current, nxt, 0.9, lift=0.03, moving=group)
            current = nxt
        drive.posture = "fold"
    else:
        drive.mode = "manual"
        current = dict(folded)
        for group in reversed(drive.groups):
            nxt = {k: (crouch[k] if k in group else current[k]) for k in current}
            yield from _move_feet(ctx, legs, current, nxt, 0.9, lift=0.03, moving=group)
            current = nxt
        yield from _move_feet(ctx, legs, crouch, stand, 1.5)
        drive.reset_to("stand")
        drive.posture = "stand"
        drive.mode = "walk"
    t0 = ctx.time
    while ctx.time - t0 < 0.5:                  # let the servos settle before checking
        yield
    return verify_posture(ctx, target)


def verify_posture(ctx, posture: str, tolerance: float = 0.15) -> tuple[bool, str]:
    """Proprioceptive check: joint encoders vs. the posture, and body height from leg kinematics."""
    expected = posture_joints(ctx.spec, posture)
    q = ctx.state.joints
    worst = max(abs(q[j] - v) for j, v in expected.items())
    legs = ctx.drive.legs
    feet = np.array([leg.fk([q[j] for j in leg.joints]) for leg in legs])
    _, height = posture_feet(ctx.spec, legs, posture)
    est_height = float(-feet[:, 2].mean() + legs[0].foot_radius)
    boxes = np.array(ctx.kin.footprint_boxes(q))
    span = (float(boxes[:, 1].max() - boxes[:, 0].min()), float(boxes[:, 3].max() - boxes[:, 2].min()))
    ok = worst < tolerance and abs(est_height - height) < 0.02
    detail = (f"joints within {math.degrees(worst):.0f} deg of the {posture} posture, body at "
              f"{est_height * 100:.1f} cm, footprint {span[0] * 100:.0f} x {span[1] * 100:.0f} cm")
    return ok, detail
