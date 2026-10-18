"""Humanoid geometry: joint names, postures, stances, leg kinematics (pure numpy, no MuJoCo).

The MJCF itself is generated in robotcode.description (`base.kind: humanoid`); the balance
controller lives in robotcode.balance and the fighting behaviours in robotcode.boxing.

Leg kinematics follow the usual humanoid layout (Kajita et al., "Introduction to Humanoid
Robotics", ch. 2): hip yaw-roll-pitch intersecting at the hip, knee pitch, ankle pitch-roll
intersecting at the ankle. The analytic IK works in the textbook convention (+x roll, +y pitch,
+z yaw on both legs); `_SIGNS` maps it onto this robot's joint axes, which are mirrored so that a
positive angle means the same motion on both sides (flexion, abduction, toes up, toes out).
"""
from __future__ import annotations

import math

import numpy as np

SIDES = ("L", "R")
LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
UPPER_JOINTS = ("waist_yaw",) + tuple(f"{s}_{j}" for s in SIDES for j in ARM_JOINTS)

# robot joint = sign * textbook joint, per side
_SIGNS = {
    "L": {"hip_yaw": 1, "hip_roll": 1, "hip_pitch": -1, "knee": 1, "ankle_pitch": -1, "ankle_roll": 1},
    "R": {"hip_yaw": -1, "hip_roll": -1, "hip_pitch": -1, "knee": 1, "ankle_pitch": -1, "ankle_roll": -1},
}


def is_humanoid(spec: dict) -> bool:
    return spec["base"].get("kind") == "humanoid"


def humanoid_joint_names(spec: dict) -> list[str]:
    if not is_humanoid(spec):
        return []
    return [f"{s}_{j}" for s in SIDES for j in LEG_JOINTS] + list(UPPER_JOINTS)


def joint_type(name: str) -> str:
    """'L_hip_pitch' -> 'hip_pitch'."""
    return name[2:] if name[:2] in ("L_", "R_") else name


def side_sign(side: str) -> int:
    return 1 if side == "L" else -1


# ---- rotations -------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(rot: np.ndarray) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


# ---- postures and stances ------------------------------------------------------------
def posture(spec: dict, name: str) -> dict[str, float]:
    """Upper-body joint targets (rad) of a named posture. Keys like `elbow` apply to both
    sides, `L_elbow` / `R_elbow` to one side only; joints not listed are 0."""
    values = spec["base"]["postures"].get(name, {}) or {}
    out = {}
    for joint in UPPER_JOINTS:
        deg = values.get(joint, values.get(joint_type(joint), 0.0))
        out[joint] = math.radians(deg)
    return out


def leg_geometry(spec: dict) -> dict:
    legs = spec["base"]["legs"]
    return {"hip": np.array(legs["hip"], dtype=float), "thigh": legs["thigh"]["length"],
            "shank": legs["shank"]["length"], "ankle_height": 2 * legs["foot"]["half"][2]}


def stance(spec: dict, name: str) -> dict:
    """A stance: where each ankle stands relative to the pelvis (x, y, yaw) and pelvis height."""
    cfg = spec["base"]["stances"][name]
    feet = {side: (float(cfg[side][0]), float(cfg[side][1]), math.radians(cfg[side][2])) for side in SIDES}
    return {"feet": feet, "pelvis_height": float(cfg["pelvis_height"]), "lead": cfg.get("lead", "L")}


def initial(spec: dict) -> tuple[str, str]:
    init = spec["base"].get("initial", {})
    return init.get("stance", "square"), init.get("posture", "stand")


def pelvis_height(spec: dict, stance_name: str | None = None) -> float:
    return stance(spec, stance_name or initial(spec)[0])["pelvis_height"]


def stance_joints(spec: dict, stance_name: str, posture_name: str) -> dict[str, float]:
    """All joint angles for standing level in a stance with an upper-body posture."""
    st = stance(spec, stance_name)
    geo = leg_geometry(spec)
    q = posture(spec, posture_name)
    for side in SIDES:
        x, y, yaw = st["feet"][side]
        ankle = np.array([x, y, geo["ankle_height"] - st["pelvis_height"]])
        q.update(leg_ik(spec, side, ankle, rot_z(yaw))[0])
    return q


# ---- leg kinematics ----------------------------------------------------------------
def hip_position(spec: dict, side: str) -> np.ndarray:
    h = leg_geometry(spec)["hip"].copy()
    h[1] *= side_sign(side)
    return h


def leg_fk(spec: dict, side: str, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Ankle position and foot orientation in the pelvis frame from the leg's joint angles."""
    geo = leg_geometry(spec)
    t = {j: _SIGNS[side][j] * q[f"{side}_{j}"] for j in LEG_JOINTS}
    hip_rot = rot_z(t["hip_yaw"]) @ rot_x(t["hip_roll"]) @ rot_y(t["hip_pitch"])
    knee = hip_position(spec, side) + hip_rot @ np.array([0.0, 0.0, -geo["thigh"]])
    shank_rot = hip_rot @ rot_y(t["knee"])
    ankle = knee + shank_rot @ np.array([0.0, 0.0, -geo["shank"]])
    foot_rot = shank_rot @ rot_y(t["ankle_pitch"]) @ rot_x(t["ankle_roll"])
    return ankle, foot_rot


def leg_ik(spec: dict, side: str, ankle: np.ndarray, foot_rot: np.ndarray) -> tuple[dict[str, float], bool]:
    """Joint angles placing the ankle at `ankle` with foot orientation `foot_rot` (pelvis frame).

    Returns (joint angles, reachable). Out-of-reach targets are met with a straight knee.
    """
    geo = leg_geometry(spec)
    a, b = geo["thigh"], geo["shank"]
    r = foot_rot.T @ (hip_position(spec, side) - np.asarray(ankle, dtype=float))   # hip seen from the foot
    c = float(np.linalg.norm(r))
    cos_knee = (c * c - a * a - b * b) / (2 * a * b)
    reachable = -1.0 <= cos_knee <= 1.0
    knee = math.acos(min(1.0, max(-1.0, cos_knee)))
    ankle_roll = math.atan2(r[1], r[2])
    ankle_pitch = math.atan2(-a * math.sin(knee), b + a * math.cos(knee)) - math.atan2(r[0], math.hypot(r[1], r[2]))
    hip_rot = foot_rot @ rot_x(-ankle_roll) @ rot_y(-(knee + ankle_pitch))
    hip_yaw = math.atan2(-hip_rot[0, 1], hip_rot[1, 1])
    hip_roll = math.atan2(hip_rot[2, 1], math.hypot(hip_rot[0, 1], hip_rot[1, 1]))
    hip_pitch = math.atan2(-hip_rot[2, 0], hip_rot[2, 2])
    t = {"hip_yaw": hip_yaw, "hip_roll": hip_roll, "hip_pitch": hip_pitch, "knee": knee,
         "ankle_pitch": ankle_pitch, "ankle_roll": ankle_roll}
    return {f"{side}_{j}": _SIGNS[side][j] * t[j] for j in LEG_JOINTS}, reachable
