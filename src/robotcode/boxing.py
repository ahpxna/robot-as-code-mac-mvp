"""Boxing skills for the humanoid: punches, guard, opponent tracking and a scripted fighter.

Everything here is onboard software: it reads only the robot's own sensors through the HAL
(joint encoders, IMU, foot pads, the head camera) and commands servo targets. The referee that
scores a match lives in robotcode.match and is the only part that looks at simulator state.

    head camera (RGB-D)  ->  OpponentTracker  ->  opponent head/body position (stance frame)
    BoxerBrain (style)   ->  footwork (BalanceController.move) + moves (MoveExecutor)
    MoveExecutor         ->  arm/waist targets  (arm IK aims each punch at the opponent)
    BalanceController    ->  leg targets that keep the robot standing under it all
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .balance import BalanceController, HumanoidBody, quat_to_mat
from .humanoid import ARM_JOINTS, UPPER_JOINTS, posture, rot_z, yaw_of


# ---- arm kinematics -----------------------------------------------------------------
class ArmIK:
    """Fist placement by damped least squares on the robot's own kinematic model (pelvis frame)."""

    def __init__(self, spec: dict, body: HumanoidBody):
        self.spec, self.body = spec, body
        m = body.model
        self.limits = {name: tuple(m.jnt_range[m.joint(name).id]) for name in UPPER_JOINTS}
        self._dof = {name: int(m.joint(name).dofadr[0]) for name in UPPER_JOINTS}

    def fist(self, q: dict[str, float], hand: str) -> np.ndarray:
        self.body.update(q)
        return self.body.site(f"{hand}_fist")[0]

    def solve(self, hand: str, target: np.ndarray, q0: dict[str, float], free: list[str],
              iterations: int = 40, damping: float = 2e-3) -> tuple[dict[str, float], float]:
        import mujoco

        q = dict(q0)
        m, d = self.body.model, self.body.data
        jac = np.zeros((3, m.nv))
        site = m.site(f"{hand}_fist").id
        error = math.inf
        for _ in range(iterations):
            pos = self.fist(q, hand)
            delta = np.asarray(target, dtype=float) - pos
            error = float(np.linalg.norm(delta))
            if error < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jac, None, site)
            j = jac[:, [self._dof[n] for n in free]]
            dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), delta)
            step = min(1.0, 0.35 / max(1e-9, float(np.abs(dq).max())))
            for i, n in enumerate(free):
                q[n] = float(np.clip(q[n] + step * dq[i], *self.limits[n]))
        return q, error


# ---- moves --------------------------------------------------------------------------
@dataclass
class ActiveMove:
    name: str
    hand: str
    start: float
    strike: dict[str, float]          # upper-body targets at full extension
    wind: dict[str, float] | None = None
    path: list[tuple[float, np.ndarray]] | None = None   # fist waypoints (time from start, pelvis frame)
    waist: tuple[float, float] = (0.0, 0.0)
    q: dict[str, float] | None = None # warm start for the per-tick IK
    retract_from: dict[str, float] | None = None
    retract_at: float | None = None
    landed: bool = False              # the glove felt contact


class MoveExecutor:
    """Runs one move at a time and snaps back to the guard.

    Straight punches drive the fist along straight lines (optionally through a via point, e.g.
    dipping to stomach height for a body shot), solving the arm IK every tick, and aim at the
    part of the target the opponent's guard covers least. Hooks wind the arm wide and twist the
    torso through. A punch snaps back as soon as its glove feels contact: a jammed arm would
    otherwise push its own robot over.
    """

    CANDIDATES = [np.array(v) for v in ((0, 0, 0), (0, 0.05, 0), (0, -0.05, 0), (0, 0, 0.05),
                                        (0, 0.05, 0.04), (0, -0.05, 0.04), (0, 0, -0.04))]

    def __init__(self, spec: dict, body: HumanoidBody, lead: str = "L"):
        self.spec = spec
        self.moves = spec.get("moves", {})
        self.ik = ArmIK(spec, body)
        self.lead = lead
        self.rear = "R" if lead == "L" else "L"
        self.guard_name = "guard"
        self.contact_force = spec.get("boxing", {}).get("glove_contact", 15.0)
        self.active: ActiveMove | None = None
        self.thrown: list[tuple[float, str]] = []

    def guard(self) -> dict[str, float]:
        return posture(self.spec, self.guard_name)

    @property
    def busy(self) -> bool:
        return self.active is not None

    def hand_of(self, name: str) -> str:
        return self.lead if self.moves[name]["hand"] == "lead" else self.rear

    def start(self, name: str, target: dict[str, np.ndarray], now: float,
              obstacles: np.ndarray | None = None) -> bool:
        """Throw `name` at a target in the pelvis frame ({'head': xyz, 'body': xyz}). `obstacles`
        are opponent points (pelvis frame) that may block the punch, e.g. gloves and forearms."""
        if self.active is not None or name not in self.moves:
            return False
        mv = self.moves[name]
        hand = self.hand_of(name)
        side = 1.0 if hand == "L" else -1.0
        aim = np.asarray(target[mv.get("aim", "head")], dtype=float)
        guard = self.guard()
        q = dict(guard)
        arm = [f"{hand}_{j}" for j in ARM_JOINTS]
        if mv.get("kind") == "hook":                        # swing in from the outside
            wind = dict(guard)
            for joint, deg in mv["wind"].items():
                key = "waist_yaw" if joint == "waist" else f"{hand}_{joint}"
                wind[key] = math.radians(deg * (side if joint == "waist" else 1.0))
            q["waist_yaw"] = math.radians(mv.get("waist", 0.0)) * side
            q[f"{hand}_elbow"] = math.radians(mv.get("elbow", 60))
            q[f"{hand}_shoulder_roll"] = math.radians(60)
            goal = aim + np.array([0.0, side * mv.get("side", 0.0), 0.0])
            q, _ = self.ik.solve(hand, goal, q, [f"{hand}_shoulder_pitch", f"{hand}_shoulder_roll",
                                                 f"{hand}_shoulder_yaw"])
            self.active = ActiveMove(name, hand, now, q, wind=wind)
        else:                                               # straight line from the guard to the target
            waist = math.radians(mv.get("waist", 0.0))
            q["waist_yaw"] = waist
            shoulder = self._shoulder(q, hand)
            aim = self._open_line(shoulder, aim, obstacles)
            line = aim - shoulder
            goal = aim + mv.get("reach", 0.0) * line / max(1e-6, float(np.linalg.norm(line)))
            fist = self.ik.fist(guard, hand)
            path = [(0.0, fist)]
            t = 0.0
            if "via" in mv:                                 # e.g. dip to stomach height first
                via = np.array(mv["via"], dtype=float) * np.array([1.0, side, 1.0])
                t += mv.get("via_s", 0.08)
                path.append((t, via))
            path.append((t + mv["extend_s"], goal))
            q[f"{hand}_elbow"] = math.radians(20)
            q, _ = self.ik.solve(hand, goal, q, arm)
            self.active = ActiveMove(name, hand, now, q, path=path, waist=(guard["waist_yaw"], waist),
                                     q=dict(guard))
        self.thrown.append((now, name))
        return True

    def _open_line(self, shoulder: np.ndarray, aim: np.ndarray, obstacles: np.ndarray | None) -> np.ndarray:
        """Candidate aim point whose punch line passes farthest from the opponent's guard."""
        if obstacles is None or len(obstacles) == 0:
            return aim
        best, best_gap = aim, -1.0
        for offset in self.CANDIDATES:
            point = aim + offset
            d = point - shoulder
            length = float(np.linalg.norm(d))
            u = d / max(1e-6, length)
            along = (obstacles - shoulder) @ u
            near = obstacles[(along > 0.05) & (along < length - 0.06)]       # between fist and target
            if len(near) == 0:
                return point
            rel = near - shoulder
            gap = float(np.min(np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)))
            if gap > best_gap + 0.01:
                best, best_gap = point, gap
        return best

    def _shoulder(self, q: dict[str, float], hand: str) -> np.ndarray:
        self.ik.body.update(q)
        return self.ik.body.data.body(f"{hand}_shoulder_pitch_link").xpos.copy()

    def _progress(self, a: ActiveMove, joints: dict[str, float]) -> float:
        """How far the measured fist has travelled from the guard towards the strike pose (0..1)."""
        start = self.ik.fist(self.guard(), a.hand)
        goal = self.ik.fist(a.strike, a.hand)
        now = self.ik.fist({j: joints.get(j, v) for j, v in a.strike.items()}, a.hand)
        span = goal - start
        return float(np.dot(now - start, span) / max(1e-6, float(span @ span)))

    def _strike_time(self, a: ActiveMove, mv: dict) -> float:
        if a.path is not None:
            return a.path[-1][0]
        return mv.get("wind_s", 0.0) + mv["extend_s"]

    def _straight(self, a: ActiveMove, t: float) -> dict[str, float]:
        """Fist on its straight-line path at time t (IK warm-started from the last tick)."""
        for (t0, p0), (t1, p1) in zip(a.path, a.path[1:]):
            if t <= t1:
                break
        u = min(1.0, max(0.0, (t - t0) / max(1e-6, t1 - t0)))
        point = p0 + u * (p1 - p0)
        total = a.path[-1][0]
        w = min(1.0, t / max(1e-6, total))
        a.q["waist_yaw"] = a.waist[0] + w * (a.waist[1] - a.waist[0])
        a.q, _ = self.ik.solve(a.hand, point, a.q, [f"{a.hand}_{j}" for j in ARM_JOINTS], iterations=8)
        return dict(a.q)

    def update(self, now: float, state) -> dict[str, float]:
        """Upper-body targets for this tick."""
        guard = self.guard()
        if self.active is None:
            return guard
        mv, a = self.moves[self.active.name], self.active
        t = now - a.start
        if a.retract_at is None:
            strike_t = self._strike_time(a, mv)
            # Contact counts once the fist is well on its way to the target (earlier it is a glove
            # clash, e.g. the opponent's punch landing on this glove).
            hit = (state.touch.get(f"{a.hand}_glove_hit", 0.0) > self.contact_force
                   and self._progress(a, state.joints) > 0.6)
            if hit:
                a.landed = True
            if hit or t >= strike_t + mv["hold_s"]:
                a.retract_at = now
                a.retract_from = {j: state.joints.get(j, v) for j, v in a.strike.items()}
            elif a.wind is not None:
                return dict(a.wind) if t < mv.get("wind_s", 0.0) else dict(a.strike)
            elif a.path is not None:
                return self._straight(a, t)
            else:
                return dict(a.strike)
        u = (now - a.retract_at) / mv["retract_s"]
        if u >= 1.0:
            self.active = None
            return guard
        s = u * u * (3 - 2 * u)
        return {j: a.retract_from[j] + s * (guard[j] - a.retract_from[j]) for j in guard}


# ---- opponent tracking ----------------------------------------------------------------
@dataclass
class OpponentEstimate:
    stamp: float
    head: np.ndarray            # pelvis frame at capture time, m
    body: np.ndarray
    points: np.ndarray          # opponent surface points (pelvis frame), subsampled
    head_odom: np.ndarray       # stance (odometry) frame
    distance: float             # pelvis -> opponent head, horizontal
    bearing: float              # relative to the pelvis heading, rad (+ = to the left)
    nearest: float = math.inf   # closest opponent point, horizontal from the pelvis (reliable up close)


class OpponentTracker:
    """Find the opponent in the head camera by its shell colour and back-project with depth.

    The camera pose in the pelvis frame comes from the robot's own kinematic model (waist yaw,
    measured joints); the pelvis pose in the odometry frame from the balance controller's leg
    odometry and the IMU.
    """

    def __init__(self, spec: dict, body: HumanoidBody, colour):
        self.spec, self.body = spec, body
        self.colour = np.asarray(colour[:3], dtype=float)
        cam = spec["sensors"]["head_camera"]
        self.width, self.height = cam["width"], cam["height"]
        b = spec["base"]
        self.head_radius = b["head"]["radius"]
        self.torso_below_top = b["head"]["above_torso"] + b["head"]["radius"]      # head top -> torso centre
        self.torso_depth = b["torso"]["half"][0] * 0.7
        f = 0.5 * self.height / math.tan(math.radians(cam["fovy_deg"]) / 2)
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # Camera frame: x right, y up, looking along -z.
        self._rays = np.stack([(u - (self.width - 1) / 2) / f, -(v - (self.height - 1) / 2) / f,
                               -np.ones_like(u, dtype=float)], axis=-1)
        self.estimate: OpponentEstimate | None = None

    def update(self, frame, state, balance: BalanceController) -> OpponentEstimate | None:
        rgb, depth = frame.rgb, frame.depth
        mask = _colour_mask(rgb.astype(float) / 255.0, self.colour) & (depth > 0.15) & (depth < 3.5)
        if mask.sum() < 30:
            return None
        pts_cam = self._rays[mask] * depth[mask][:, None]
        self.body.update(state.joints)
        self.body.camera_poses()
        cam = self.body.data.cam("head_camera")
        pts = pts_cam @ cam.xmat.reshape(3, 3).T + cam.xpos                # pelvis frame
        world = quat_to_mat(state.orientation)
        tilt = rot_z(-yaw_of(world)) @ world
        level = pts @ tilt.T                                                # gravity-aligned, pelvis heading
        top = float(np.percentile(level[:, 2], 99))
        # Head: the cap within 6 cm of the top (the guard gloves sit lower). The camera sees its
        # front, so the centre lies about half a radius further away.
        # Raised gloves can reach the same height, but they are in front of the head: keep the
        # farthest part of the top band.
        band = level[level[:, 2] > top - 0.08]
        r_band = np.linalg.norm(band[:, :2], axis=1)
        head_pts = band[r_band >= r_band.max() - 0.09]
        top = float(np.max(head_pts[:, 2]))
        head_lvl = np.array([*np.median(head_pts[:, :2], axis=0), top - self.head_radius])
        rng = float(np.linalg.norm(head_lvl[:2]))
        if rng > 1e-3:
            head_lvl[:2] += head_lvl[:2] / rng * self.head_radius * 0.5
        # Torso: the band below the guard; its visible front is half a torso depth before the centre.
        body_sel = (level[:, 2] < top - 0.30) & (level[:, 2] > top - 0.50)
        if body_sel.sum() > 10:
            body_lvl = np.array([*np.median(level[body_sel, :2], axis=0), top - self.torso_below_top])
            r_b = float(np.linalg.norm(body_lvl[:2]))
            body_lvl[:2] += body_lvl[:2] / max(1e-3, r_b) * self.torso_depth
        else:
            body_lvl = head_lvl - np.array([0.0, 0.0, self.torso_below_top - self.head_radius])
        head, body = tilt.T @ head_lvl, tilt.T @ body_lvl                   # back to the pelvis frame
        st = balance.status
        r_sp = rot_z(balance.heading()) @ tilt
        base = st.pelvis if st is not None else np.zeros(3)
        step = max(1, len(pts) // 400)
        self.estimate = OpponentEstimate(state.stamp, head, body, pts[::step], r_sp @ head + base,
                                         float(np.linalg.norm(head_lvl[:2])),
                                         math.atan2(head_lvl[1], head_lvl[0]),
                                         float(np.percentile(np.linalg.norm(level[:, :2], axis=1), 2)))
        return self.estimate


def _colour_mask(pix: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Pixels whose chromaticity matches the shell colour (robust to shading)."""
    total = pix.sum(axis=-1) + 1e-6
    chroma = pix / total[..., None]
    ref = colour / colour.sum()
    return (np.linalg.norm(chroma - ref, axis=-1) < 0.12) & (pix.max(axis=-1) > 0.15)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
