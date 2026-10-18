"""Balance and footwork for a position-controlled humanoid, from onboard sensors only.

Inputs each control tick: IMU orientation and gyro, joint encoders and speeds, heel/toe force
pads. Output: a target for every servo.

    planned footsteps        ->  centre-of-pressure (ZMP) plan  ->  capture-point (DCM) reference
    encoders + IMU tilt      ->  feet, centre of mass (COM) and its velocity in the stance frame
    capture-point error      ->  pelvis shift (ankle strategy), on top of the planned COM motion
    pelvis + feet + lean     ->  analytic leg IK  ->  servo targets

The capture point xi = c + c_dot / omega (omega = sqrt(g / z_com)) is where the robot would have
to put its centre of pressure to come to rest (Pratt et al. 2006). For a planned centre-of-
pressure path p(t) the bounded capture-point reference is solved backwards in time from the last
footstep (Englsberger et al., "Three-dimensional bipedal walking control based on divergent
component of motion", 2015); following it moves the COM smoothly onto each stance foot before the
other foot lifts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .humanoid import (SIDES, hip_position, humanoid_joint_names, joint_type, leg_fk, leg_geometry, leg_ik, posture, rot_x, rot_y, rot_z, stance,
                       stance_joints, yaw_of)

GRAVITY = 9.81


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class HumanoidBody:
    """The robot's own kinematic model (COM, sites), built from its description like a URDF."""

    def __init__(self, spec: dict):
        import mujoco

        from .description import robot_xml

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(robot_xml(spec, floating=False))
        self.data = mujoco.MjData(self.model)
        self._base = self.model.body("base").id
        self.mass = float(self.model.body_subtreemass[self._base])
        self._qadr = {self.model.joint(j).name: int(self.model.joint(j).qposadr[0]) for j in range(self.model.njnt)}

    def update(self, q: dict[str, float]) -> None:
        for j, v in q.items():
            self.data.qpos[self._qadr[j]] = v
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_comPos(self.model, self.data)

    def camera_poses(self) -> None:
        """Camera frames are only computed on request (after update)."""
        self._mj.mj_camlight(self.model, self.data)

    def com(self) -> np.ndarray:
        """Whole-body centre of mass in the pelvis frame (after update)."""
        return self.data.subtree_com[self._base].copy()

    def site(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        s = self.data.site(name)
        return s.xpos.copy(), s.xmat.reshape(3, 3).copy()

    def holding_torques(self, q: dict[str, float], up: np.ndarray, foot_forces: dict[str, float]) -> dict[str, float]:
        """Joint torques that hold the robot still in configuration q: gravity on every link plus
        the floor pushing up on each foot with the given force (N). `up` is the vertical in the
        pelvis frame. Static equilibrium per link chain: tau = g(q) - sum J_foot^T f_foot."""
        m, d = self.model, self.data
        self.update(q)
        m.opt.gravity[:] = -GRAVITY * up
        d.qvel[:] = 0.0
        bias = np.zeros(m.nv)
        self._mj.mj_rne(m, d, 0, bias)
        jacp = np.zeros((3, m.nv))
        for side, force in foot_forces.items():
            if force <= 0.0:
                continue
            body = m.body(f"{side}_foot")
            self._mj.mj_jac(m, d, jacp, None, d.xpos[body.id], body.id)
            bias -= jacp.T @ (force * up)
        return {name: float(bias[m.joint(name).dofadr[0]]) for name in q}


@dataclass
class Step:
    side: str                             # swing foot
    goal: np.ndarray                      # ankle placement in S: x, y, yaw
    lift_at: float = 0.0                  # controller clock
    land_at: float = 0.0
    start: np.ndarray | None = None
    planned_at: float = 0.0


@dataclass
class Segment:
    """Centre of pressure moving linearly from a to b during [t0, t1]."""
    t0: float
    t1: float
    a: np.ndarray
    b: np.ndarray


@dataclass
class BalanceStatus:
    com: np.ndarray                       # COM in the stance frame (x, y, z), m
    com_velocity: np.ndarray              # horizontal, m/s
    capture_point: np.ndarray             # horizontal, stance frame
    capture_ref: np.ndarray               # planned capture point
    support: list[str]                    # feet carrying load
    tilt_deg: float                       # pelvis tilt from vertical (IMU)
    fallen: bool
    pelvis: np.ndarray                    # estimated pelvis position in the stance frame
    loads: dict[str, float] = field(default_factory=dict)


class BalanceController:
    """Stance balance and footwork for one humanoid.

    The stance frame S is fixed to the floor (an odometry frame): x forward, origin under the
    pelvis of the initial stance. Foot placements in S are known (planned, then corrected by where
    each foot really landed), so the pelvis pose in S follows from the leg kinematics of the loaded
    feet plus the IMU tilt, and the COM from the robot's own mass model.
    """

    def __init__(self, spec: dict, rate_hz: float):
        self.spec = spec
        self.dt = 1.0 / rate_hz
        self.body = HumanoidBody(spec)
        g = spec.get("balance", {})
        self.k_dcm = g.get("dcm_gain", 1.0)               # pelvis shift per metre of capture-point error
        self.k_trim = g.get("com_trim", 1.0)              # 1/s: slow trim of what the mass model gets wrong
        self.k_sag = g.get("sag_integral", 4.0)           # 1/s: removes the joint sag under load
        self.v_alpha = 1.0 - math.exp(-2 * math.pi * g.get("velocity_filter_hz", 12.0) * self.dt)
        self.contact_force = g.get("contact_force", 25.0)
        self.fall_tilt = math.radians(g.get("fall_tilt_deg", 45.0))
        self.max_trim = g.get("max_trim", 0.06)
        self.k_foot = g.get("foot_flat_gain", 1.0)        # rad of correction per rad of measured foot tilt
        self.max_foot_fix = math.radians(g.get("max_foot_fix_deg", 12.0))
        self.calibration_s = g.get("calibration_s", 1.2)   # stand still this long after start-up (IMU calibration)
        self.feedforward = g.get("com_feedforward", 1.0)  # pelvis shift per metre of COM shift wanted
        self.ff_gain = g.get("torque_feedforward", 1.0)   # share of the holding torque fed forward
        self.kp = {j: spec["base"]["joints"][joint_type(j)]["kp"] for j in humanoid_joint_names(spec)}
        gait = spec.get("gait", {})
        self.t_shift = gait.get("shift_s", 0.4)            # centre of pressure moves to the stance foot
        self.t_swing = gait.get("swing_s", 0.4)
        self.lift = gait.get("lift", 0.05)
        self.inset = gait.get("com_inset", 0.02)           # pressure point inside the stance foot, towards the swing foot
        self.lift_tolerance = gait.get("lift_tolerance", 0.04)   # m of capture-point error allowed at lift-off
        self.max_wait = gait.get("max_wait_s", 2.0)
        geo = leg_geometry(spec)
        foot = spec["base"]["legs"]["foot"]
        self.ankle_height = geo["ankle_height"]
        self.max_leg = 0.97 * (geo["thigh"] + geo["shank"])
        self.sole = np.array([foot.get("forward", 0.0), 0.0, -self.ankle_height])
        self.foot_half = np.array(foot["half"][:2])
        self.stance_name, self.posture_name, self.lead = None, None, "L"
        self.height = 0.0
        self.feet: dict[str, np.ndarray] = {}            # ankle placements in S: x, y, yaw
        self.upper: dict[str, float] = {}
        self.com_offset = np.zeros(2)                     # desired COM relative to the middle of the soles
        self.lean = np.zeros(2)                           # commanded torso lean (pitch, roll), rad
        self.steps: list[Step] = []                       # queued; steps[0] is the next / current one
        self.clock = 0.0
        self._plan: list[Segment] = []                    # centre-of-pressure plan (empty = stand)
        self._nominal_yaw = 0.0                           # mean foot yaw of the stance when the pelvis faces x
        self._com_ref: np.ndarray | None = None
        self._velocity = np.zeros(2)
        self._trim = np.zeros(2)
        self._sag = np.zeros(2)                           # integral of the pelvis attitude error (pitch, roll)
        self._imu_bias = np.zeros(2)                      # learned IMU attitude error (pitch, roll), rad
        self.status: BalanceStatus | None = None

    # ---- commands ----------------------------------------------------------------
    def set_stance(self, stance_name: str, posture_name: str | None = None) -> None:
        st = stance(self.spec, stance_name)
        self.stance_name = stance_name
        self.height = st["pelvis_height"]
        self.feet = {side: np.array(st["feet"][side], dtype=float) for side in SIDES}
        self._nominal_yaw = _mean_angle([self.feet[s][2] for s in SIDES])
        mid = np.mean([self.feet[s][:2] for s in SIDES], axis=0)
        self._shape = {s: np.array([*(self.feet[s][:2] - mid), self.feet[s][2]]) for s in SIDES}
        self.lead = st["lead"]
        if posture_name:
            self.set_posture(posture_name)

    def set_posture(self, posture_name: str) -> None:
        self.posture_name = posture_name
        self.upper = posture(self.spec, posture_name)

    def nominal_joints(self) -> dict[str, float]:
        return stance_joints(self.spec, self.stance_name, self.posture_name)

    def sole_centre(self, side: str, feet: dict | None = None) -> np.ndarray:
        x, y, yaw = (feet or self.feet)[side][:3]
        return np.array([x, y]) + rot_z(yaw)[:2, :2] @ self.sole[:2]

    def support_centre(self, feet: dict | None = None) -> np.ndarray:
        return np.mean([self.sole_centre(s, feet) for s in SIDES], axis=0)

    def heading(self, feet: dict | None = None) -> float:
        """Direction the stance faces in S (the pelvis yaw it is built around)."""
        return _wrap(_mean_angle([f[2] for f in (feet or self.feet).values()]) - self._nominal_yaw)

    @property
    def stepping(self) -> bool:
        return bool(self.steps)

    @property
    def ready(self) -> bool:
        """Start-up calibration (standing still) is over."""
        return self.clock >= self.calibration_s

    def move(self, forward: float = 0.0, left: float = 0.0, turn: float = 0.0) -> None:
        """Queue the two steps that carry the whole stance by (forward, left) in its own heading
        frame and turn it by `turn` rad about its centre. Boxing footwork: the foot on the side of
        travel moves first and the other one follows (step-drag), so the stance never crosses."""
        feet = {s: self.feet[s].copy() for s in SIDES}
        for st in self.steps:                              # plan from where queued steps will end
            feet[st.side] = st.goal.copy()
        centre = np.mean([f[:2] for f in feet.values()], axis=0)
        heading = _wrap(self.heading(feet) + turn)
        shift = rot_z(self.heading(feet))[:2, :2] @ np.array([forward, left])
        # Every move re-lays the configured stance around the new centre, so the small errors of
        # each landing never accumulate into a narrower (or crossed) stance.
        rot = rot_z(heading)[:2, :2]
        goals = {s: np.array([*(centre + shift + rot @ self._shape[s][:2]), _wrap(self._shape[s][2] + heading)])
                 for s in SIDES}
        rear = "R" if self.lead == "L" else "L"
        if abs(left) > abs(forward):
            order = ["L", "R"] if left > 0 else ["R", "L"]
        elif abs(turn) > 1e-6 and abs(forward) < 1e-6:
            order = ["L", "R"] if turn > 0 else ["R", "L"]
        else:
            order = [self.lead, rear] if forward >= 0 else [rear, self.lead]
        for s in order:
            self.steps.append(Step(s, goals[s], planned_at=self.clock))
        self._replan()

    def stop_stepping(self) -> None:
        """Drop the steps that have not started; a foot in the air still lands."""
        self.steps = [st for st in self.steps if st.lift_at <= self.clock]
        self._replan()

    def _replan(self, recover: bool = False) -> None:
        """Centre-of-pressure plan for the queued steps: before each swing it moves onto the stance
        foot; after the last landing it returns to the middle of the feet. `recover` first brings
        the pressure back to the middle of both feet (used when a lift-off had to be postponed)."""
        start = self.clock
        p = self._pressure_ref(start)
        feet = {s: self.feet[s].copy() for s in SIDES}
        plan: list[Segment] = []
        t = start
        if recover:
            centre = self.support_centre(feet) + self.com_offset
            plan.append(Segment(t, t + self.t_shift, p, centre))
            p, t = centre, t + self.t_shift
        for st in self.steps:
            if st.lift_at and st.lift_at <= self.clock:     # already in the air: keep its timing
                other = "R" if st.side == "L" else "L"
                q = self._stance_point(other, st.side, feet)
                plan.append(Segment(t, st.land_at, q, q))
                p, t = q, st.land_at
                feet[st.side] = st.goal.copy()
                continue
            other = "R" if st.side == "L" else "L"
            q = self._stance_point(other, st.side, feet)
            plan.append(Segment(t, t + self.t_shift, p, q))
            st.lift_at = t + self.t_shift
            st.land_at = st.lift_at + self.t_swing
            plan.append(Segment(st.lift_at, st.land_at, q, q))
            feet[st.side] = st.goal.copy()
            p, t = q, st.land_at
        end = self.support_centre(feet) + self.com_offset
        plan.append(Segment(t, t + self.t_shift, p, end))
        self._plan = plan

    def _stance_point(self, stance_side: str, swing_side: str, feet: dict) -> np.ndarray:
        c = self.sole_centre(stance_side, feet)
        towards = self.sole_centre(swing_side, feet) - c
        return c + self.inset * towards / max(1e-6, float(np.linalg.norm(towards)))

    def _pressure_ref(self, t: float) -> np.ndarray:
        for seg in self._plan:
            if seg.t0 <= t < seg.t1:
                a = (t - seg.t0) / max(1e-9, seg.t1 - seg.t0)
                return seg.a + a * (seg.b - seg.a)
        if self._plan and t >= self._plan[-1].t1:
            return self._plan[-1].b.copy()
        return self.support_centre() + self.com_offset

    def _capture_ref(self, t: float, omega: float) -> np.ndarray:
        """Bounded capture-point reference for the pressure plan, solved backwards from its end."""
        if not self._plan or t >= self._plan[-1].t1:
            return self._pressure_ref(t)
        xi_end = self._plan[-1].b.copy()
        for seg in reversed(self._plan):
            v = (seg.b - seg.a) / max(1e-9, seg.t1 - seg.t0)
            if seg.t0 <= t < seg.t1:
                p = seg.a + v * (t - seg.t0)
                return p + v / omega + (xi_end - seg.b - v / omega) * math.exp(omega * (t - seg.t1))
            xi_end = seg.a + v / omega + (xi_end - seg.b - v / omega) * math.exp(-omega * (seg.t1 - seg.t0))
        return self._pressure_ref(t)

    def _feet_flat(self, tilt: np.ndarray, fk: dict, limit: float = math.radians(2.0)) -> bool:
        return all(max(abs(a) for a in _foot_tilt(tilt, fk[s][1])) < limit for s in SIDES)

    # ---- control tick --------------------------------------------------------------
    def update(self, state, upper: dict[str, float] | None = None) -> dict[str, float]:
        self.clock += self.dt
        q = state.joints
        world = quat_to_mat(state.orientation)
        self.body.update(q)
        load = {s: state.touch.get(f"{s}_heel_force", 0.0) + state.touch.get(f"{s}_toe_force", 0.0) for s in SIDES}
        support = [s for s in SIDES if load[s] > self.contact_force]
        weights = np.array([load[s] + 5.0 for s in SIDES])
        weights /= weights.sum()
        fk = {s: leg_fk(self.spec, s, q) for s in SIDES}

        # Kinematic-inertial fusion: with both feet flat on the floor the legs give the pelvis
        # tilt without the IMU's mounting/attitude error; learn that error while standing quietly.
        kin = sum(w * (rot_z(self.feet[s][2]) @ fk[s][1].T) for w, s in zip(weights, SIDES))
        reached = np.array([math.atan2(-kin[2, 0], math.hypot(kin[2, 1], kin[2, 2])), math.atan2(kin[2, 1], kin[2, 2])])
        imu_tilt = rot_z(-yaw_of(world)) @ world
        imu = np.array([math.atan2(-imu_tilt[2, 0], math.hypot(imu_tilt[2, 1], imu_tilt[2, 2])),
                        math.atan2(imu_tilt[2, 1], imu_tilt[2, 2])])
        total = load["L"] + load["R"]
        quiet = (total > 0.5 * self.body.mass * GRAVITY and min(load.values()) > 0.15 * total
                 and float(np.linalg.norm(state.gyro)) < 0.3 and not (self.steps and self.clock >= self.steps[0].lift_at))
        # A large disagreement means the feet are not flat (rolling on an edge): learn nothing then.
        if quiet and (self.clock < self.calibration_s
                      or float(np.max(np.abs(imu - reached - self._imu_bias))) < math.radians(1.5)):
            rate = 4.0 if self.clock < self.calibration_s else 0.3   # 1/s: fast while calibrating at start-up
            self._imu_bias += min(1.0, rate * self.dt) * (np.clip(imu - reached, -0.05, 0.05) - self._imu_bias)
        corrected = imu - self._imu_bias
        tilt = rot_y(corrected[0]) @ rot_x(corrected[1])             # roll/pitch only
        yaw = sum(w * _wrap(self.feet[s][2] - yaw_of(tilt @ fk[s][1])) for w, s in zip(weights, SIDES))
        r_sp = rot_z(yaw) @ tilt                                     # pelvis orientation in S
        com_p = self.body.com()
        com = sum(w * (np.array([self.feet[s][0], self.feet[s][1], self.ankle_height]) + r_sp @ (com_p - fk[s][0]))
                  for w, s in zip(weights, SIDES))
        pelvis = com - r_sp @ com_p

        # COM velocity by differentiating the kinematics (joint speeds + gyro), not the noisy position:
        # c = foot + R (com - ankle)  =>  c_dot = R (omega x r + dr/dq q_dot)
        eps = 0.01
        q_next = {j: v + eps * state.joint_velocities.get(j, 0.0) for j, v in q.items()}
        self.body.update(q_next)
        com_next = self.body.com()
        velocity = np.zeros(3)
        for w, s in zip(weights, SIDES):
            r = com_p - fk[s][0]
            dr = (com_next - leg_fk(self.spec, s, q_next)[0] - r) / eps
            velocity += w * (r_sp @ (np.cross(state.gyro, r) + dr))
        c = com[:2]
        self._velocity += self.v_alpha * (velocity[:2] - self._velocity)
        omega = math.sqrt(GRAVITY / max(0.3, com[2]))
        xi = c + self._velocity / omega

        # Planned motion: capture-point reference and the COM that follows it.
        xi_ref = self._capture_ref(self.clock, omega)
        if self._com_ref is None:
            self._com_ref = c.copy()
        self._com_ref = self._com_ref + self.dt * omega * (xi_ref - self._com_ref)
        tilt_angle = math.acos(max(-1.0, min(1.0, world[2, 2])))
        self.status = BalanceStatus(com, self._velocity.copy(), xi, xi_ref, support, math.degrees(tilt_angle),
                                    tilt_angle > self.fall_tilt, pelvis, load)

        # Footwork: swing feet follow a smooth arc between their lift-off and touchdown times.
        feet = {side: self.feet[side].copy() for side in SIDES}
        lift = {side: 0.0 for side in SIDES}
        # Do not lift a foot while still off balance (e.g. just after being hit): wait, re-planning
        # the weight shift, until the capture point is back on its reference.
        err = float(np.linalg.norm(xi - xi_ref))
        shifting = self.steps and self.steps[0].start is None
        if shifting and ((self.clock >= self.steps[0].lift_at - self.dt and err > self.lift_tolerance)
                         or err > 1.5 * self.lift_tolerance):  # going wrong during the weight shift: abort early
            if self.clock - self.steps[0].planned_at < self.max_wait:
                self._replan(recover=True)              # weight back over both feet, then try again
            else:
                self.steps = []                         # give up on this move, just stand
                self._replan(recover=True)
        if self.steps and self.steps[0].start is not None and load[self.steps[0].side] > self.contact_force:
            st = self.steps[0]                          # touched down early: the step is over
            if (self.clock - st.lift_at) > 0.5 * (st.land_at - st.lift_at):
                st.land_at = self.clock
        landed_any = False
        while self.steps and self.clock >= self.steps[0].land_at:
            st = self.steps.pop(0)
            other = "R" if st.side == "L" else "L"
            # Plan on from where the foot really landed, measured from the stance foot.
            stance_ankle = np.array([*self.feet[other][:2], self.ankle_height])
            landed = stance_ankle + r_sp @ (fk[st.side][0] - fk[other][0])
            landed_yaw = self.feet[other][2] + _wrap(yaw_of(tilt @ fk[st.side][1]) - yaw_of(tilt @ fk[other][1]))
            if float(np.linalg.norm(landed[:2] - st.goal[:2])) < 0.15:
                st.goal = np.array([landed[0], landed[1], landed_yaw])
            self.feet[st.side] = st.goal.copy()
            feet[st.side] = st.goal.copy()
            landed_any = True
        if landed_any:
            self._replan()                              # the rest of the plan starts from the real feet
        if self.steps and self.clock >= self.steps[0].lift_at:
            st = self.steps[0]
            if st.start is None:
                st.start = self.feet[st.side].copy()
            a = min(1.0, (self.clock - st.lift_at) / max(1e-9, st.land_at - st.lift_at))
            b = min(1.0, max(0.0, (a - 0.15) / 0.7))          # lift first, travel, then set down
            s_ = b * b * (3 - 2 * b)
            feet[st.side] = st.start + s_ * np.array([*(st.goal[:2] - st.start[:2]), _wrap(st.goal[2] - st.start[2])])
            lift[st.side] = self.lift * math.sin(math.pi * a)

        # Ankle strategy: put the pelvis where the planned COM wants it, push it against the
        # capture-point error and slowly trim what the mass model gets wrong.
        centre = self.support_centre(feet)
        nominal = np.mean([f[:2] for f in feet.values()], axis=0)
        # The integrators only learn static offsets: they are frozen while the robot is visibly off
        # balance or a foot is rolling on an edge, where the kinematic estimates are not valid.
        settled = float(np.linalg.norm(xi - xi_ref)) < 0.04 and self._feet_flat(tilt, fk)
        if settled:
            self._trim = np.clip(self._trim + self.dt * self.k_trim * (self._com_ref - c), -self.max_trim, self.max_trim)
        shift = (nominal + self.feedforward * (self._com_ref - centre) + self._trim
                 - self.k_dcm * (xi - xi_ref))
        pelvis_cmd = np.array([shift[0], shift[1], self.height])

        # Joint sag: the pelvis attitude reached through the loaded legs (feet flat) versus the one
        # commanded; integrate the difference into the command so the legs hold the pelvis level.
        if support and settled:
            self._sag = np.clip(self._sag + self.dt * self.k_sag * (self.lean - reached), -0.06, 0.06)
        lean = self.lean + self._sag
        r_cmd = rot_z(self.heading(feet)) @ rot_y(lean[0]) @ rot_x(lean[1])

        # Keep both legs inside their reach: pull the pelvis back towards the nominal if needed.
        for _ in range(8):
            if all(np.linalg.norm(r_cmd.T @ (np.array([f[0], f[1], self.ankle_height + lift[sd]]) - pelvis_cmd)
                                  - hip_position(self.spec, sd)) <= self.max_leg for sd, f in feet.items()):
                break
            pelvis_cmd[:2] = nominal + 0.8 * (pelvis_cmd[:2] - nominal)

        targets = dict(self.upper)
        if upper:
            targets.update(upper)
        # A foot that rolls onto its heel, toe or edge (the whole robot tipping) is pushed back flat:
        # its measured tilt in the world is fed back into its orientation target. A flat foot (the
        # body swaying about the ankle) gets no correction, so the ankle stiffness stays intact.
        for side in SIDES:
            x, y, foot_yaw = feet[side]
            ankle = np.array([x, y, self.ankle_height + lift[side]]) - pelvis_cmd
            pitch_f, roll_f = _foot_tilt(tilt, fk[side][1])
            fix = np.clip(-self.k_foot * np.array([pitch_f, roll_f]), -self.max_foot_fix, self.max_foot_fix)
            if lift[side] > 0.0:
                fix[:] = 0.0                            # a swinging foot is placed by the plan
            foot = rot_z(foot_yaw) @ rot_y(fix[0]) @ rot_x(fix[1])
            joints, _ = leg_ik(self.spec, side, r_cmd.T @ ankle, r_cmd.T @ foot)
            targets.update(joints)

        # Feed-forward: servos are springs, so under load they sag by torque / kp. Offset each
        # target by the torque that holding the body needs (gravity + the floor under the feet,
        # shared as the feet are loaded; nothing on a swinging foot).
        # The split between the feet follows the planned centre of pressure (smooth, unlike the
        # measured loads, which spike at every touchdown).
        cl, cr = self.sole_centre("L", feet), self.sole_centre("R", feet)
        across = cl - cr
        left = float(np.clip(np.dot(self._pressure_ref(self.clock) - cr, across) / max(1e-6, across @ across), 0, 1))
        share = {"L": left, "R": 1.0 - left}
        if self.steps and self.clock >= self.steps[0].lift_at:
            share[self.steps[0].side] = 0.0
            share["R" if self.steps[0].side == "L" else "L"] = 1.0
        forces = {s: self.body.mass * GRAVITY * share[s] for s in SIDES}
        torque = self.body.holding_torques(targets, tilt.T @ np.array([0.0, 0.0, 1.0]), forces)
        for j, tau in torque.items():
            targets[j] += self.ff_gain * tau / self.kp[j]
        return targets


def _foot_tilt(tilt: np.ndarray, foot_rot: np.ndarray) -> tuple[float, float]:
    """Pitch and roll of a foot with respect to gravity (its own heading removed)."""
    f_level = tilt @ foot_rot
    f_rp = rot_z(-yaw_of(f_level)) @ f_level
    return (math.atan2(-f_rp[2, 0], math.hypot(f_rp[2, 1], f_rp[2, 2])), math.atan2(f_rp[2, 1], f_rp[2, 2]))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _mean_angle(angles) -> float:
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))
