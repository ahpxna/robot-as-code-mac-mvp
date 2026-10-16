"""Kinematics computed from the robot description, as robot_state_publisher / MoveIt do with a URDF.

A private, fixed-base MuJoCo model of the robot alone is used purely as a
kinematics library (forward kinematics, Jacobians). It is fed measured joint
positions; it knows nothing about the simulated world.
"""
from __future__ import annotations

import numpy as np

from .description import arm_joint_names, robot_xml


class Kinematics:
    def __init__(self, spec: dict):
        import mujoco

        self._mj = mujoco
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_string(robot_xml(spec, floating=False))
        self.data = mujoco.MjData(self.model)
        self.arm_joints = arm_joint_names(spec)
        self._qadr = {j: int(self.model.joint(j).qposadr[0]) for j in self.arm_joints}
        self._dof = [int(self.model.joint(j).dofadr[0]) for j in self.arm_joints]
        self.limits = {j: tuple(self.model.jnt_range[self.model.joint(j).id]) for j in self.arm_joints}
        self._update(self.default_joints())

    def default_joints(self) -> dict[str, float]:
        stow = (self.spec.get("arm") or {}).get("stow", {})
        return {j: float(stow.get(j, self.model.qpos0[self._qadr[j]])) for j in self.arm_joints}

    def clamp(self, q: dict[str, float]) -> dict[str, float]:
        return {j: float(np.clip(v, *self.limits[j])) for j, v in q.items()}

    def _update(self, q: dict[str, float] | None) -> None:
        if q:
            for j, v in q.items():
                self.data.qpos[self._qadr[j]] = v
        self._mj.mj_kinematics(self.model, self.data)
        self._mj.mj_comPos(self.model, self.data)
        self._mj.mj_camlight(self.model, self.data)

    def site(self, name: str, q: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Pose of a site in the base frame."""
        self._update(q)
        s = self.data.site(name)
        return s.xpos.copy(), s.xmat.reshape(3, 3).copy()

    def camera(self, name: str, q: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._update(q)
        c = self.data.cam(name)
        return c.xpos.copy(), c.xmat.reshape(3, 3).copy()

    def ik(self, target: np.ndarray, q0: dict[str, float], site: str = "grasp_site",
           iterations: int = 60, tolerance: float = 5e-4, damping: float = 1e-3):
        """Damped least-squares position IK in the base frame, respecting joint limits.

        Works for any arm described in the robot spec, not only the Cartesian one.
        Returns (joint targets, residual error in metres).
        """
        q = dict(q0)
        jacp = np.zeros((3, self.model.nv))
        error = np.inf
        for _ in range(iterations):
            pos, _ = self.site(site, q)
            delta = np.asarray(target, dtype=float) - pos
            error = float(np.linalg.norm(delta))
            if error < tolerance:
                break
            self._mj.mj_jacSite(self.model, self.data, jacp, None, self.model.site(site).id)
            jac = jacp[:, self._dof]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), delta)
            q = self.clamp({j: q[j] + float(dq[i]) for i, j in enumerate(self.arm_joints)})
        return q, error

    def footprint_boxes(self, q: dict[str, float] | None = None) -> list[tuple[float, float, float, float]]:
        """Per-part footprint: (xmin, xmax, ymin, ymax) of every collision geom in the base frame.

        The union of these boxes is much tighter than one bounding rectangle (the gripper sticks
        out in front, but only 17 cm wide), which matters when turning in narrow spaces."""
        self._update(q or self.default_joints())
        m, d = self.model, self.data
        boxes = []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            size = m.geom_size[g]
            gtype = m.geom_type[g]
            if gtype == self._mj.mjtGeom.mjGEOM_SPHERE:
                half = np.full(3, size[0])
            elif gtype in (self._mj.mjtGeom.mjGEOM_CYLINDER, self._mj.mjtGeom.mjGEOM_CAPSULE):
                half = np.array([size[0], size[0], size[1]])
            else:
                half = size[:3]
            ext = np.abs(rot) @ half
            c = d.geom_xpos[g]
            boxes.append((float(c[0] - ext[0]), float(c[0] + ext[0]), float(c[1] - ext[1]), float(c[1] + ext[1])))
        return boxes

    def footprint(self, q: dict[str, float] | None = None) -> tuple[float, float, float, float]:
        """Bounding box (xmin, xmax, ymin, ymax) of the robot's collision geoms in the base frame."""
        boxes = np.array(self.footprint_boxes(q))
        return float(boxes[:, 0].min()), float(boxes[:, 1].max()), float(boxes[:, 2].min()), float(boxes[:, 3].max())
