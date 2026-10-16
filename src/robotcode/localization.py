"""State estimation: where am I on the map?

* Odometry: travelled distance from the wheel encoders, heading change from the
  gyro (wheels slip when turning; the gyro does not, but drifts).
* Correction: every new LiDAR scan is aligned to the prior map with a
  likelihood-field Gauss-Newton matcher (the core of AMCL/Cartographer-style
  localisation), which removes odometry and gyro drift.
"""
from __future__ import annotations

import math

import numpy as np

from .hal import LaserScan, RobotState
from .mapping import Grid, PriorMap
from .math_utils import wrap_angle


class Odometry:
    def __init__(self, spec: dict):
        wheels = spec["base"]["wheels"]
        self.radius = wheels["radius"]
        self.track = wheels["track"]
        self._prev_wheels: np.ndarray | None = None
        self._prev_stamp: float | None = None

    def update(self, state: RobotState) -> tuple[float, float]:
        """Return (forward distance, heading change) since the previous call."""
        if self._prev_wheels is None:
            self._prev_wheels, self._prev_stamp = state.wheel_position.copy(), state.stamp
            return 0.0, 0.0
        dl, dr = (state.wheel_position - self._prev_wheels) * self.radius
        dt = state.stamp - self._prev_stamp
        self._prev_wheels, self._prev_stamp = state.wheel_position.copy(), state.stamp
        return (dl + dr) / 2.0, float(state.gyro[2]) * dt


class ScanMatcher:
    """Align a scan to the prior map's LiDAR slice by minimising point-to-obstacle distance."""

    def __init__(self, prior: PriorMap, *, inlier_distance: float = 0.12, stride: int = 2):
        self.field = prior.lidar_distance
        gy, gx = np.gradient(self.field.data, self.field.resolution)
        self.grad_x = Grid(self.field.origin, self.field.resolution, gx)
        self.grad_y = Grid(self.field.origin, self.field.resolution, gy)
        self.inlier_distance = inlier_distance
        self.stride = stride

    @staticmethod
    def scan_points(scan: LaserScan, stride: int = 1, max_range: float = 7.0) -> np.ndarray:
        r = scan.ranges[::stride]
        a = scan.angles[::stride]
        ok = np.isfinite(r) & (r < max_range)
        return np.stack([scan.mount[0] + r[ok] * np.cos(a[ok]), scan.mount[1] + r[ok] * np.sin(a[ok])], axis=1)

    def score(self, points: np.ndarray, pose: np.ndarray) -> float:
        c, s = math.cos(pose[2]), math.sin(pose[2])
        wx = pose[0] + c * points[:, 0] - s * points[:, 1]
        wy = pose[1] + s * points[:, 0] + c * points[:, 1]
        d = np.abs(self.field.bilinear(wx, wy, default=1.0))
        return float(np.mean(np.minimum(d, 0.3)))

    def match(self, scan: LaserScan, guess: np.ndarray, iterations: int = 12):
        """Return (pose, inlier ratio, mean inlier residual) or None if the scan is unusable."""
        pts = self.scan_points(scan, self.stride)
        if len(pts) < 30:
            return None
        pose = np.array(guess, dtype=float)
        ratio, residual = 0.0, np.inf
        for _ in range(iterations):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            wx = pose[0] + c * pts[:, 0] - s * pts[:, 1]
            wy = pose[1] + s * pts[:, 0] + c * pts[:, 1]
            d = self.field.bilinear(wx, wy, default=1.0)           # signed distance to the surface
            inl = np.abs(d) < self.inlier_distance
            ratio = float(inl.mean())
            if inl.sum() < 20:
                return None
            gx = self.grad_x.bilinear(wx[inl], wy[inl])
            gy = self.grad_y.bilinear(wx[inl], wy[inl])
            px, py = pts[inl, 0], pts[inl, 1]
            dtheta = gx * (-s * px - c * py) + gy * (c * px - s * py)
            jac = np.stack([gx, gy, dtheta], axis=1)
            r = d[inl]
            residual = float(np.mean(np.abs(r)))
            h = jac.T @ jac + np.diag([1e-3, 1e-3, 1e-3])
            step = -np.linalg.solve(h, jac.T @ r)
            step[:2] = np.clip(step[:2], -0.05, 0.05)
            step[2] = float(np.clip(step[2], -0.05, 0.05))
            pose += step
            if np.linalg.norm(step[:2]) < 1e-4 and abs(step[2]) < 1e-4:
                break
        pose[2] = wrap_angle(pose[2])
        return pose, ratio, residual

    def global_search(self, scan: LaserScan, guess: np.ndarray, *, xy_window: float = 0.3,
                      yaw_window: float = 0.5) -> np.ndarray:
        """Coarse grid search around a rough guess (initial pose), refined by match()."""
        pts = self.scan_points(scan, 3)
        best, best_score = np.array(guess, dtype=float), np.inf
        for dyaw in np.linspace(-yaw_window, yaw_window, 21):
            for dx in np.linspace(-xy_window, xy_window, 7):
                for dy in np.linspace(-xy_window, xy_window, 7):
                    cand = np.array([guess[0] + dx, guess[1] + dy, guess[2] + dyaw])
                    sc = self.score(pts, cand)
                    if sc < best_score:
                        best, best_score = cand, sc
        refined = self.match(scan, best)
        return refined[0] if refined else best


class Localizer:
    """Pose of the base in the map frame (x, y, yaw), plus a covariance-free health signal."""

    def __init__(self, prior: PriorMap, spec: dict, initial_pose):
        self.odometry = Odometry(spec)
        self.matcher = ScanMatcher(prior)
        self.pose = np.array(initial_pose, dtype=float)
        self.initialized = False
        self.last_scan_stamp = -1.0
        self.match_quality = 0.0
        self.corrections = 0
        self.rejections = 0
        self.distance_travelled = 0.0

    def predict(self, state: RobotState) -> None:
        ds, dyaw = self.odometry.update(state)
        mid = self.pose[2] + dyaw / 2
        self.pose += np.array([ds * math.cos(mid), ds * math.sin(mid), dyaw])
        self.pose[2] = wrap_angle(self.pose[2])
        self.distance_travelled += abs(ds)

    def correct(self, scan: LaserScan | None) -> None:
        if scan is None or scan.stamp <= self.last_scan_stamp:
            return
        self.last_scan_stamp = scan.stamp
        if not self.initialized:
            self.pose = self.matcher.global_search(scan, self.pose)
            self.initialized = True
            return
        result = self.matcher.match(scan, self.pose)
        if result is None:
            self.rejections += 1
            return
        pose, ratio, residual = result
        jump = np.hypot(*(pose[:2] - self.pose[:2]))
        if ratio < 0.35 or jump > 0.25 or abs(wrap_angle(pose[2] - self.pose[2])) > 0.25:
            self.rejections += 1
            return
        self.match_quality = ratio
        gain = 0.8
        self.pose[:2] += gain * (pose[:2] - self.pose[:2])
        self.pose[2] = wrap_angle(self.pose[2] + gain * wrap_angle(pose[2] - self.pose[2]))
        self.corrections += 1
