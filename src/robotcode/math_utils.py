"""Small geometry helpers shared by the controllers (independently unit-testable)."""
import math

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_robot(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    return (math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy)


def wheel_targets(forward: float, turn: float, *, radius: float = .075,
                  track: float = .44) -> tuple[float, float]:
    """Differential drive inverse kinematics. Wheel joints spin about +Y: positive spin rolls toward +X."""
    return ((forward - turn * track / 2) / radius,
            (forward + turn * track / 2) / radius)


def to_map(pose, points: np.ndarray) -> np.ndarray:
    """Transform (N, 2) points from the base frame of `pose` = (x, y, yaw) to the map frame."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.stack([pose[0] + c * pts[:, 0] - s * pts[:, 1], pose[1] + s * pts[:, 0] + c * pts[:, 1]], axis=1)


def to_base(pose, points: np.ndarray) -> np.ndarray:
    """Transform (N, 2) map points into the base frame of `pose`."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    dx, dy = pts[:, 0] - pose[0], pts[:, 1] - pose[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)
