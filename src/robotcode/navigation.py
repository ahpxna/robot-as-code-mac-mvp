"""Navigation: costmap, global planner, path follower and approach-pose selection.

The structure mirrors Nav2:
  Costmap        prior map + obstacle layer marked/cleared by LiDAR, inflated by distance
  plan_path      shortest path on the costmap (Dijkstra on an 8-connected grid) + shortcutting
  PathFollower   pure pursuit with a LiDAR collision check on the predicted motion
  Navigator      go_to() / dock() as cooperative skills, approach & observation pose selection
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .hal import LaserScan
from .localization import ScanMatcher
from .mapping import Grid, PriorMap, Surface
from .math_utils import clamp, to_base, to_map, wrap_angle, wheel_targets


@dataclass
class NavResult:
    ok: bool
    reason: str = ""
    replans: int = 0


class DiffDriveController:
    """Twist -> wheel speeds with speed and acceleration limits (ros2_control diff_drive_controller)."""

    def __init__(self, hal, spec: dict):
        self.hal = hal
        base = spec["base"]
        self.radius = base["wheels"]["radius"]
        self.track = base["wheels"]["track"]
        lim = base["limits"]
        self.max_v, self.max_w = lim["max_linear_speed"], lim["max_angular_speed"]
        self.max_dv, self.max_dw = lim["max_linear_accel"], lim["max_angular_accel"]
        self.v = self.w = 0.0
        self._no_forward_until = -1.0
        self._now = 0.0
        self._limits = (self.max_v, self.max_w, self.max_dv, self.max_dw)

    def set_carrying(self, carrying: bool) -> None:
        """Gentler speeds and accelerations while an object hangs in the gripper."""
        v, w, dv, dw = self._limits
        self.max_v, self.max_w, self.max_dv, self.max_dw = (
            (0.8 * v, 0.6 * w, 0.6 * dv, 0.4 * dw) if carrying else (v, w, dv, dw))

    def inhibit_forward(self, until: float) -> None:
        """Safety override: refuse forward motion until `until` (s)."""
        self._no_forward_until = until

    def command(self, v: float, w: float, dt: float) -> None:
        self._now = self.hal.time()
        if self._now < self._no_forward_until:
            v = min(v, 0.0)
            self.v = min(self.v, 0.0)
        v = clamp(v, -self.max_v, self.max_v)
        w = clamp(w, -self.max_w, self.max_w)
        self.v += clamp(v - self.v, -self.max_dv * dt, self.max_dv * dt)
        self.w += clamp(w - self.w, -self.max_dw * dt, self.max_dw * dt)
        self.hal.command_wheels(*wheel_targets(self.v, self.w, radius=self.radius, track=self.track))

    def halt(self) -> None:
        self.v = self.w = 0.0
        self.hal.command_wheels(0.0, 0.0)

    def tick(self, dt: float) -> None:
        """Wheel setpoints are sent in command(); nothing to do per tick (legs need it, see legged.py)."""


class Costmap:
    def __init__(self, prior: PriorMap, *, lethal_radius: float, inflation: float = 0.35,
                 obstacle_decay_s: float = 10.0):
        self.prior = prior
        self.grid = prior.occupancy
        self.static = prior.occupancy.data
        self.lethal_radius = lethal_radius
        self.inflation = inflation
        self.decay = obstacle_decay_s
        self.seen = np.full(self.static.shape, -np.inf)        # 2D LiDAR layer
        # 3D layer (3D LiDAR, ToF, known drops): obstacles above/below the 2D scan plane. Kept apart so
        # a 2D beam passing over a low object does not clear it.
        self.seen3d = np.full(self.static.shape, -np.inf)
        self.decay3d = 15.0
        self._cache_key = None
        self._distance = None

    def update(self, pose, scan: LaserScan, now: float, max_range: float = 4.0) -> None:
        r, a = scan.ranges, scan.angles
        hit = np.isfinite(r) & (r < max_range)
        ends = to_map(pose, np.stack([scan.mount[0] + r[hit] * np.cos(a[hit]),
                                      scan.mount[1] + r[hit] * np.sin(a[hit])], axis=1))
        # Clear the obstacle layer along each beam (free space), then mark unexplained endpoints.
        # Cells marked "until the future" (obstacles below the scan plane) are not cleared.
        rr = np.where(np.isfinite(r), np.minimum(r, max_range), max_range) - 0.12
        steps = np.arange(0.1, max_range, self.grid.resolution * 0.7)
        along = steps[None, :] < rr[:, None]
        xs = scan.mount[0] + steps[None, :] * np.cos(a)[:, None]
        ys = scan.mount[1] + steps[None, :] * np.sin(a)[:, None]
        free = to_map(pose, np.stack([xs[along], ys[along]], axis=1))
        ix, iy = self.grid.to_cell(free[:, 0], free[:, 1])
        ok = self.grid.inside(ix, iy)
        ix, iy = ix[ok], iy[ok]
        clear = self.seen[iy, ix] <= now
        self.seen[iy[clear], ix[clear]] = -np.inf
        if len(ends):
            unexplained = self.prior.distance.value(ends[:, 0], ends[:, 1]) > 0.08
            ix, iy = self.grid.to_cell(ends[unexplained, 0], ends[unexplained, 1])
            ok = self.grid.inside(ix, iy)
            self.seen[iy[ok], ix[ok]] = now

    def mark_points(self, xy_map: np.ndarray, now: float) -> None:
        """Mark 3D obstacle points (map frame) that the prior map does not explain."""
        if len(xy_map) == 0:
            return
        unexplained = self.prior.distance.value(xy_map[:, 0], xy_map[:, 1]) > 0.08
        ix, iy = self.grid.to_cell(xy_map[unexplained, 0], xy_map[unexplained, 1])
        ok = self.grid.inside(ix, iy)
        self.seen3d[iy[ok], ix[ok]] = np.maximum(self.seen3d[iy[ok], ix[ok]], now)

    def update_cloud(self, pose, points: np.ndarray, now: float, *, z_max: float, max_range: float = 4.0) -> np.ndarray:
        """3D LiDAR sweep (base frame): floor returns clear the 3D layer, returns between 3 cm and the
        robot's height mark it. Returns the obstacle points in the map frame."""
        near = np.hypot(points[:, 0], points[:, 1]) < max_range
        pts = points[near]
        xy = to_map(pose, pts[:, :2])
        floor = np.abs(pts[:, 2]) < 0.02
        ix, iy = self.grid.to_cell(xy[floor, 0], xy[floor, 1])
        ok = self.grid.inside(ix, iy)
        ix, iy = ix[ok], iy[ok]
        clear = self.seen3d[iy, ix] <= now
        self.seen3d[iy[clear], ix[clear]] = -np.inf
        # Range noise lifts the odd floor return above any small threshold; a real obstacle returns
        # several points per cell, so require two in the same sweep (and > 4 cm, about 4 sigma).
        obstacle = (pts[:, 2] > 0.04) & (pts[:, 2] < z_max + 0.05)
        ix, iy = self.grid.to_cell(xy[obstacle, 0], xy[obstacle, 1])
        cells, inverse, counts = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True,
                                           return_counts=True)
        solid = counts[inverse.ravel()] >= 2
        self.mark_points(xy[obstacle][solid], now)
        return xy[obstacle][solid]

    def add_obstacle(self, xy, radius: float, now: float, lifetime: float = 120.0) -> None:
        """Mark an obstacle no sensor may see again soon (e.g. something dropped on the floor)."""
        grid = self.grid
        ny, nx = self.static.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        cx, cy = grid.to_world(xx, yy)
        near = (cx - xy[0]) ** 2 + (cy - xy[1]) ** 2 <= radius ** 2
        self.seen3d[near] = now + lifetime - self.decay3d

    def occupied(self, now: float) -> np.ndarray:
        return self.static | (self.seen > now - self.decay) | (self.seen3d > now - self.decay3d)

    def distance(self, now: float) -> Grid:
        occ = self.occupied(now)
        key = occ.tobytes()
        if key != self._cache_key:
            self._cache_key = key
            self._distance = Grid(self.grid.origin, self.grid.resolution,
                                  ndimage.distance_transform_edt(~occ) * self.grid.resolution)
        return self._distance

    def cell_costs(self, now: float) -> np.ndarray:
        dist = self.distance(now).data
        cost = 1.0 + 12.0 * np.exp(-np.maximum(dist - self.lethal_radius, 0.0) / (self.inflation / 3))
        cost[dist < self.lethal_radius] = np.inf
        return cost


def plan_path(costmap: Costmap, start, goal, now: float, *, start_free_radius: float = 0.45,
              goal_free_radius: float = 0.25) -> np.ndarray | None:
    grid = costmap.grid
    cost = costmap.cell_costs(now)
    occ = costmap.occupied(now)
    ny, nx = cost.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = grid.to_world(xx, yy)
    # Let the robot leave / reach tight spots (e.g. a docking pose next to a table) through the
    # inflated zone, but never through occupied cells.
    for (px, py), radius in ((start, start_free_radius), (goal, goal_free_radius)):
        near = (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2
        cost[near & ~np.isfinite(cost) & ~occ] = 40.0
    sx, sy = grid.to_cell(*start)
    gx, gy = grid.to_cell(*goal)
    if not (grid.inside(sx, sy) and grid.inside(gx, gy)) or not np.isfinite(cost[gy, gx]):
        return None
    ok = np.isfinite(cost)
    ok[sy, sx] = True
    cost[sy, sx] = min(cost[sy, sx], 40.0)
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = (slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
        b = (slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
        both = ok[a] & ok[b]
        step = math.hypot(dx, dy) * grid.resolution
        w = step * (cost[a] + cost[b]) / 2
        rows.append(idx[a][both])
        cols.append(idx[b][both])
        weights.append(w[both])
    rows, cols, weights = map(np.concatenate, (rows, cols, weights))
    graph = coo_matrix((np.concatenate([weights, weights]), (np.concatenate([rows, cols]),
                                                             np.concatenate([cols, rows]))),
                       shape=(ny * nx, ny * nx)).tocsr()
    start_i, goal_i = idx[sy, sx], idx[gy, gx]
    dist, pred = dijkstra(graph, indices=start_i, return_predecessors=True)
    if not np.isfinite(dist[goal_i]):
        return None
    cells = [goal_i]
    while cells[-1] != start_i:
        cells.append(pred[cells[-1]])
    cells = np.array(cells[::-1])
    px, py = grid.to_world(cells % nx, cells // nx)
    path = np.stack([px, py], axis=1)
    path[0] = start
    path[-1] = goal
    return _shortcut(path, costmap, now)


def _segment_clear(a: np.ndarray, b: np.ndarray, costmap: Costmap, now: float, clearance: float) -> bool:
    n = max(2, int(np.linalg.norm(b - a) / (costmap.grid.resolution * 0.5)))
    pts = a[None, :] + (b - a)[None, :] * np.linspace(0, 1, n)[:, None]
    return bool(np.all(costmap.distance(now).value(pts[:, 0], pts[:, 1]) >= clearance))


def _shortcut(path: np.ndarray, costmap: Costmap, now: float) -> np.ndarray:
    clearance = costmap.lethal_radius + 0.08
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _segment_clear(path[i], path[j], costmap, now, clearance):
            j -= 1
        out.append(path[j])
        i = j
    out = np.array(out)
    dense = [out[0]]
    for a, b in zip(out[:-1], out[1:]):
        n = max(1, int(np.linalg.norm(b - a) / 0.1))
        dense.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(dense)


def footprint_clearance(boxes, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
    best = np.full(np.shape(lx), np.inf)
    for xmin, xmax, ymin, ymax in boxes:
        dx = np.maximum(np.maximum(xmin - lx, lx - xmax), 0.0)
        dy = np.maximum(np.maximum(ymin - ly, ly - ymax), 0.0)
        inside = np.minimum(np.minimum(lx - xmin, xmax - lx), np.minimum(ly - ymin, ymax - ly))
        best = np.minimum(best, np.where((dx == 0) & (dy == 0), -inside, np.hypot(dx, dy)))
    return best


class PathFollower:
    """Pure pursuit + rotate-in-place, with speed scaled by a LiDAR collision forecast."""

    def __init__(self, path: np.ndarray, goal_yaw: float | None, footprint: list, *, max_v: float, max_w: float,
                 lookahead: float = 0.35, pos_tol: float = 0.05, yaw_tol: float = 0.035, margin: float = 0.04):
        self.path = path
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.goal_yaw = goal_yaw
        self.footprint = footprint
        self.max_v, self.max_w = max_v, max_w
        self.lookahead, self.pos_tol, self.yaw_tol, self.margin = lookahead, pos_tol, yaw_tol, margin
        self.index = 0
        self.aligning = False
        self.turn_dir: float | None = None

    def _point_at(self, s: float) -> np.ndarray:
        s = min(s, self.s[-1])
        k = int(np.searchsorted(self.s, s, side="right") - 1)
        k = min(k, len(self.path) - 2)
        t = (s - self.s[k]) / max(self.s[k + 1] - self.s[k], 1e-9)
        return self.path[k] + t * (self.path[k + 1] - self.path[k])

    def clearance(self, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
        """Signed distance from base-frame points to the footprint (union of boxes; negative inside)."""
        return footprint_clearance(self.footprint, lx, ly)

    ROTATION_MARGIN = 0.015     # turning in place is slow and controlled: a smaller safety margin than driving

    def rotation_free(self, angle: float, obstacles: np.ndarray) -> bool:
        """Can the robot turn in place by `angle` (rad, signed) without its footprint touching anything?"""
        return self.time_to_collision(0.0, math.copysign(1.0, angle), obstacles, horizon=abs(angle),
                                      margin=self.ROTATION_MARGIN) == math.inf

    def time_to_collision(self, v: float, w: float, obstacles: np.ndarray, horizon: float = 1.2,
                          margin: float | None = None) -> float:
        """First time (s) at which driving at (v, w) brings the footprint within `margin` of a LiDAR
        point (base-frame points). Points already that close only count if the motion gets closer,
        so the robot can always back away from something it docked next to."""
        if len(obstacles) == 0:
            return math.inf
        near = np.hypot(obstacles[:, 0], obstacles[:, 1]) < 1.0 + abs(v) * horizon
        pts = obstacles[near]
        if len(pts) == 0:
            return math.inf
        d0 = self.clearance(pts[:, 0], pts[:, 1])
        # Sample every ~5 cm of travel / ~6 degrees of rotation, at least 8 times.
        samples = max(8, int(max(abs(v) * horizon / 0.05, abs(w) * horizon / 0.1)))
        for t in np.linspace(horizon / samples, horizon, samples):
            th = w * t
            if abs(w) < 1e-6:
                ox, oy = v * t, 0.0
            else:
                ox, oy = v / w * math.sin(th), v / w * (1 - math.cos(th))
            c, s = math.cos(th), math.sin(th)
            dx, dy = pts[:, 0] - ox, pts[:, 1] - oy
            d = self.clearance(c * dx + s * dy, -s * dx + c * dy)
            if np.any((d < (self.margin if margin is None else margin)) & (d < d0 - 0.002)):
                return float(t)
        return math.inf

    def step(self, pose: np.ndarray, obstacles: np.ndarray) -> tuple[float, float, str]:
        pos = pose[:2]
        window = slice(self.index, min(self.index + 30, len(self.path)))
        d = np.linalg.norm(self.path[window] - pos, axis=1)
        self.index = self.index + int(np.argmin(d))
        remaining = self.s[-1] - self.s[self.index] + float(np.min(d))
        goal = self.path[-1]
        if self.aligning or np.linalg.norm(goal - pos) < self.pos_tol or (
                self.index >= len(self.path) - 2 and np.linalg.norm(goal - pos) < 2 * self.pos_tol):
            self.aligning = True
            if self.goal_yaw is None:
                return 0.0, 0.0, "arrived"
            err = wrap_angle(self.goal_yaw - pose[2])
            if abs(err) < self.yaw_tol:
                return 0.0, 0.0, "arrived"
            w = clamp(2.0 * err, -self.max_w, self.max_w)
            w = math.copysign(max(abs(w), 0.12), w)
            return 0.0, w, "running"
        target = self._point_at(self.s[self.index] + self.lookahead)
        dx, dy = target - pos
        alpha = wrap_angle(math.atan2(dy, dx) - pose[2])
        if abs(alpha) > 0.6:
            if self.turn_dir is None:
                # Pick a turning direction whose WHOLE sweep is free: the short way, else the long way.
                short = math.copysign(1.0, alpha)
                if self.rotation_free(short * abs(alpha), obstacles):
                    self.turn_dir = short
                elif self.rotation_free(-short * (2 * math.pi - abs(alpha)), obstacles):
                    self.turn_dir = -short
                else:
                    return 0.0, 0.0, "blocked"
            remaining = alpha if math.copysign(1.0, alpha) == self.turn_dir else alpha - self.turn_dir * 2 * math.pi
            if self.time_to_collision(0.0, self.turn_dir, obstacles, horizon=min(abs(remaining), 0.6),
                                      margin=self.ROTATION_MARGIN) < math.inf:
                self.turn_dir = None
                return 0.0, 0.0, "blocked"
            w = self.turn_dir * clamp(1.8 * abs(remaining), 0.4, self.max_w)
            return 0.0, w, "running"
        self.turn_dir = None
        v = min(self.max_v, math.sqrt(2 * 0.35 * max(remaining - 0.02, 0.0)) + 0.04) * max(math.cos(alpha), 0.2)
        dist = max(math.hypot(dx, dy), 1e-3)
        w = clamp(2.0 * v * math.sin(alpha) / dist + 0.6 * alpha, -self.max_w, self.max_w)
        ttc = self.time_to_collision(v, w, obstacles)
        if ttc < 1.2:
            scale = clamp((ttc - 0.25) / 0.9, 0.0, 1.0)
            v *= scale
            if scale == 0.0:
                return 0.0, 0.0, "blocked"
        return v, w, "running"


class Navigator:
    def __init__(self, ctx):
        self.ctx = ctx
        spec = ctx.spec
        self.footprint = ctx.kin.footprint_boxes()
        corners = np.array([(x, y) for x0, x1, y0, y1 in self.footprint for x in (x0, x1) for y in (y0, y1)])
        self.half_width = float(np.abs(corners[:, 1]).max())
        self.front = float(corners[:, 0].max())
        self.circumradius = float(np.hypot(corners[:, 0], corners[:, 1]).max())
        self._sample = np.concatenate([
            np.stack(np.meshgrid(np.linspace(x0, x1, max(2, int((x1 - x0) / 0.04) + 1)),
                                 np.linspace(y0, y1, max(2, int((y1 - y0) / 0.04) + 1))), axis=-1).reshape(-1, 2)
            for x0, x1, y0, y1 in self.footprint])
        self.costmap = Costmap(ctx.prior, lethal_radius=self.half_width + 0.03)
        lim = spec["base"]["limits"]
        self.max_v, self.max_w = lim["max_linear_speed"], lim["max_angular_speed"]
        self.path: np.ndarray | None = None
        self.goal = None
        self.robot_height = ctx.kin.height()
        self._scan_map = np.zeros((0, 2))
        self._cloud_map = np.zeros((0, 2))

    # ---- perception of free space ---------------------------------------------
    def obstacles_base(self) -> np.ndarray:
        """Obstacle points in the current base frame from every ranging sensor (for collision checks).
        LiDAR points are kept in the map frame between sweeps so they stay put while the robot moves."""
        parts = [to_base(self.ctx.pose, pts) for pts in (self._scan_map, self._cloud_map) if len(pts)]
        prox = self.ctx.proximity
        if prox is not None:
            hit = np.isfinite(prox.ranges)
            m = prox.mounts[hit]
            r = prox.ranges[hit]
            parts.append(np.stack([m[:, 0] + r * np.cos(m[:, 2]), m[:, 1] + r * np.sin(m[:, 2])], axis=1))
        if not parts:
            return np.zeros((0, 2))
        pts = np.concatenate(parts)
        return pts[np.hypot(pts[:, 0], pts[:, 1]) < 3.0]

    def update(self, now: float) -> None:
        if self.ctx.scan is not None:
            self.costmap.update(self.ctx.pose, self.ctx.scan, now)
            self._scan_map = to_map(self.ctx.pose, ScanMatcher.scan_points(self.ctx.scan, 1, max_range=3.0))

    def update_cloud(self, now: float) -> None:
        cloud = self.ctx.cloud
        if cloud is not None:
            points = cloud.points
            ctx = self.ctx
            if ctx.world.held is not None and ctx.arm is not None:
                # Attached object: the LiDAR sees what the gripper holds. Like MoveIt's attached
                # collision objects, returns around it are part of the robot, not obstacles.
                # Only the held object's own volume is masked: anything else near the gripper is a
                # real obstacle that the object (or the gripper) is about to hit.
                held = ctx.world.objects[ctx.world.held]
                otype = ctx.catalog["types"][held.type]
                height = 2 * otype["radius"] if otype["shape"] == "sphere" else otype["height"]
                below = held.history[-1]["grasp_offset"] if held.history else height / 2
                grip = ctx.arm.gripper_position()
                on_object = ((np.linalg.norm(points[:, :2] - grip[:2], axis=1) < otype["radius"] + 0.02)
                             & (points[:, 2] > grip[2] - below - 0.02) & (points[:, 2] < grip[2] + height - below + 0.02))
                points = points[~on_object]
            points = points + np.array([0.0, 0.0, ctx.base_height])     # heights above the floor
            self._cloud_map = self.costmap.update_cloud(ctx.pose, points, now, z_max=self.robot_height)

    def update_proximity(self, now: float) -> None:
        prox = self.ctx.proximity
        if prox is None:
            return
        hit = np.isfinite(prox.ranges)
        if np.any(hit):
            m, r = prox.mounts[hit], prox.ranges[hit]
            pts = np.stack([m[:, 0] + r * np.cos(m[:, 2]), m[:, 1] + r * np.sin(m[:, 2])], axis=1)
            self.costmap.mark_points(to_map(self.ctx.pose, pts), now)

    def footprint_free(self, pose, margin: float = 0.0, now: float | None = None) -> bool:
        pts = self._sample
        if margin:
            # Push the sample points outward from the base origin by `margin` (cheap dilation).
            norm = np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-6)
            pts = np.concatenate([pts, pts + pts / norm * margin])
        pts = to_map(pose, pts)
        occ = self.costmap.occupied(self.ctx.time if now is None else now)
        ix, iy = self.costmap.grid.to_cell(pts[:, 0], pts[:, 1])
        if not np.all(self.costmap.grid.inside(ix, iy)):
            return False
        return not bool(np.any(occ[iy, ix]))

    # ---- skills ------------------------------------------------------------------
    def _path_still_free(self, follower: "PathFollower", now: float) -> bool:
        """Is the rest of the path clear of obstacles the LiDAR has seen since planning?"""
        path = follower.path
        s0 = follower.s[follower.index] + 0.3
        ahead = path[(follower.s > s0) & (follower.s < follower.s[-1] - 0.4)]
        if len(ahead) == 0:
            return True
        dist = self.costmap.distance(now).value(ahead[:, 0], ahead[:, 1])
        return bool(np.all(dist >= self.costmap.lethal_radius))

    def go_to(self, goal_xy, goal_yaw: float | None = None, *, max_time: float = 150.0, pos_tol: float = 0.05):
        """Generator: drive to a map position (and heading). Returns NavResult."""
        ctx = self.ctx
        goal = np.array(goal_xy, dtype=float)
        replans, stalls, blocked_since, start_time = 0, 0, None, ctx.time
        self.goal = (goal, goal_yaw)
        ctx.drive.set_carrying(ctx.world.held is not None)
        if not self.can_turn():
            yield from self.escape()

        def new_follower(previous=None):
            path = plan_path(self.costmap, ctx.pose[:2].copy(), goal, ctx.time)
            self.path = path
            if path is None:
                return None
            follower = PathFollower(path, goal_yaw, self.footprint, max_v=ctx.drive.max_v, max_w=ctx.drive.max_w,
                                    pos_tol=pos_tol)
            if previous is not None:
                # Keep turning the way we were: flipping direction at every replan makes the robot dither.
                follower.turn_dir = previous.turn_dir
            return follower

        follower = new_follower()
        if follower is None:
            ctx.drive.halt()
            return NavResult(False, f"no collision-free path to ({goal[0]:.2f}, {goal[1]:.2f})")
        progress_t, progress_pose, last_check, last_replan = ctx.time, ctx.pose.copy(), ctx.time, ctx.time
        while True:
            now = ctx.time
            if now - start_time > max_time:
                ctx.drive.halt()
                return NavResult(False, "navigation timed out", replans)
            lost = ctx.grip_lost()
            if lost:
                ctx.drive.halt()
                return NavResult(False, lost, replans)
            v, w, status = follower.step(ctx.pose, self.obstacles_base())
            if status == "arrived":
                ctx.drive.halt()
                self.path = None
                return NavResult(True, "arrived", replans)
            need_replan = False
            if status == "blocked":
                ctx.drive.command(0.0, 0.0, ctx.dt)
                blocked_since = blocked_since if blocked_since is not None else now
                if now - blocked_since > 1.5:
                    yield from self.escape(0.3)
                    need_replan, blocked_since = True, None
            else:
                blocked_since = None
                ctx.drive.command(v, w, ctx.dt)
                # Re-check the path against new sensor data only while actually driving, and not more than
                # every 3 s: an obstacle revealed bit by bit must not flip the route left/right each second.
                if now - last_check > 1.0 and now - last_replan > 3.0 and v > 0.05:
                    last_check = now
                    if not self._path_still_free(follower, now):
                        ctx.log("new obstacle on the path, replanning")
                        need_replan = True
            # Stall detection: commanding motion but not moving (wheels slipping, wedged on something low).
            if now - progress_t > 3.0:
                moved = np.hypot(*(ctx.pose[:2] - progress_pose[:2]))
                turned = abs(wrap_angle(ctx.pose[2] - progress_pose[2]))
                if moved < 0.03 and turned < 0.08 and blocked_since is None and not follower.aligning:
                    stalls += 1
                    ctx.log(f"robot is not moving although commanded (stall {stalls})")
                    if stalls > 2:
                        ctx.drive.halt()
                        return NavResult(False, "stuck: the robot does not move when commanded", replans)
                    yield from self.escape(0.3)
                    need_replan = True
                progress_t, progress_pose = now, ctx.pose.copy()
            if need_replan:
                replans += 1
                if replans > 8:
                    ctx.drive.halt()
                    return NavResult(False, "path blocked by an obstacle", replans)
                if status == "blocked":
                    ctx.log(f"path blocked, replanning ({replans})")
                follower = new_follower(follower)
                if follower is None:
                    ctx.drive.halt()
                    return NavResult(False, "path blocked and no alternative route", replans)
                progress_t, progress_pose, last_replan = ctx.time, ctx.pose.copy(), ctx.time
            yield

    def dock(self, pose, *, speed: float = 0.12, pos_tol: float = 0.012, yaw_tol: float = 0.02,
             max_time: float = 20.0):
        """Generator: short, slow, straight-line approach to an exact pose (pre-validated as free)."""
        # A differential drive cannot slide sideways: the residual lateral error is left for the
        # arm's wrist joint to absorb, so only the along-track distance and heading are servoed.
        ctx = self.ctx
        target = np.array(pose, dtype=float)
        start = ctx.time
        while ctx.time - start < max_time:
            ex, ey = to_base(ctx.pose, target[None, :2])[0]
            yaw_err = wrap_angle(target[2] - ctx.pose[2])
            if abs(ex) < pos_tol:
                if abs(yaw_err) < yaw_tol:
                    ctx.drive.halt()
                    return NavResult(True, f"docked (lateral offset {ey:+.3f} m)")
                v, w = 0.0, math.copysign(clamp(abs(1.5 * yaw_err), 0.08, 0.4), yaw_err)
            else:
                v = clamp(1.2 * ex, -speed, speed)
                v = math.copysign(max(abs(v), 0.03), v)
                steer = 2.0 * ey * (1.0 if ex > 0 else -1.0) if abs(ex) > 0.04 else 0.0
                w = clamp(1.5 * yaw_err + steer, -0.4, 0.4)
            ctx.drive.command(v, w, ctx.dt)
            yield
        ctx.drive.halt()
        return NavResult(False, "dock timed out")

    def approach(self, pose, *, backoff: float = 0.35, max_time: float = 150.0):
        """Generator: navigate to a pre-pose behind `pose`, then dock straight in."""
        pose = np.array(pose, dtype=float)
        heading = np.array([math.cos(pose[2]), math.sin(pose[2])])
        # Pre-pose on the approach line where the robot has the most room to turn in place.
        pre, best = pose.copy(), -np.inf
        for b in np.linspace(backoff, 0.0, 8):
            cand = pose.copy()
            cand[:2] = pose[:2] - heading * b
            if not self.footprint_free(cand, margin=0.03):
                continue
            room = self.turning_room(cand[:2])
            if room >= 0.02:
                pre = cand
                break
            if room > best:
                pre, best = cand, room
        result = yield from self.go_to(pre[:2], pre[2], max_time=max_time, pos_tol=0.06)
        if not result.ok:
            return result
        docked = yield from self.dock(pose)
        return NavResult(docked.ok, docked.reason, result.replans)

    # ---- pose selection ------------------------------------------------------------
    def manipulation_poses(self, target_xy, surface: Surface | None, *, reach_min: float, reach_max: float,
                           ideal_reach: float, now: float | None = None) -> list[np.ndarray]:
        """Candidate base poses from which the gripper can reach target_xy, best first.

        `reach_*` are distances from the base origin to the target along the heading.
        """
        target = np.asarray(target_xy, dtype=float)
        candidates = []
        if surface is not None:
            standoff = self.front + 0.08
            for mid, normal, half_len in surface.edges():
                depth = float(np.dot(mid - target, normal))
                along = target + normal * depth - mid
                if depth < -0.02 or np.linalg.norm(along) > half_len + 0.05:
                    continue
                # Straight-on first, then angled approaches (useful when neighbours crowd the object).
                for angle in (0.0, 0.35, -0.35, 0.6, -0.6):
                    c, s = math.cos(angle), math.sin(angle)
                    direction = np.array([c * normal[0] - s * normal[1], s * normal[0] + c * normal[1]])
                    dist = (depth + standoff) / c
                    if dist > reach_max:
                        continue
                    dist = max(dist, reach_min)
                    base_xy = target + direction * dist
                    yaw = math.atan2(-direction[1], -direction[0])
                    candidates.append((depth + 0.12 * abs(angle), np.array([base_xy[0], base_xy[1], yaw])))
        else:
            for k in range(16):
                yaw = k * math.pi / 8
                base_xy = target - ideal_reach * np.array([math.cos(yaw), math.sin(yaw)])
                candidates.append((0.0, np.array([base_xy[0], base_xy[1], yaw])))
        here = self.ctx.pose[:2]
        scored = []
        for d, p in candidates:
            if not self.footprint_free(p, margin=0.02, now=now):
                continue
            # Avoid slots the robot cannot turn its way out of (between a chair and a wall, say):
            # its pre-approach point, 0.35 m back, should leave room to turn in place.
            heading = np.array([math.cos(p[2]), math.sin(p[2])])
            room = self.turning_room(p[:2] - 0.35 * heading)
            penalty = 0.0 if room >= 0.08 else 0.6 + 2.0 * (0.08 - room)
            scored.append((d + 0.1 * np.linalg.norm(p[:2] - here) + penalty, p))
        return [p for _, p in sorted(scored, key=lambda item: item[0])]

    def observation_poses(self, surface: Surface, *, distance: float = 0.75) -> list[np.ndarray]:
        """Poses facing the surface from each accessible side, with room to turn in place there."""
        out = []
        dist = self.costmap.distance(self.ctx.time)
        for mid, normal, half_len in surface.edges():
            tangent = np.array([-normal[1], normal[0]])
            for d in (distance, distance - 0.15, distance + 0.15, distance + 0.3):
                found = None
                for lateral in (0.0, 0.25, -0.25, 0.45, -0.45):
                    if abs(lateral) > half_len + 0.1:
                        continue
                    p = mid + normal * d + tangent * lateral
                    yaw = math.atan2(surface.center[1] - p[1], surface.center[0] - p[0])
                    pose = np.array([p[0], p[1], yaw])
                    if dist.bilinear(p[0], p[1]) >= self.circumradius + 0.03 and self.footprint_free(pose, 0.06):
                        found = pose
                        break
                if found is not None:
                    out.append(found)
                    break
        here = self.ctx.pose[:2]
        return sorted(out, key=lambda p: np.linalg.norm(p[:2] - here))

    def turn_to(self, yaw: float, *, tol: float = 0.035, max_time: float = 10.0):
        """Generator: rotate in place to a heading (refuses if there is no room to turn)."""
        ctx = self.ctx
        follower = PathFollower(np.array([ctx.pose[:2], ctx.pose[:2] + 1e-3]), None, self.footprint,
                                max_v=0.1, max_w=self.max_w)
        start = ctx.time
        while ctx.time - start < max_time:
            err = wrap_angle(yaw - ctx.pose[2])
            if abs(err) < tol:
                ctx.drive.halt()
                return NavResult(True, "turned")
            w = math.copysign(clamp(abs(2.0 * err), 0.12, self.max_w), err)
            if follower.time_to_collision(0.0, w, self.obstacles_base(), horizon=0.5) < 0.5:
                ctx.drive.halt()
                return NavResult(False, "no room to turn")
            ctx.drive.command(0.0, w, ctx.dt)
            yield
        ctx.drive.halt()
        return NavResult(False, "turn timed out")

    def back_off(self, distance: float, speed: float = 0.1):
        """Generator: reverse straight by `distance` if the LiDAR shows the rear is clear."""
        ctx = self.ctx
        start = ctx.pose[:2].copy()
        follower = PathFollower(np.array([start, start + 1e-3]), None, self.footprint, max_v=speed, max_w=0.1)
        while np.linalg.norm(ctx.pose[:2] - start) < distance:
            if follower.time_to_collision(-speed, 0.0, self.obstacles_base(), horizon=0.6) < 0.6:
                ctx.drive.halt()
                return False
            ctx.drive.command(-speed, 0.0, ctx.dt)
            yield
        ctx.drive.halt()
        return True

    def turning_room(self, xy=None) -> float:
        xy = self.ctx.pose[:2] if xy is None else xy
        return float(self.costmap.distance(self.ctx.time).bilinear(xy[0], xy[1])) - self.circumradius

    def _probe(self) -> "PathFollower":
        here = self.ctx.pose[:2].copy()
        return PathFollower(np.array([here, here + 1e-3]), None, self.footprint, max_v=0.1, max_w=1.0)

    def can_turn(self) -> bool:
        """Room to turn in place, by the map AND by what the ranging sensors see right now
        (a door jamb 5 cm from the side is in the map, but only the sensors know how close it is)."""
        if self.turning_room() < 0.03:
            return False
        probe, obstacles = self._probe(), self.obstacles_base()
        return any(probe.rotation_free(a, obstacles) for a in (math.pi, -math.pi))

    def escape(self, max_distance: float = 0.6):
        """Generator: drive straight out of a tight spot (e.g. after docking at a table, or wedged in
        a doorway) until the robot can turn in place: backwards if the rear is clear, else forwards."""
        ctx = self.ctx
        start = ctx.pose[:2].copy()
        probe = self._probe()
        # Head (forwards or backwards) towards more open space.
        heading = np.array([math.cos(ctx.pose[2]), math.sin(ctx.pose[2])])
        ahead, behind = self.turning_room(start + 0.2 * heading), self.turning_room(start - 0.2 * heading)
        direction = 1.0 if ahead > behind + 0.02 else -1.0
        while not self.can_turn() and np.linalg.norm(ctx.pose[:2] - start) < max_distance:
            obstacles = self.obstacles_base()
            if probe.time_to_collision(0.1 * direction, 0.0, obstacles, horizon=0.6) < 0.6:
                if probe.time_to_collision(-0.1 * direction, 0.0, obstacles, horizon=0.6) < 0.6:
                    break
                direction = -direction
            ctx.drive.command(0.1 * direction, 0.0, ctx.dt)
            yield
        ctx.drive.halt()
        return self.can_turn()
