"""Prior map of a world: occupancy grids plus annotated rooms, surfaces and places.

The map is *built* from the world's static geometry (the role a floor plan or a
one-off SLAM session plays for a real deployment), saved under build/maps/, and
then *loaded* by the onboard software. The robot stack never touches the
simulator's model or state.
"""
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import BUILD_DIR, load_world

_GEOM_PLANE, _GEOM_SPHERE, _GEOM_CAPSULE, _GEOM_ELLIPSOID, _GEOM_CYLINDER, _GEOM_BOX = 0, 2, 3, 4, 5, 6


def normalize_text(text: str) -> str:
    """Casefold and strip Vietnamese diacritics so 'Phòng Khách' == 'phong khach'."""
    text = text.casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


@dataclass
class Grid:
    origin: np.ndarray          # world (x, y) of the lower-left corner of cell (0, 0)
    resolution: float
    data: np.ndarray            # indexed [iy, ix]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def to_cell(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return ix, iy

    def to_world(self, ix, iy):
        return (self.origin[0] + (np.asarray(ix) + 0.5) * self.resolution,
                self.origin[1] + (np.asarray(iy) + 0.5) * self.resolution)

    def inside(self, ix, iy):
        ny, nx = self.data.shape
        return (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)

    def value(self, x, y, default=0.0):
        """Nearest-cell lookup, vectorised; `default` outside the grid."""
        ix, iy = self.to_cell(x, y)
        ok = self.inside(ix, iy)
        out = np.full(np.shape(ix), default, dtype=float)
        out[ok] = self.data[iy[ok], ix[ok]]
        return out

    def bilinear(self, x, y, default=0.0):
        fx = (np.asarray(x) - self.origin[0]) / self.resolution - 0.5
        fy = (np.asarray(y) - self.origin[1]) / self.resolution - 0.5
        ny, nx = self.data.shape
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
        out = np.full(np.shape(fx), default, dtype=float)
        if np.any(ok):
            ax, ay = fx[ok] - x0[ok], fy[ok] - y0[ok]
            d = self.data
            xs, ys = x0[ok], y0[ok]
            out[ok] = ((1 - ax) * (1 - ay) * d[ys, xs] + ax * (1 - ay) * d[ys, xs + 1]
                       + (1 - ax) * ay * d[ys + 1, xs] + ax * ay * d[ys + 1, xs + 1])
        return out


@dataclass
class Surface:
    name: str
    kind: str
    center: np.ndarray
    half: np.ndarray            # half extents along the surface's own axes
    yaw: float
    top_z: float
    room: str | None
    aliases: list[str] = field(default_factory=list)

    def to_local(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = np.asarray(x) - self.center[0], np.asarray(y) - self.center[1]
        return c * dx + s * dy, -s * dx + c * dy

    def contains(self, x, y, margin: float = 0.0):
        lx, ly = self.to_local(x, y)
        return (np.abs(lx) <= self.half[0] + margin) & (np.abs(ly) <= self.half[1] + margin)

    def edges(self):
        """(midpoint, outward unit normal, half length) of the four sides."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ax, ay = np.array([c, s]), np.array([-s, c])
        hx, hy = self.half
        return [(self.center + ax * hx, ax, hy), (self.center - ax * hx, -ax, hy),
                (self.center + ay * hy, ay, hx), (self.center - ay * hy, -ay, hx)]


@dataclass
class Room:
    name: str
    rect: tuple[float, float, float, float]
    aliases: list[str]
    search_from: tuple[float, float] | None

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.rect
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


@dataclass
class Place:
    name: str
    xy: np.ndarray
    yaw: float | None
    radius: float
    kind: str
    aliases: list[str]


@dataclass
class PriorMap:
    name: str
    occupancy: Grid             # anything between the floor and robot height
    lidar: Grid                 # slice at LiDAR height (what the scanner can see)
    lidar_height: float
    rooms: dict[str, Room]
    surfaces: dict[str, Surface]
    places: dict[str, Place]
    user: dict
    start_pose: tuple[float, float, float]
    distance: Grid = None       # metres to the nearest occupied cell (navigation)
    lidar_distance: Grid = None  # signed metres to the nearest LiDAR-visible surface (localisation)

    def __post_init__(self):
        res, lres = self.occupancy.resolution, self.lidar.resolution
        self.distance = Grid(self.occupancy.origin, res,
                             ndimage.distance_transform_edt(~self.occupancy.data) * res)
        # Distance to the obstacle *surface*, also inside obstacles: otherwise a pose that pushes scan
        # points into a thick obstacle (a counter, a fridge) costs nothing and becomes a false optimum.
        outside = ndimage.distance_transform_edt(~self.lidar.data) * lres
        inside = ndimage.distance_transform_edt(self.lidar.data) * lres
        # Signed (positive in free space, negative inside, zero on the surface) so that the
        # interpolated field crosses zero exactly at the obstacle boundary.
        signed = np.where(self.lidar.data, -(inside - lres / 2), outside - lres / 2)
        self.lidar_distance = Grid(self.lidar.origin, lres, signed)

    def room_of(self, x: float, y: float) -> str | None:
        for room in self.rooms.values():
            if room.contains(x, y):
                return room.name
        return None

    def surface_at(self, x: float, y: float, margin: float = 0.03) -> Surface | None:
        for surface in self.surfaces.values():
            if surface.contains(x, y, margin):
                return surface
        return None

    def vocabulary(self) -> dict[str, list[str]]:
        vocab = {name: [name.replace("_", " ")] + room.aliases for name, room in self.rooms.items()}
        vocab.update({name: [name.replace("_", " ")] + s.aliases for name, s in self.surfaces.items()})
        vocab.update({name: [name.replace("_", " ")] + p.aliases for name, p in self.places.items()})
        vocab["user"] = ["user"] + list(self.user.get("aliases", []))
        return vocab

    # ---- persistence -------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.npz"
        meta = {
            "name": self.name, "lidar_height": self.lidar_height, "user": self.user,
            "start_pose": list(self.start_pose), "resolution": self.occupancy.resolution,
            "origin": self.occupancy.origin.tolist(), "lidar_resolution": self.lidar.resolution,
            "rooms": {k: {"rect": list(r.rect), "aliases": r.aliases,
                          "search_from": list(r.search_from) if r.search_from else None}
                      for k, r in self.rooms.items()},
            "surfaces": {k: {"kind": s.kind, "center": s.center.tolist(), "half": s.half.tolist(),
                             "yaw": s.yaw, "top_z": s.top_z, "room": s.room, "aliases": s.aliases}
                         for k, s in self.surfaces.items()},
            "places": {k: {"xy": p.xy.tolist(), "yaw": p.yaw, "radius": p.radius, "kind": p.kind,
                           "aliases": p.aliases} for k, p in self.places.items()},
        }
        np.savez_compressed(path, occupancy=self.occupancy.data, lidar=self.lidar.data,
                            meta=np.array(json.dumps(meta, ensure_ascii=False)))
        _write_pgm(directory / f"{self.name}.pgm", self.occupancy.data)
        return path

    @classmethod
    def load(cls, path: Path) -> "PriorMap":
        blob = np.load(path)
        meta = json.loads(str(blob["meta"]))
        origin, res = np.array(meta["origin"]), meta["resolution"]
        return cls(
            name=meta["name"],
            occupancy=Grid(origin, res, blob["occupancy"].astype(bool)),
            lidar=Grid(origin, meta["lidar_resolution"], blob["lidar"].astype(bool)),
            lidar_height=meta["lidar_height"],
            rooms={k: Room(k, tuple(v["rect"]), v["aliases"], tuple(v["search_from"]) if v["search_from"] else None)
                   for k, v in meta["rooms"].items()},
            surfaces={k: Surface(k, v["kind"], np.array(v["center"]), np.array(v["half"]), v["yaw"], v["top_z"],
                                 v["room"], v["aliases"]) for k, v in meta["surfaces"].items()},
            places={k: Place(k, np.array(v["xy"]), v["yaw"], v["radius"], v["kind"], v["aliases"])
                    for k, v in meta["places"].items()},
            user=meta["user"],
            start_pose=tuple(meta["start_pose"]),
        )


def _write_pgm(path: Path, occupied: np.ndarray) -> None:
    """Nav2-style map image (white free, black occupied), top row = max y."""
    img = np.where(occupied[::-1], 0, 254).astype(np.uint8)
    header = f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode()
    path.write_bytes(header + img.tobytes())


# ---------------------------------------------------------------------------
# Building the map from static world geometry
# ---------------------------------------------------------------------------
def static_geoms(model):
    """Collidable geoms attached to the world body (the static environment)."""
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != 0 or model.geom_type[gid] == _GEOM_PLANE:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        yield gid


def _inside(gtype: int, size: np.ndarray, local: np.ndarray) -> np.ndarray:
    if gtype == _GEOM_BOX:
        return np.all(np.abs(local) <= size[:3], axis=1)
    if gtype == _GEOM_CYLINDER:
        return (local[:, 0] ** 2 + local[:, 1] ** 2 <= size[0] ** 2) & (np.abs(local[:, 2]) <= size[1])
    if gtype == _GEOM_CAPSULE:
        z = np.clip(local[:, 2], -size[1], size[1])
        return local[:, 0] ** 2 + local[:, 1] ** 2 + (local[:, 2] - z) ** 2 <= size[0] ** 2
    if gtype == _GEOM_SPHERE:
        return np.sum(local ** 2, axis=1) <= size[0] ** 2
    if gtype == _GEOM_ELLIPSOID:
        return np.sum((local / size[:3]) ** 2, axis=1) <= 1.0
    raise ValueError(f"map builder does not support geom type {gtype}")


def _half_extents(gtype: int, size: np.ndarray) -> np.ndarray:
    if gtype == _GEOM_BOX or gtype == _GEOM_ELLIPSOID:
        return size[:3].copy()
    if gtype == _GEOM_CYLINDER:
        return np.array([size[0], size[0], size[1]])
    if gtype == _GEOM_CAPSULE:
        return np.array([size[0], size[0], size[1] + size[0]])
    return np.array([size[0]] * 3)


def rasterize(model, data, grid: Grid, z_range: tuple[float, float], z_step: float = 0.02) -> np.ndarray:
    """Mark cells whose column intersects any static geom inside z_range (3x3 sub-samples per cell)."""
    occupied = np.zeros(grid.shape, dtype=bool)
    res = grid.resolution
    sub = (np.arange(3) + 0.5) / 3 * res
    for gid in static_geoms(model):
        gtype = int(model.geom_type[gid])
        pos = data.geom_xpos[gid]
        rot = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        half = np.abs(rot) @ _half_extents(gtype, size)
        z0, z1 = max(z_range[0], pos[2] - half[2]), min(z_range[1], pos[2] + half[2])
        if z0 > z1:
            continue
        ix0, iy0 = grid.to_cell(pos[0] - half[0], pos[1] - half[1])
        ix1, iy1 = grid.to_cell(pos[0] + half[0], pos[1] + half[1])
        ny, nx = grid.shape
        ix0, iy0, ix1, iy1 = max(ix0, 0), max(iy0, 0), min(ix1, nx - 1), min(iy1, ny - 1)
        if ix0 > ix1 or iy0 > iy1:
            continue
        cx = grid.origin[0] + np.arange(ix0, ix1 + 1) * res
        cy = grid.origin[1] + np.arange(iy0, iy1 + 1) * res
        xs = (cx[:, None] + sub[None, :]).ravel()
        ys = (cy[:, None] + sub[None, :]).ravel()
        zs = np.unique(np.concatenate([np.arange(z0, z1, z_step), [z1]]))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        hit = _inside(gtype, size, (pts - pos) @ rot).reshape(len(xs), len(ys), len(zs)).any(axis=2)
        hit = hit.reshape(ix1 - ix0 + 1, 3, iy1 - iy0 + 1, 3).any(axis=(1, 3))
        occupied[iy0:iy1 + 1, ix0:ix1 + 1] |= hit.T
    return occupied


def build_prior_map(world_name: str, *, robot_height: float = 1.45, lidar_height: float = 0.175) -> PriorMap:
    import mujoco

    world = load_world(world_name)
    model = mujoco.MjModel.from_xml_path(str(world["world_path"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    res = world["map"]["resolution"]
    x0, y0, x1, y1 = world["map"]["bounds"]
    shape = (int(math.ceil((y1 - y0) / res)), int(math.ceil((x1 - x0) / res)))
    origin = np.array([x0, y0])
    occupancy = rasterize(model, data, Grid(origin, res, np.zeros(shape, dtype=bool)), (0.02, robot_height))
    # LiDAR slice at a finer resolution: it is what scan matching aligns against.
    lres = world["map"].get("lidar_resolution", res / 2)
    lshape = (int(math.ceil((y1 - y0) / lres)), int(math.ceil((x1 - x0) / lres)))
    lidar = rasterize(model, data, Grid(origin, lres, np.zeros(lshape, dtype=bool)),
                      (lidar_height - 0.01, lidar_height + 0.01), z_step=0.01)

    rooms = {name: Room(name, tuple(r["rect"]), list(r.get("aliases", [])),
                        tuple(r["search_from"]) if r.get("search_from") else None)
             for name, r in world.get("rooms", {}).items()}

    def room_of(x, y):
        return next((r.name for r in rooms.values() if r.contains(x, y)), None)

    surfaces = {}
    for name, s in world.get("surfaces", {}).items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s["geom"])
        if gid < 0:
            raise ValueError(f"surface {name!r} references unknown geom {s['geom']!r}")
        if model.geom_type[gid] != _GEOM_BOX:
            raise ValueError(f"surface {name!r} must be a box geom")
        rot = data.geom_xmat[gid].reshape(3, 3)
        pos = data.geom_xpos[gid].copy()
        size = model.geom_size[gid]
        surfaces[name] = Surface(name, s.get("kind", "table"), pos[:2], size[:2].copy(),
                                 math.atan2(rot[1, 0], rot[0, 0]), float(pos[2] + size[2]),
                                 room_of(*pos[:2]), list(s.get("aliases", [])))
    places = {}
    for name, p in world.get("places", {}).items():
        pose = p["pose"]
        places[name] = Place(name, np.array(pose[:2], dtype=float), pose[2] if len(pose) > 2 else None,
                             float(p.get("radius", 0.25)), p.get("kind", "waypoint"), list(p.get("aliases", [])))
    return PriorMap(world["name"], Grid(origin, res, occupancy), Grid(origin, lres, lidar), lidar_height,
                    rooms, surfaces, places, dict(world.get("user", {})), tuple(world["start_pose"]))


def get_prior_map(world_name: str, *, lidar_height: float = 0.175, robot_height: float = 1.45,
                  rebuild: bool = False) -> PriorMap:
    """Load build/maps/<world>_<lidar>_<height>.npz, rebuilding it when the world description changed.

    The map depends on the robot: what is an obstacle for a 1.4 m robot (a table top) is a roof
    for a 15 cm one, and the LiDAR slice sits at that robot's scanner height."""
    world = load_world(world_name)
    directory = BUILD_DIR / "maps"
    path = directory / f"{world['name']}_{int(round(lidar_height * 1000))}mm_h{int(round(robot_height * 100))}cm.npz"
    sources = [world["world_path"], world["dir"] / "semantic.yaml"]
    fresh = path.exists() and all(path.stat().st_mtime >= s.stat().st_mtime for s in sources)
    if fresh and not rebuild:
        return PriorMap.load(path)
    prior = build_prior_map(world_name, lidar_height=lidar_height, robot_height=robot_height)
    saved = prior.save(directory)
    saved.replace(path)
    (directory / f"{world['name']}.pgm").replace(path.with_suffix(".pgm"))
    return prior
