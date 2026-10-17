"""Perception from the RGB-D head camera, plus the robot's memory of objects.

Two detectors share one geometric back end (GeometricDetector) and the same
`detect(frame, pose, joints, exclude) -> list[Detection]` signature:

* ObjectDetector (`color`, default) -- a classical pipeline on pixels only:
  colour segmentation (HSV) -> connected components -> back-projection with the
  depth image and camera intrinsics -> transform to the map frame (camera
  kinematics + localisation) -> geometric classification against the object
  catalog (height / width) -> object centre from the visible top surface.
* LearnedObjectDetector (`learned`, perception_learned.py) -- an open-vocabulary
  model (OWLv2) proposes 2D boxes for text queries built from the catalog; the
  same depth back-projection and geometry turn each box into a 3D object.
  Optional dependency: pip install -e '.[vision]'.

Which one runs: `make_detector()` (CLI --detector > robot YAML perception.detector
> catalog perception.detector > color).

WorldModel keeps beliefs about objects: where each was last seen, what the
robot is holding, and forgets objects that are no longer where it expected.
"""
from __future__ import annotations

import colorsys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import ConfigError
from .hal import RGBDFrame
from .mapping import PriorMap, Surface, normalize_text
from .math_utils import to_map

DETECTORS = ("color", "learned")


@dataclass
class Detection:
    color: str
    type: str
    position: np.ndarray        # object centre, map frame
    top_z: float
    height: float
    width: float
    pixels: int
    support: str | None         # surface name, or None for the floor
    stamp: float
    fit_error: float

    @property
    def label(self) -> str:
        return f"{self.color} {self.type}"


@dataclass
class ObjectBelief:
    id: str
    color: str
    type: str
    position: np.ndarray
    support: str | None
    room: str | None
    last_seen: float
    seen_count: int = 1
    held: bool = False
    history: list = field(default_factory=list)
    fit_error: float = float("inf")     # best shape-classification fit seen so far

    @property
    def label(self) -> str:
        return f"{self.color} {self.type}"


@dataclass
class Blob:
    """One candidate object's pixels lifted to the map frame, with its measured size and shape."""
    xy: np.ndarray              # (N, 2) points, map frame
    z: np.ndarray               # (N,) point heights
    center_guess: np.ndarray    # median of xy
    surface: Surface | None     # what it stands on (None = floor)
    support_z: float
    top_z: float
    height: float
    width: float                # extent across the viewing direction
    taper: float | None         # top width / middle width: ~0.7 sphere, ~1 cylinder
    bearing: np.ndarray         # unit vector camera -> object, map frame
    pixels: int


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=2), x.min(axis=2)
    delta = mx - mn
    s = np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6), 0.0)
    safe = np.maximum(delta, 1e-6)
    h = np.where(mx == r, ((g - b) / safe) % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    return (h * 60.0) % 360.0, s, mx


class GeometricDetector:
    """Shared back end: an object's pixels -> 3D points in the map frame -> size and shape -> catalog type.

    Subclasses decide which pixels belong to one object (`_detect`); this class turns them into Detections.
    """

    kind = ""
    MAX_FIT_ERROR = 0.9

    def __init__(self, ctx):
        self.ctx = ctx
        cat = ctx.catalog
        p = cat.get("perception", {})
        self.min_pixels = p.get("min_pixels", 12)
        self.max_range = p.get("max_range", 4.0)
        self.types = {}
        self.shapes = {}
        for name, t in cat["types"].items():
            height = 2 * t["radius"] if t["shape"] == "sphere" else t["height"]
            self.types[name] = (height, 2 * t["radius"], t["radius"])
            self.shapes[name] = t["shape"]
        self.frames = 0
        self.busy_s = 0.0           # total time spent in detect()

    def detect(self, frame: RGBDFrame, pose: np.ndarray, joints: dict[str, float],
               exclude: list[np.ndarray] | None = None) -> list[Detection]:
        start = time.perf_counter()
        self.frames += 1
        cam_pos, cam_rot = self.ctx.kin.camera(frame.camera, joints)
        cam_pos = cam_pos + np.array([0.0, 0.0, self.ctx.base_height])     # base frame -> height above floor
        detections = [d for d in self._detect(frame, pose, cam_pos, cam_rot)
                      if not (exclude and any(np.linalg.norm(d.position[:2] - e[:2]) < 0.12 for e in exclude))]
        self.busy_s += time.perf_counter() - start
        return detections

    def _detect(self, frame: RGBDFrame, pose, cam_pos, cam_rot) -> list[Detection]:
        raise NotImplementedError

    # ---- geometry shared by all detectors --------------------------------------------------
    @staticmethod
    def back_project(us, vs, d, frame: RGBDFrame, cam_pos, cam_rot, pose) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (u, v) with depth d -> (N, 2) map-frame xy and (N,) heights."""
        xc = (us + 0.5 - frame.cx) / frame.fx * d
        yc = -(vs + 0.5 - frame.cy) / frame.fy * d
        pts_cam = np.stack([xc, yc, -d], axis=1)
        pts_base = pts_cam @ cam_rot.T + cam_pos
        return to_map(pose, pts_base[:, :2]), pts_base[:, 2]

    def measure(self, xy, z, cam_pos, pose, surface: Surface | None = None) -> Blob | None:
        center_guess = np.median(xy, axis=0)
        top_z = float(np.percentile(z, 97))
        if surface is None:
            surface = self.ctx.prior.surface_at(center_guess[0], center_guess[1], margin=0.06)
            if surface is not None and top_z < surface.top_z + 0.02:
                surface = None          # below the table top: the object is on the floor under it
        support_z = surface.top_z if surface else 0.0
        height = top_z - support_z
        if height < 0.025:
            return None
        cam_xy = to_map(pose, cam_pos[None, :2])[0]
        bearing = center_guess - cam_xy
        bearing /= max(np.linalg.norm(bearing), 1e-6)
        normal = np.array([-bearing[1], bearing[0]])
        proj = xy @ normal
        width = float(np.percentile(proj, 97) - np.percentile(proj, 3))
        # Shape cue: a sphere narrows towards its top, a cylinder does not.
        upper = z > support_z + 0.8 * height
        middle = np.abs(z - (support_z + 0.5 * height)) < 0.15 * height
        taper = None
        if upper.sum() >= 4 and middle.sum() >= 4:
            w_mid = np.percentile(proj[middle], 95) - np.percentile(proj[middle], 5)
            if w_mid > 0.02:
                taper = float((np.percentile(proj[upper], 95) - np.percentile(proj[upper], 5)) / w_mid)
        return Blob(xy, z, center_guess, surface, support_z, top_z, height, width, taper, bearing, len(z))

    def fit_errors(self, blob: Blob) -> dict[str, float]:
        """How badly each catalog type explains the measured size and shape (0 = perfect)."""
        errors = {}
        for name, (h_t, w_t, _) in self.types.items():
            err = abs(blob.height - h_t) / h_t + 0.5 * abs(blob.width - w_t) / w_t
            if blob.taper is not None:
                err += 0.8 * abs(blob.taper - (0.68 if self.shapes[name] == "sphere" else 0.97))
            errors[name] = err
        return errors

    def make_detection(self, color: str, otype: str, fit_error: float, blob: Blob, stamp: float) -> Detection:
        h_t, _, radius = self.types[otype]
        top = blob.z >= blob.top_z - 0.01
        if top.sum() >= 4:
            # The top band also catches the rim's camera-facing side: shift back along the view ray.
            center = blob.xy[top].mean(axis=0) + blob.bearing * 0.35 * radius
        else:
            center = blob.center_guess + blob.bearing * 0.6 * radius
        return Detection(color, otype, np.array([center[0], center[1], blob.support_z + h_t / 2]), blob.top_z,
                         blob.height, blob.width, int(blob.pixels), blob.surface.name if blob.surface else None,
                         stamp, float(fit_error))


class ObjectDetector(GeometricDetector):
    """Colour detector: HSV segmentation of the catalog colours; type from metric size and shape."""

    kind = "color"

    def __init__(self, ctx):
        super().__init__(ctx)
        p = ctx.catalog.get("perception", {})
        self.min_s, self.min_v = p.get("min_saturation", 0.5), p.get("min_value", 0.18)
        self.hue_tol = p.get("hue_tolerance_deg", 14)
        self.hues = {}
        for name, c in ctx.catalog["colors"].items():
            h, s, v = colorsys.rgb_to_hsv(*c["rgba"][:3])
            if s >= self.min_s and v >= self.min_v:      # white, grey, black: no hue to segment by
                self.hues[name] = h * 360.0

    def _detect(self, frame: RGBDFrame, pose, cam_pos, cam_rot) -> list[Detection]:
        hue, sat, val = _hsv(frame.rgb)
        depth = frame.depth
        valid = (depth > 0) & (depth < self.max_range) & (sat >= self.min_s) & (val >= self.min_v)
        detections = []
        for color, h0 in self.hues.items():
            dh = np.abs((hue - h0 + 180.0) % 360.0 - 180.0)
            mask = valid & (dh <= self.hue_tol)
            if mask.sum() < self.min_pixels:
                continue
            labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
            for k in range(1, n + 1):
                vs, us = np.nonzero(labels == k)
                if len(vs) < self.min_pixels:
                    continue
                det = self._component(color, us, vs, depth[vs, us], frame, cam_pos, cam_rot, pose)
                if det is not None:
                    detections.append(det)
        return detections

    def _component(self, color, us, vs, d, frame: RGBDFrame, cam_pos, cam_rot, pose) -> Detection | None:
        # Silhouette pixels mix object colour with background depth: keep the object's depth layer.
        med = float(np.median(d))
        keep = np.abs(d - med) < 0.12
        us, vs, d = us[keep], vs[keep], d[keep]
        if len(d) < self.min_pixels:
            return None
        xy, z = self.back_project(us, vs, d, frame, cam_pos, cam_rot, pose)
        blob = self.measure(xy, z, cam_pos, pose)
        if blob is None:
            return None
        errors = self.fit_errors(blob)
        best = min(errors, key=errors.get)
        if errors[best] > self.MAX_FIT_ERROR:
            return None
        return self.make_detection(color, best, errors[best], blob, frame.stamp)


ColorDetector = ObjectDetector


def detector_kind(spec: dict, catalog: dict, override: str | None = None) -> str:
    """Which detector to run: explicit choice > robot spec > object catalog > colour."""
    kind = (override or spec.get("perception", {}).get("detector")
            or catalog.get("perception", {}).get("detector") or "color")
    if kind not in DETECTORS:
        raise ConfigError(f"unknown detector {kind!r}; choose one of {', '.join(DETECTORS)}")
    return kind


def make_detector(ctx, kind: str | None = None) -> GeometricDetector:
    kind = detector_kind(ctx.spec, ctx.catalog, kind)
    if kind == "learned":
        from .perception_learned import LearnedObjectDetector
        return LearnedObjectDetector(ctx)
    return ObjectDetector(ctx)


class WorldModel:
    MATCH_DISTANCE = 0.25

    def __init__(self, prior: PriorMap, catalog: dict):
        self.prior = prior
        self.catalog = catalog
        self.objects: dict[str, ObjectBelief] = {}
        self.held: str | None = None
        self._counter = 0

    # ---- language grounding ---------------------------------------------------------
    def parse_object(self, text: str) -> tuple[str | None, str | None]:
        """'chai nước màu đỏ' -> ('bottle', 'red'). Returns (type, color); None when unspecified."""
        t = f" {normalize_text(text)} "
        found_type, found_color = None, None
        best_len = 0
        for name, spec in self.catalog["types"].items():
            for alias in [name] + spec.get("aliases", []):
                a = normalize_text(alias)
                if f" {a} " in t and len(a) > best_len:
                    found_type, best_len = name, len(a)
        best_len = 0
        for name, spec in self.catalog["colors"].items():
            for alias in [name] + spec.get("aliases", []):
                a = normalize_text(alias)
                if f" {a} " in t and len(a) > best_len:
                    found_color, best_len = name, len(a)
        return found_type, found_color

    def query(self, otype: str | None = None, color: str | None = None, *, room: str | None = None,
              surface: str | None = None, include_held: bool = False) -> list[ObjectBelief]:
        out = [o for o in self.objects.values()
               if (otype is None or o.type == otype) and (color is None or o.color == color)
               and (room is None or o.room == room) and (surface is None or o.support == surface)
               and (include_held or not o.held)]
        return sorted(out, key=lambda o: -o.last_seen)

    # ---- belief update ----------------------------------------------------------------
    SAME_SPOT = 0.045       # centres of two solid objects are >= 6.6 cm apart; detections err by ~1 cm

    def _confusable(self, a: str, b: str) -> bool:
        """Shapes the classifier can mix up from some viewpoints (similar height: cup / apple / can),
        unlike a 20 cm bottle vs a 10 cm cup."""
        types = self.catalog["types"]

        def height(t):
            return 2 * types[t]["radius"] if types[t]["shape"] == "sphere" else types[t]["height"]
        return abs(height(a) - height(b)) < 0.045

    def integrate(self, detections: list[Detection]) -> list[ObjectBelief]:
        touched = []
        for det in detections:
            match, best = None, self.MATCH_DISTANCE
            for o in self.objects.values():
                if o.held or o.color != det.color:
                    continue
                dist = float(np.linalg.norm(o.position[:2] - det.position[:2]))
                if o.type != det.type and (dist >= self.SAME_SPOT or not self._confusable(o.type, det.type)):
                    continue
                if dist < best:
                    match, best = o, dist
            if match is not None and match.type != det.type:
                # Seen from another angle the shape classifier changed its mind (cup vs apple): keep
                # one object and the type of the better geometric fit.
                if det.fit_error < match.fit_error:
                    match.type = det.type
            if match is None:
                self._counter += 1
                match = ObjectBelief(f"{det.color}_{det.type}_{self._counter}", det.color, det.type,
                                     det.position.copy(), det.support,
                                     self.prior.room_of(*det.position[:2]), det.stamp, 0)
                self.objects[match.id] = match
            # Newer and closer observations dominate.
            w = 0.6 if match.seen_count else 1.0
            match.position = (1 - w) * match.position + w * det.position
            match.support, match.room = det.support, self.prior.room_of(*match.position[:2])
            match.last_seen, match.seen_count = det.stamp, match.seen_count + 1
            match.fit_error = min(match.fit_error, det.fit_error)
            touched.append(match)
        return touched

    def forget_unseen(self, visible, seen: list[ObjectBelief]) -> list[str]:
        """Drop beliefs that should have been visible in this frame but were not detected."""
        dropped = []
        seen_ids = {o.id for o in seen}
        for o in list(self.objects.values()):
            if o.held or o.id in seen_ids:
                continue
            if visible(o.position):
                dropped.append(o.id)
                del self.objects[o.id]
        return dropped

    def mark_held(self, obj_id: str) -> None:
        self.held = obj_id
        self.objects[obj_id].held = True

    def mark_released(self, position: np.ndarray, support: str | None, stamp: float) -> ObjectBelief | None:
        if self.held is None:
            return None
        obj = self.objects[self.held]
        obj.held, obj.position, obj.support, obj.last_seen = False, np.array(position, dtype=float), support, stamp
        obj.room = self.prior.room_of(*obj.position[:2])
        self.held = None
        return obj

    def summary(self) -> list[str]:
        out = []
        for o in sorted(self.objects.values(), key=lambda o: o.id):
            where = "in the gripper" if o.held else (f"on {o.support}" if o.support else "on the floor")
            out.append(f"{o.label} {where} ({o.room or 'unknown room'}) at ({o.position[0]:.2f}, {o.position[1]:.2f})")
        return out
