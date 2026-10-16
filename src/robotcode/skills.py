"""Skill library: the bounded set of actions a planner (rules or LLM) may request.

Each skill is a generator `skill(ctx, **args)` that commands the robot through
the stack's controllers, yields once per control tick, and returns a
SkillResult. Skills check their own preconditions and verify their outcome from
sensor data -- a skill never reports success it has not observed.

Registry metadata (description, parameters, required capabilities) is what the
LLM planner sees as its tool list and what plan validation checks against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .mapping import Surface, normalize_text
from .math_utils import to_base, to_map


@dataclass
class SkillResult:
    ok: bool
    message: str
    data: dict = field(default_factory=dict)


@dataclass
class SkillSpec:
    name: str
    func: Callable
    description: str
    params: dict[str, str]
    required: tuple[str, ...]
    needs: frozenset[str]


SKILLS: dict[str, SkillSpec] = {}


def skill(name: str, description: str, params: dict[str, str] | None = None, required=(), needs=()):
    def register(func):
        SKILLS[name] = SkillSpec(name, func, description, params or {}, tuple(required), frozenset(needs))
        return func
    return register


# ---------------------------------------------------------------------------
# Grounding language/plan arguments onto the map
# ---------------------------------------------------------------------------
@dataclass
class Target:
    kind: str                   # room | surface | surface_kind | place | user
    name: str
    surfaces: list[Surface] = field(default_factory=list)


def resolve_target(ctx, text: str) -> Target | None:
    """Map a name or alias ('desk', 'bàn làm việc', 'kitchen', 'me', 'table') to a map entity."""
    prior = ctx.prior
    t = normalize_text(text).replace("_", " ").strip()
    if t in {normalize_text(a) for a in ["user"] + prior.user.get("aliases", [])}:
        surface = prior.surfaces.get(prior.user.get("handover_surface", ""))
        return Target("user", "user", [surface] if surface else [])
    for name, room in prior.rooms.items():
        if t in {normalize_text(a) for a in [name.replace("_", " ")] + room.aliases}:
            return Target("room", name, [s for s in prior.surfaces.values() if s.room == name])
    for name, s in prior.surfaces.items():
        if t in {normalize_text(a) for a in [name.replace("_", " ")] + s.aliases}:
            return Target("surface", name, [s])
    for name, p in prior.places.items():
        if t in {normalize_text(a) for a in [name.replace("_", " ")] + p.aliases}:
            return Target("place", name)
    kinds = {s.kind for s in prior.surfaces.values()}
    generic = {"table": ["table", "ban", "cai ban"], "counter": ["counter", "quay"], "shelf": ["shelf", "ke"]}
    for kind, words in generic.items():
        if kind in kinds and t in words:
            return Target("surface_kind", kind, [s for s in prior.surfaces.values() if s.kind == kind])
    return None


def describe_object(otype: str | None, color: str | None) -> str:
    return " ".join(x for x in (color, otype) if x) or "object"


def _parse_object(ctx, text: str):
    otype, color = ctx.world.parse_object(text)
    if otype is None and color is None:
        return None
    return otype, color


# ---------------------------------------------------------------------------
# Primitive behaviours used by several skills
# ---------------------------------------------------------------------------
def _camera_sees(ctx, point_map: np.ndarray, pose: np.ndarray) -> bool:
    """Would a point be inside the camera frustum (and in range) from the current pose?"""
    cam = ctx.spec["sensors"]["head_camera"]
    cam_pos, cam_rot = ctx.kin.camera("head_camera", ctx.state.joints)
    p_base = np.array([*to_base(pose, point_map[None, :2])[0], point_map[2]])
    pc = cam_rot.T @ (p_base - cam_pos)
    depth = -pc[2]
    if depth < 0.3 or depth > 2.5:
        return False
    f = (cam["height"] / 2) / math.tan(math.radians(cam["fovy_deg"]) / 2)
    u = cam["width"] / 2 + f * pc[0] / depth
    v = cam["height"] / 2 - f * pc[1] / depth
    return 10 < u < cam["width"] - 10 and 10 < v < cam["height"] - 10


def look(ctx, *, settle: float = 0.3, forget: bool = True):
    """Generator: hold still, capture one RGB-D frame, detect and update the world model."""
    ctx.drive.halt()
    start = ctx.time
    while ctx.time - start < settle:
        yield
    frame = ctx.hal.capture_rgbd()
    exclude = []
    if ctx.world.held and ctx.arm:
        exclude.append(np.array([*to_map(ctx.pose, ctx.arm.gripper_position()[None, :2])[0], 0.0]))
    detections = ctx.detector.detect(frame, ctx.pose.copy(), ctx.state.joints, exclude=exclude)
    seen = ctx.world.integrate(detections)
    dropped = []
    if forget:
        pose = ctx.pose.copy()
        dropped = ctx.world.forget_unseen(lambda p: _camera_sees(ctx, p, pose), seen)
    if detections:
        ctx.log("saw: " + ", ".join(f"{d.label} on {d.support or 'floor'}" for d in detections))
    for obj_id in dropped:
        ctx.log(f"{obj_id} is no longer where it was last seen")
    return seen


def _settle(ctx, seconds: float):
    start = ctx.time
    while ctx.time - start < seconds:
        yield


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
@skill("navigate_to", "Drive to a room, a piece of furniture, a named place or the user.",
       {"target": "room / surface / place name, or 'user'"}, required=("target",), needs=("navigate",))
def navigate_to(ctx, target: str):
    tgt = resolve_target(ctx, target)
    if tgt is None:
        return SkillResult(False, f"unknown destination {target!r}")
    if ctx.arm is not None:
        yield from ctx.arm.stow(holding=ctx.world.held is not None)
    if tgt.kind == "room":
        room = ctx.prior.rooms[tgt.name]
        goal = np.array(room.search_from if room.search_from else room.center)
        res = yield from ctx.nav.go_to(goal)
    elif tgt.kind == "place":
        place = ctx.prior.places[tgt.name]
        res = yield from ctx.nav.go_to(place.xy, place.yaw)
    else:
        poses = [p for s in tgt.surfaces for p in ctx.nav.observation_poses(s)]
        if not poses:
            return SkillResult(False, f"no free spot near {tgt.name}")
        res = yield from ctx.nav.go_to(poses[0][:2], poses[0][2])
    if not res.ok:
        return SkillResult(False, f"could not reach {tgt.name}: {res.reason}")
    return SkillResult(True, f"arrived at {tgt.name}", {"room": ctx.prior.room_of(*ctx.pose[:2])})


@skill("look_around", "Capture camera images (turning in place) and report the objects seen.",
       {}, needs=("perceive",))
def look_around(ctx):
    seen = {}
    for k in range(6):
        for obj in (yield from look(ctx)):
            seen[obj.id] = obj
        if k < 5:
            yield from ctx.nav.turn_to(ctx.pose[2] + math.pi / 3)
    labels = sorted({o.label for o in seen.values()})
    return SkillResult(True, "I can see: " + (", ".join(labels) if labels else "nothing I recognise"),
                       {"objects": [o.id for o in seen.values()]})


@skill("find_object", "Search for an object, looking at the furniture of a room first if one is given.",
       {"object": "object description, e.g. 'red bottle'", "room": "optional room to search first",
        "surface": "optional furniture name or kind ('table') to search first"},
       required=("object",), needs=("navigate", "perceive"))
def find_object(ctx, object: str, room: str | None = None, surface: str | None = None):
    parsed = _parse_object(ctx, object)
    if parsed is None:
        return SkillResult(False, f"I do not know what {object!r} is")
    otype, color = parsed
    label = describe_object(otype, color)

    def matches():
        return ctx.world.query(otype, color)

    hint_room = resolve_target(ctx, room) if room else None
    hint_surface = resolve_target(ctx, surface) if surface else None
    candidates: list[Surface] = list(ctx.prior.surfaces.values())

    def priority(s: Surface):
        score = 0
        if hint_room is not None and hint_room.kind == "room" and s.room != hint_room.name:
            score += 2
        if hint_surface is not None and s not in hint_surface.surfaces:
            score += 1
        return score

    known = [o for o in matches()
             if (hint_room is None or hint_room.kind != "room" or o.room == hint_room.name)]
    if known:
        return SkillResult(True, f"I remember the {label} on {known[0].support or 'the floor'}",
                           {"object": known[0].id})
    candidates.sort(key=lambda s: (priority(s), float(np.linalg.norm(s.center - ctx.pose[:2]))))
    searched = []
    for s in candidates:
        poses = ctx.nav.observation_poses(s)
        if not poses:
            continue
        if ctx.arm is not None:
            yield from ctx.arm.stow(holding=ctx.world.held is not None)
        res = yield from ctx.nav.go_to(poses[0][:2], poses[0][2])
        if not res.ok:
            ctx.log(f"could not reach the {s.name} to look: {res.reason}")
            continue
        yield from look(ctx)
        found = matches()
        if not found and max(s.half) > 0.45:
            # Long furniture does not fit in one camera frame from close by: pan the base.
            heading = ctx.pose[2]
            for offset in (0.5, -0.5):
                turned = yield from ctx.nav.turn_to(heading + offset)
                if turned.ok:
                    yield from look(ctx)
                if matches():
                    break
            found = matches()
        searched.append(s.name)
        if found:
            obj = found[0]
            note = ""
            if hint_room is not None and hint_room.kind == "room" and obj.room != hint_room.name:
                note = f" (not in the {hint_room.name}: found in the {obj.room})"
            return SkillResult(True, f"found the {label} on the {obj.support or 'floor'}{note}",
                               {"object": obj.id, "searched": searched})
    return SkillResult(False, f"could not find a {label} after searching {', '.join(searched) or 'nothing'}",
                       {"searched": searched})


def _target_belief(ctx, object: str):
    parsed = _parse_object(ctx, object)
    if parsed is None:
        return None
    found = ctx.world.query(*parsed)
    return found[0] if found else None


def _choose_grasp_approach(ctx, obj, poses: list[np.ndarray], x0: float):
    """Pick the approach whose open-finger sweep misses the other objects nearby.

    Returns (base pose, gripper opening or None for fully open). Falls back to a narrower opening
    (object diameter + 3 cm) when every approach would brush a neighbour."""
    g = ctx.spec["gripper"]
    t = g["finger"]["thickness"]
    radius = ctx.catalog["types"][obj.type]["radius"]
    others = [(o.position[:2], ctx.catalog["types"][o.type]["radius"]) for o in ctx.world.objects.values()
              if o.id != obj.id and not o.held]

    def clearance(pose, half_span):
        heading = np.array([np.cos(pose[2]), np.sin(pose[2])])
        start = pose[:2] + heading * (x0 - 0.05)
        end = obj.position[:2] + heading * 0.05
        seg = end - start
        best = np.inf
        for pos, r in others:
            u = np.clip(np.dot(pos - start, seg) / max(np.dot(seg, seg), 1e-9), 0.0, 1.0)
            best = min(best, float(np.linalg.norm(pos - (start + u * seg))) - r - half_span)
        return best

    for opening in (None, 2 * radius + 0.035):
        half = (g["max_opening"] if opening is None else opening) / 2 + t
        for pose in poses:
            if clearance(pose, half) > 0.01:
                return pose, opening
    return None, None


def _grasp_height(ctx, otype: str) -> float:
    t = ctx.catalog["types"][otype]
    height = 2 * t["radius"] if t["shape"] == "sphere" else t["height"]
    finger_half = ctx.spec["gripper"]["finger"]["height"] / 2
    return max(0.55 * height, finger_half + 0.022)


@skill("pick_object", "Approach a previously found object, grasp it and lift it into the carry pose.",
       {"object": "object description, e.g. 'red bottle'"}, required=("object",),
       needs=("navigate", "perceive", "manipulate"))
def pick_object(ctx, object: str):
    if ctx.world.held:
        return SkillResult(False, f"already holding {ctx.world.objects[ctx.world.held].label}")
    obj = _target_belief(ctx, object)
    if obj is None:
        return SkillResult(False, f"I have not seen a {object} yet; it must be found first",
                           {"needs": "find_object"})
    otype = obj.type
    radius = ctx.catalog["types"][otype]["radius"]
    reach_lo, reach_hi = ctx.kin.limits["reach"]
    x0 = ctx.kin.site("grasp_site", {**ctx.kin.default_joints(), "reach": reach_lo})[0][0]
    for attempt in range(1, 3):
        surface = ctx.prior.surfaces.get(obj.support) if obj.support else None
        poses = ctx.nav.manipulation_poses(obj.position[:2], surface, reach_min=x0 + 0.04,
                                           reach_max=x0 + reach_hi - 0.03, ideal_reach=x0 + 0.15)
        if not poses:
            return SkillResult(False, f"no free spot to stand and reach the {obj.label}")
        pose, opening = _choose_grasp_approach(ctx, obj, poses, x0)
        if pose is None:
            blockers = ", ".join(sorted({o.label for o in ctx.world.objects.values() if o.id != obj.id
                                         and o.support == obj.support and not o.held}))
            return SkillResult(False, f"cannot reach the {obj.label} without knocking over the {blockers} next "
                                      f"to it; please move it or ask me to move it first")
        yield from ctx.arm.stow()
        yield from ctx.gripper.open(opening)
        nav = yield from ctx.nav.approach(pose)
        if not nav.ok:
            ctx.log(f"approach failed: {nav.reason}")
            continue
        # Re-observe from close range: the object may have moved, or the map pose be slightly off.
        seen = yield from look(ctx, forget=False)
        close = [o for o in seen if o.id == obj.id]
        if not close:
            if obj.id not in ctx.world.objects:
                return SkillResult(False, f"the {obj.label} is no longer there", {"needs": "find_object"})
            ctx.log(f"cannot see the {obj.label} from the approach pose (attempt {attempt})")
            continue
        obj = close[0]
        # Neighbours may only become visible from close by: re-check the finger sweep from here.
        clear_here, opening_here = _choose_grasp_approach(ctx, obj, [ctx.pose.copy()], x0)
        if clear_here is None:
            ctx.log(f"from here the fingers would hit a neighbour of the {obj.label}; trying another approach")
            yield from ctx.arm.stow()
            continue
        if opening_here != opening:
            opening = opening_here
            ctx.log(f"neighbouring object close to the {obj.label}: narrowing the gripper to "
                    f"{(opening or ctx.gripper.max_opening) * 100:.0f} cm")
            yield from ctx.gripper.open(opening)
        support_z = ctx.prior.surfaces[obj.support].top_z if obj.support else 0.0
        target_xy = to_base(ctx.pose, obj.position[None, :2])[0]
        finger_len = ctx.spec["gripper"]["finger"]["length"]
        # Keep the palm off the object; a sphere must be pinched through its centre to be stable.
        palm_gap = 0.003 if ctx.catalog["types"][otype]["shape"] == "sphere" else 0.012
        target_xy[0] -= max(0.0, radius + palm_gap - finger_len / 2)
        grasp_z = support_z + _grasp_height(ctx, otype)
        target = np.array([target_xy[0], target_xy[1], grasp_z])
        q, err = ctx.kin.ik(target, ctx.state.joints)
        if err > 0.01:
            ctx.log(f"{obj.label} out of reach from here (IK error {err:.3f} m)")
            continue
        # Raise to grasp height first (clear of the table), centre laterally, then extend.
        err = yield from ctx.arm.move_to(target, order=("lift", "side"))
        if err is None or err > 0.02:
            ctx.log(f"arm could not reach the grasp pose (error {err})")
            yield from ctx.arm.stow()
            continue
        check = yield from ctx.gripper.close()
        expected = 2 * radius
        # An off-centre grasp closes on a chord narrower than the object: unstable, so reject it.
        if not check.holding or abs(check.width - expected) > 0.010:
            ctx.log(f"grasp failed: forces {check.forces[0]:.1f}/{check.forces[1]:.1f} N, "
                    f"width {check.width:.3f} m (expected {expected:.3f})")
            yield from ctx.gripper.open()
            yield from ctx.arm.stow()
            continue
        yield from ctx.arm.move_joints({"lift": ctx.state.joints["lift"] + 0.06}, speed_scale=0.5)
        yield from _settle(ctx, 0.3)
        if not ctx.gripper.check().holding:
            ctx.log("object slipped out of the gripper while lifting")
            yield from ctx.gripper.open()
            yield from ctx.arm.stow()
            yield from look(ctx)
            continue
        ctx.world.mark_held(obj.id)
        ctx.world.objects[obj.id].history.append({"grasp_offset": float(_grasp_height(ctx, otype))})
        yield from ctx.arm.stow(holding=True)
        if not ctx.gripper.check().holding:
            ctx.world.mark_released([ctx.pose[0], ctx.pose[1], 0.0], None, ctx.time)
            return SkillResult(False, f"dropped the {obj.label} while retracting the arm")
        return SkillResult(True, f"picked up the {obj.label} (grip width {check.width * 100:.1f} cm, "
                                 f"forces {check.forces[0]:.0f}/{check.forces[1]:.0f} N)",
                           {"object": obj.id, "attempts": attempt})
    return SkillResult(False, f"failed to pick up the {obj.label} after 2 attempts")


def _placement_points(ctx, surface: Surface, clearance: float = 0.12) -> list[np.ndarray]:
    """Free spots on a surface, 0.12-0.2 m in from an edge, away from known objects, best first."""
    others = [o.position[:2] for o in ctx.world.objects.values() if not o.held and o.support == surface.name]
    user = np.array(ctx.prior.user.get("position", [np.inf, np.inf]), dtype=float)
    points = []
    for mid, normal, half_len in surface.edges():
        tangent = np.array([-normal[1], normal[0]])
        for depth in (0.14, 0.2):
            for lateral in np.linspace(-half_len + 0.08, half_len - 0.08, 7):
                p = mid - normal * depth + tangent * lateral
                if not surface.contains(p[0], p[1], margin=-0.06):
                    continue
                if any(np.linalg.norm(p - o) < clearance for o in others):
                    continue
                points.append((abs(lateral) + float(np.linalg.norm(p - user)) * 0.3, p))
    return [p for _, p in sorted(points, key=lambda item: item[0])]


@skill("place_object", "Put the held object down on a surface, a floor spot, or within the user's reach.",
       {"target": "surface / place name, a room, or 'user'"}, required=("target",),
       needs=("navigate", "manipulate"))
def place_object(ctx, target: str):
    if not ctx.world.held:
        return SkillResult(False, "I am not holding anything")
    obj = ctx.world.objects[ctx.world.held]
    tgt = resolve_target(ctx, target)
    if tgt is None:
        return SkillResult(False, f"unknown place {target!r}")
    offset = obj.history[-1]["grasp_offset"] if obj.history else _grasp_height(ctx, obj.type)
    reach_lo, reach_hi = ctx.kin.limits["reach"]
    x0 = ctx.kin.site("grasp_site", {**ctx.kin.default_joints(), "reach": reach_lo})[0][0]
    options = []                # (surface or None, point, support_z)
    if tgt.kind == "place":
        place = ctx.prior.places[tgt.name]
        options.append((None, place.xy.copy(), 0.0))
    else:
        surfaces = tgt.surfaces
        if tgt.kind == "room":
            surfaces = [s for s in surfaces if s.kind in ("table", "counter")]
        for s in sorted(surfaces, key=lambda s: float(np.linalg.norm(s.center - ctx.pose[:2]))):
            options.extend((s, p, s.top_z) for p in _placement_points(ctx, s))
    if not options:
        return SkillResult(False, f"no free space to put the {obj.label} on {tgt.name}")
    for surface, point, support_z in options[:6]:
        poses = ctx.nav.manipulation_poses(point, surface, reach_min=x0 + 0.04, reach_max=x0 + reach_hi - 0.03,
                                           ideal_reach=x0 + 0.15)
        if not poses:
            continue
        nav = yield from ctx.nav.approach(poses[0])
        if not nav.ok:
            ctx.log(f"approach for placing failed: {nav.reason}")
            continue
        target_xy = to_base(ctx.pose, point[None, :2])[0]
        release_z = support_z + offset + 0.012
        # Near the floor the gripper may not go low enough: release from slightly higher (<= 5 cm drop).
        lowest = ctx.kin.site("grasp_site", {**ctx.state.joints, "lift": ctx.kin.limits["lift"][0]})[0][2] + 0.005
        if release_z < lowest:
            if lowest - (support_z + offset) > 0.05:
                return SkillResult(False, f"the gripper cannot get low enough to put the {obj.label} down there")
            release_z = lowest
        hover = np.array([target_xy[0], target_xy[1], release_z + 0.04])
        err = yield from ctx.arm.move_to(hover, order=("lift", "side"))
        if err is None or err > 0.025:
            ctx.log(f"cannot reach the placing pose (error {err})")
            yield from ctx.arm.stow(holding=True)
            continue
        yield from ctx.arm.move_joints({"lift": max(ctx.state.joints["lift"] - 0.04, ctx.kin.limits["lift"][0])},
                                       speed_scale=0.3)
        yield from ctx.gripper.open()
        yield from _settle(ctx, 0.4)
        yield from ctx.arm.move_joints({"lift": ctx.state.joints["lift"] + 0.05}, speed_scale=0.5)
        released_at = to_map(ctx.pose, ctx.arm.gripper_position()[None, :2])[0]
        support_name = surface.name if surface else None
        ctx.world.mark_released([released_at[0], released_at[1], support_z + offset], support_name, ctx.time)
        yield from ctx.arm.stow()
        # Verify with the camera: back away until the release point is inside the camera's view.
        target3 = np.array([point[0], point[1], support_z + 0.05])
        backed = 0.0
        while not _camera_sees(ctx, target3, ctx.pose) and backed < 0.7:
            step = yield from ctx.nav.back_off(0.1)
            if not step:
                break
            backed += 0.1
        seen = yield from look(ctx, forget=False)
        here = [o for o in seen if o.id == obj.id
                and np.linalg.norm(o.position[:2] - point) < 0.15]
        where = surface.name if surface else tgt.name
        if not here:
            return SkillResult(False, f"released the {obj.label} but cannot see it on the {where}")
        return SkillResult(True, f"put the {obj.label} on the {where}", {"object": obj.id, "where": where})
    return SkillResult(False, f"could not reach any free spot on {tgt.name}")


@skill("verify_object_at", "Look at a place and confirm that an object is there.",
       {"object": "object description", "target": "surface / place / 'user'"}, required=("object", "target"),
       needs=("navigate", "perceive"))
def verify_object_at(ctx, object: str, target: str):
    parsed = _parse_object(ctx, object)
    tgt = resolve_target(ctx, target)
    if parsed is None or tgt is None:
        return SkillResult(False, "cannot verify: unknown object or place")
    names = {s.name for s in tgt.surfaces}
    for s in tgt.surfaces[:2]:
        poses = ctx.nav.observation_poses(s)
        if not poses:
            continue
        res = yield from ctx.nav.go_to(poses[0][:2], poses[0][2])
        if not res.ok:
            continue
        yield from look(ctx)
        hits = [o for o in ctx.world.query(*parsed) if o.support in names]
        if hits:
            return SkillResult(True, f"confirmed: the {hits[0].label} is on the {hits[0].support}")
    if tgt.kind == "place":
        place = ctx.prior.places[tgt.name]
        hits = [o for o in ctx.world.query(*parsed) if np.linalg.norm(o.position[:2] - place.xy) < place.radius + 0.1]
        if hits:
            return SkillResult(True, f"confirmed: the {hits[0].label} is at the {tgt.name}")
    return SkillResult(False, f"could not confirm the {object} at {tgt.name}")


@skill("get_robot_status", "Report battery, position, what the robot holds and the objects it remembers.")
def get_robot_status(ctx):
    st = ctx.state
    room = ctx.prior.room_of(*ctx.pose[:2]) or "unknown"
    held = ctx.world.objects[ctx.world.held].label if ctx.world.held else "nothing"
    msg = (f"battery {st.battery.soc:.0%} ({st.battery.voltage:.1f} V), in the {room} at "
           f"({ctx.pose[0]:.2f}, {ctx.pose[1]:.2f}), holding {held}; known objects: "
           + ("; ".join(ctx.world.summary()) or "none"))
    return SkillResult(True, msg)
    yield  # pragma: no cover  (makes this a generator)


@skill("stop_robot", "Stop all motion immediately.")
def stop_robot(ctx):
    ctx.drive.halt()
    return SkillResult(True, "stopped")
    yield  # pragma: no cover


@skill("say", "Tell the user something (status, a question, an explanation).", {"message": "text"},
       required=("message",))
def say(ctx, message: str):
    ctx.log(f"ROBOT SAYS: {message}")
    return SkillResult(True, message)
    yield  # pragma: no cover
