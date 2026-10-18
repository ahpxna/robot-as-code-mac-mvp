"""Compose a simulation scene: static world MJCF + generated robot + task objects.

This is the "infrastructure as code" step: a task file names a world, a robot
and the objects to spawn (with optional randomisation); this module turns that
into one MuJoCo model and writes the generated XML to build/scenes/ so every run
can be inspected and reproduced from (task, seed).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .config import BUILD_DIR, ConfigError, load_catalog, load_robot, load_world
from .description import el, mount_height, robot_mjcf
from .mapping import PriorMap, get_prior_map

APPROACH_CLEARANCE = 0.42      # free space needed in front of a surface edge for a robot to stand
DEFAULT_MAX_DEPTH = 0.28       # how far from an accessible edge a spawned object may lie


@dataclass
class ObjectSpawn:
    name: str
    type: str
    color: str
    xyz: np.ndarray
    yaw: float
    support: str               # surface / place name or "floor"


@dataclass
class Scene:
    task: dict
    seed: int
    robot_spec: dict
    world: dict
    prior: PriorMap
    catalog: dict
    objects: list[ObjectSpawn]
    robot_pose: tuple[float, float, float]
    xml: str

    def compile(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string(self.xml)
        return model, mujoco.MjData(model)


def object_height(otype: dict) -> float:
    return 2 * otype["radius"] if otype["shape"] == "sphere" else otype["height"]


def object_body(spawn: ObjectSpawn, catalog: dict) -> ET.Element:
    otype = catalog["types"][spawn.type]
    rgba = catalog["colors"][spawn.color]["rgba"]
    body = ET.Element("body", name=spawn.name, pos=" ".join(f"{v:.5f}" for v in spawn.xyz),
                      euler=f"0 0 {spawn.yaw:.4f}")
    el(body, "freejoint", name=f"{spawn.name}_free")
    rolling = otype.get("rolling_friction")
    # condim 6 adds rolling resistance (needed for round things); 4 = sliding + torsional friction.
    common = dict(name=f"{spawn.name}_geom", mass=otype["mass"], rgba=rgba, condim=6 if rolling else 4,
                  friction=[otype.get("friction", 1.0), 0.01, rolling or 0.0005])
    if otype["shape"] == "cylinder":
        el(body, "geom", type="cylinder", size=[otype["radius"], otype["height"] / 2], **common)
    elif otype["shape"] == "sphere":
        el(body, "geom", type="sphere", size=[otype["radius"]], **common)
    else:
        raise ConfigError(f"unsupported object shape {otype['shape']!r}")
    return body


def _reachable(surface, xy: np.ndarray, prior: PriorMap, max_depth: float) -> bool:
    for mid, normal, half_len in surface.edges():
        depth = float(np.dot(mid - xy, normal))          # distance from the object to this edge
        if not 0.0 <= depth <= max_depth:
            continue
        along = xy - mid + normal * depth                  # projection of xy onto the edge line
        if np.linalg.norm(along) > half_len:
            continue
        stand = xy + normal * (depth + APPROACH_CLEARANCE)
        if prior.distance.value(stand[0], stand[1]) >= APPROACH_CLEARANCE - 0.06:
            return True
    return False


def place_objects(task: dict, prior: PriorMap, catalog: dict, rng: np.random.Generator) -> list[ObjectSpawn]:
    spawns: list[ObjectSpawn] = []
    for spec in task.get("objects", []):
        name, otype_name, color = spec["name"], spec["type"], spec["color"]
        if otype_name not in catalog["types"]:
            raise ConfigError(f"object {name!r}: unknown type {otype_name!r}")
        if color not in catalog["colors"]:
            raise ConfigError(f"object {name!r}: unknown colour {color!r}")
        otype = catalog["types"][otype_name]
        radius = otype["radius"]
        # `on:` is accepted too, but YAML 1.1 parses a bare `on` key as boolean True.
        support = spec.get("support", spec.get("on", spec.get(True, "floor")))
        surface = prior.surfaces.get(support)
        if surface is None and support != "floor" and support not in prior.places:
            raise ConfigError(f"object {name!r}: unknown support {support!r}")
        base_z = surface.top_z if surface else 0.0
        jitter = float(spec.get("jitter", 0.0))
        max_depth = float(spec.get("max_depth", DEFAULT_MAX_DEPTH))
        for _ in range(2000):
            if "at" in spec:
                xy = np.array(spec["at"], dtype=float) + rng.uniform(-jitter, jitter, 2)
            elif support in prior.places:
                place = prior.places[support]
                xy = place.xy + rng.uniform(-jitter, jitter, 2)
            elif surface is not None:
                local = rng.uniform(-1, 1, 2) * np.maximum(surface.half - radius - 0.04, 0.0)
                c, s = math.cos(surface.yaw), math.sin(surface.yaw)
                xy = surface.center + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
            else:
                raise ConfigError(f"object {name!r} on the floor needs 'at'")
            if surface is not None:
                if not surface.contains(xy[0], xy[1], margin=-(radius + 0.03)):
                    continue
                if not _reachable(surface, xy, prior, max_depth):
                    continue
            if any(np.linalg.norm(xy - o.xyz[:2]) < 0.13 for o in spawns):
                continue
            break
        else:
            raise ConfigError(f"could not find a valid spawn position for {name!r}")
        z = base_z + object_height(otype) / 2 + 0.002
        spawns.append(ObjectSpawn(name, otype_name, color, np.array([xy[0], xy[1], z]),
                                  float(rng.uniform(-math.pi, math.pi)), support))
    return spawns


def robot_start(task: dict, prior: PriorMap, rng: np.random.Generator) -> tuple[float, float, float]:
    start = task.get("robot_start", {})
    pose = np.array(start.get("pose", prior.start_pose), dtype=float)
    jitter = np.array(start.get("jitter", [0.0, 0.0, 0.0]), dtype=float)
    for _ in range(200):
        candidate = pose + rng.uniform(-1, 1, 3) * jitter
        if prior.distance.value(candidate[0], candidate[1]) > 0.40:
            return float(candidate[0]), float(candidate[1]), float(candidate[2])
    return float(pose[0]), float(pose[1]), float(pose[2])


def build_scene(task: dict, *, seed: int = 0, write: bool = True) -> Scene:
    world = load_world(task.get("world", "apartment"))
    robot_spec = load_robot(task.get("robot", "helper_rover"))
    for sensor in task.get("sensors_off", []):          # ablation: run the same robot without a sensor
        robot_spec["sensors"].pop(sensor, None)
    if "initial_posture" in task:                       # e.g. a legged robot that boots folded
        robot_spec["base"]["initial_posture"] = task["initial_posture"]
    catalog = load_catalog()
    from .kinematics import Kinematics

    lidar_height = mount_height(robot_spec, robot_spec["sensors"].get("lidar", {}).get("z", 0.175))
    robot_height = mount_height(robot_spec, Kinematics(robot_spec).height()) + 0.05
    prior = get_prior_map(world["name"], lidar_height=lidar_height, robot_height=robot_height)
    rng = np.random.default_rng(seed)
    pose = robot_start(task, prior, rng)
    spawns = place_objects(task, prior, catalog, rng)

    root = ET.parse(world["world_path"]).getroot()
    root.set("model", f"{world['name']}+{robot_spec['name']}")
    robot = robot_mjcf(robot_spec, floating=True, pose=pose)
    for section in ("worldbody", "actuator", "sensor", "equality", "contact"):
        source = robot.find(section)
        if source is None:
            continue
        target = root.find(section)
        if target is None:
            target = ET.SubElement(root, section)
        target.extend(list(source))
    worldbody = root.find("worldbody")
    for spawn in spawns:
        worldbody.append(object_body(spawn, catalog))
    # Unmapped static clutter: physically present, absent from the robot's prior map.
    for i, obs in enumerate(task.get("obstacles", [])):
        el(worldbody, "geom", name=obs.get("name", f"obstacle_{i}"), type=obs.get("type", "box"),
           pos=obs["pos"], size=obs["size"], rgba=obs.get("rgba", [0.55, 0.47, 0.36, 1]))
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    if write:
        out = BUILD_DIR / "scenes"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{task.get('name', 'scene')}_seed{seed}.xml").write_text(xml, encoding="utf-8")
    return Scene(task, seed, robot_spec, world, prior, catalog, spawns, pose, xml)
