"""Fighting arena: a boxing ring plus several robots in one MuJoCo model.

Each robot is generated from its own YAML spec exactly as in the household scenes, then
namespaced with its corner ('red_', 'blue_') and given its own collision group, so the robots hit
each other but no robot collides with itself. Each corner gets its own HAL (SimRobot with the same
prefix), so the robot software never sees the namespacing.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .config import BUILD_DIR, WORLDS_DIR, load_robot, load_yaml
from .description import el, prefix_names, robot_mjcf, set_collision_group
from .humanoid import initial, is_humanoid, stance_joints


@dataclass
class Corner:
    name: str                     # 'red' / 'blue'
    prefix: str
    spec: dict
    pose: tuple[float, float, float]
    style: str


@dataclass
class Arena:
    config: dict
    match: dict
    corners: list[Corner]
    xml: str

    def compile(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string(self.xml)
        data = mujoco.MjData(model)
        for corner in self.corners:
            if is_humanoid(corner.spec):
                for joint, angle in stance_joints(corner.spec, *initial(corner.spec)).items():
                    data.qpos[model.joint(corner.prefix + joint).qposadr[0]] = angle
        mujoco.mj_forward(model, data)
        return model, data


def load_arena(name: str) -> dict:
    cfg = load_yaml(WORLDS_DIR / name / "arena.yaml")
    cfg.setdefault("name", name)
    return cfg


def _ring(root: ET.Element, cfg: dict) -> None:
    ring = cfg["ring"]
    half = ring["size"] / 2
    el(root, "compiler", angle="radian", autolimits=True)
    el(root, "option", timestep=0.002, gravity=[0, 0, -9.81], integrator="implicitfast", cone="elliptic",
       impratio=10)
    el(root, "statistic", center=[0, 0, 0.8], extent=3.0)
    visual = el(root, "visual")
    el(visual, "global", offwidth=1280, offheight=960, azimuth=cfg.get("camera", {}).get("azimuth", 90),
       elevation=cfg.get("camera", {}).get("elevation", -18))
    el(visual, "quality", shadowsize=4096)
    el(visual, "headlight", ambient=[0.35, 0.35, 0.35], diffuse=[0.5, 0.5, 0.5], specular=[0.1, 0.1, 0.1])
    el(visual, "map", znear=0.01, zfar=40)
    asset = el(root, "asset")
    el(asset, "texture", name="canvas_tex", type="2d", builtin="flat", rgb1=ring["canvas_rgba"][:3],
       rgb2=[0.8, 0.8, 0.78], width=64, height=64, mark="edge", markrgb=[0.7, 0.7, 0.7])
    el(asset, "material", name="canvas_mat", texture="canvas_tex", texrepeat=[8, 8], reflectance=0.02)
    el(asset, "material", name="apron_mat", rgba=[0.12, 0.12, 0.14, 1])
    world = el(root, "worldbody")
    el(world, "light", name="key", pos=[0, 0, 4.5], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True)
    el(world, "light", name="fill", pos=[3, -3, 3.5], dir=[-0.6, 0.6, -0.6], diffuse=[0.25, 0.25, 0.25])
    # The canvas; the floor geom keeps the name every referee and map builder looks for.
    el(world, "geom", name="floor", type="plane", size=[half + 0.6, half + 0.6, 0.1], material="canvas_mat",
       friction=[ring.get("friction", 1.0), 0.005, 0.0001])
    el(world, "geom", name="apron", type="plane", size=[half + 3, half + 3, 0.1], pos=[0, 0, -0.002],
       material="apron_mat", contype=0, conaffinity=0)
    posts, ropes = ring["posts"], ring["ropes"]
    corners = [(sx * half, sy * half) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    for i, (x, y) in enumerate(corners):
        el(world, "geom", name=f"post_{i}", type="cylinder", size=[posts["radius"], posts["height"] / 2],
           pos=[x, y, posts["height"] / 2], rgba=posts["rgba"])
    for i, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])):
        for j, z in enumerate(ropes["heights"]):
            el(world, "geom", name=f"rope_{i}_{j}", type="capsule", size=[ropes["radius"]],
               fromto=[a[0], a[1], z, b[0], b[1], z], rgba=ropes["rgba"])
    cam = cfg.get("camera", {})
    dist, elev = cam.get("distance", 4.2), math.radians(cam.get("elevation", -18))
    el(world, "camera", name="ringside", pos=[0, -dist * math.cos(elev), 0.9 - dist * math.sin(elev)],
       xyaxes=[1, 0, 0, 0, -math.sin(elev), math.cos(elev)], fovy=50)


def build_arena(match: dict, *, write: bool = True) -> Arena:
    cfg = load_arena(match.get("world", "boxing_ring"))
    root = ET.Element("mujoco", model=f"{cfg['name']}+{'+'.join(match['fighters'])}")
    _ring(root, cfg)
    sections = {name: root.find(name) for name in ("worldbody",)}
    for name in ("actuator", "sensor", "equality", "contact"):
        sections[name] = ET.SubElement(root, name)
    corners: list[Corner] = []
    fighters = match["fighters"]
    for index, (corner_name, fighter) in enumerate(fighters.items()):
        spec = copy.deepcopy(load_robot(fighter["robot"]))
        corner_cfg = cfg["corners"][corner_name]
        spec["base"]["color"] = corner_cfg["rgba"]
        x, y, yaw_deg = match.get("start", {}).get(corner_name, corner_cfg["pose"])
        pose = (float(x), float(y), math.radians(yaw_deg))
        robot = robot_mjcf(spec, floating=True, pose=pose)
        prefix = f"{corner_name}_"
        prefix_names(robot, prefix)
        set_collision_group(robot, index, len(fighters))
        for name, target in sections.items():
            source = robot.find(name)
            if source is not None:
                target.extend(list(source))
        corners.append(Corner(corner_name, prefix, spec, pose, fighter.get("style", "pressure")))
    for name in ("actuator", "sensor", "equality", "contact"):
        if not len(sections[name]):
            root.remove(sections[name])
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    if write:
        out = BUILD_DIR / "scenes"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{match.get('name', 'match')}.xml").write_text(xml, encoding="utf-8")
    return Arena(cfg, match, corners, xml)


def body_pose(model, data, prefix: str) -> np.ndarray:
    """Ground truth (referee/viewer only): x, y, yaw of a robot's base."""
    b = data.body(prefix + "base")
    rot = b.xmat.reshape(3, 3)
    return np.array([b.xpos[0], b.xpos[1], math.atan2(rot[1, 0], rot[0, 0])])
