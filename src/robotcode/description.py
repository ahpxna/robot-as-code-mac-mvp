"""Robot description as code: robots/<name>.yaml -> MJCF.

One generator produces both
* the simulated robot (floating base, inserted into a world by robotcode.scene), and
* the kinematic model used by the onboard software (fixed base) -- the role a
  URDF plays for robot_state_publisher / MoveIt on a ROS robot.

Naming contract shared with the HAL (robotcode.hal):
  joints     left_wheel, right_wheel, <arm joints>, finger_left/finger_right
  actuators  <joint>_motor, gripper_motor
  sensors    <joint>_encoder, <wheel>_speed, <arm joint>_effort, imu_gyro,
             imu_accel, finger_<side>_force, bumper_force
  frames     site imu, site lidar, site grasp_site, camera head_camera
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from .legged import SEGMENTS, is_legged, leg_joint_names, leg_mounts

# Robot geoms collide with the world but never with each other.
ROBOT_COLLISION = {"contype": 2, "conaffinity": 1}
WHEELS = ("left_wheel", "right_wheel")

_MAST_RGBA = [0.30, 0.31, 0.33, 1]
_ARM_RGBA = [0.72, 0.73, 0.75, 1]
_DARK_RGBA = [0.13, 0.14, 0.16, 1]


def fmt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return " ".join(fmt(v) for v in value)
    return f"{float(value):.6g}"


def el(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k.rstrip("_"): fmt(v) for k, v in attrs.items() if v is not None})


def arm_joint_names(spec: dict) -> list[str]:
    return list((spec.get("arm") or {}).get("joints", {}))


def actuated_joint_names(spec: dict) -> list[str]:
    """Position-servoed joints the HAL exposes (arm and legs; wheels and gripper have their own calls)."""
    return arm_joint_names(spec) + leg_joint_names(spec)


def spawn_height(spec: dict) -> float:
    if not is_legged(spec):
        return 0.0
    base = spec["base"]
    return base["postures"][base.get("initial_posture", "stand")]["body_height"] + 0.002


def mount_height(spec: dict, z: float) -> float:
    """Height above the floor of a sensor mounted at base-frame height z (standing, for legged robots)."""
    return z + (spec["base"]["postures"]["stand"]["body_height"] if is_legged(spec) else 0.0)


def has_gripper(spec: dict) -> bool:
    return bool(spec.get("gripper"))


def robot_mjcf(spec: dict, *, floating: bool = True, pose=(0.0, 0.0, 0.0)) -> ET.Element:
    """Return a standalone <mujoco> element describing the robot."""
    root = ET.Element("mujoco", model=spec["name"])
    el(root, "compiler", angle="radian", autolimits=True)
    world = el(root, "worldbody")
    actuators = el(root, "actuator")
    sensors = el(root, "sensor")
    equality = el(root, "equality")
    x, y, yaw = pose
    legged = is_legged(spec)
    # Wheeled robots: base origin on the floor. Legged robots: base origin at the body centre,
    # spawned at the height of their initial posture.
    z0 = spawn_height(spec) if (legged and floating) else 0.0
    base = el(world, "body", name="base", pos=[x, y, z0], euler=[0, 0, yaw])
    if floating:
        el(base, "freejoint", name="base_free")
    if legged:
        _build_legged_base(spec, base, actuators, sensors)
    else:
        _build_base(spec, base, actuators, sensors)
    arm = spec.get("arm")
    if arm:
        if arm.get("kind") != "cartesian":
            raise ValueError(f"unsupported arm kind {arm.get('kind')!r} (supported: cartesian)")
        wrist = _build_cartesian_arm(spec, base, actuators, sensors)
        if has_gripper(spec):
            _build_parallel_gripper(spec, wrist, actuators, sensors, equality)
    _build_sensors(spec, base, sensors)
    for section in (actuators, sensors, equality):
        if not len(section):
            root.remove(section)
    return root


def robot_xml(spec: dict, **kwargs) -> str:
    return ET.tostring(robot_mjcf(spec, **kwargs), encoding="unicode")


def chassis_top(spec: dict) -> float:
    ch = spec["base"]["chassis"]
    return ch["ground_clearance"] + ch["size"][2]


def _build_base(spec: dict, base: ET.Element, actuators: ET.Element, sensors: ET.Element) -> None:
    b = spec["base"]
    if b.get("kind", "differential_drive") != "differential_drive":
        raise ValueError(f"unsupported base kind {b.get('kind')!r}")
    ch, wh = b["chassis"], b["wheels"]
    length, width, height = ch["size"]
    clearance = ch["ground_clearance"]
    zc = clearance + height / 2
    el(base, "geom", name="chassis", type="box", size=[length / 2, width / 2, height / 2],
       pos=[0, 0, zc], mass=ch["mass"], rgba=ch.get("color", [0.2, 0.4, 0.8, 1]), **ROBOT_COLLISION)
    el(base, "geom", name="chassis_lid", type="box", size=[length / 2 - 0.03, width / 2 - 0.03, 0.004],
       pos=[0, 0, clearance + height + 0.002], rgba=[0.85, 0.86, 0.88, 1], mass=0.05, contype=0, conaffinity=0)
    battery = b.get("battery")
    if battery:
        # Heavy pack low in the chassis: visual-only geom that sets mass and centre of gravity.
        el(base, "geom", name="battery_pack", type="box", size=[0.14, 0.10, 0.03],
           pos=[0, 0, clearance + 0.035], mass=battery["mass"], contype=0, conaffinity=0, group=3,
           rgba=[0.1, 0.1, 0.1, 1])
    for i, caster in enumerate(b.get("casters", [])):
        r = caster["radius"]
        # Ball casters roll freely: frictionless contacts (condim=1). priority=1 makes the caster,
        # not the floor, decide the contact parameters (otherwise MuJoCo takes the max condim).
        el(base, "geom", name=f"caster_{i}", type="sphere", size=[r], pos=[caster["x"], caster.get("y", 0.0), r],
           mass=0.1, condim=1, priority=1, rgba=[0.2, 0.2, 0.22, 1], **ROBOT_COLLISION)
    r, w, track = wh["radius"], wh["width"], wh["track"]
    motor = wh["motor"]
    for name, sign in zip(WHEELS, (1, -1)):
        wheel = el(base, "body", name=name, pos=[wh.get("x", 0.0), sign * track / 2, r])
        el(wheel, "joint", name=name, type="hinge", axis=[0, 1, 0], damping=0.02, armature=0.005)
        # A cylinder touches the floor at both rims, and the two contacts scrub against each other when
        # turning. Collide with a thin ellipsoid (one contact patch, like an inflated tyre); draw a cylinder.
        el(wheel, "geom", name=f"{name}_tire", type="ellipsoid", size=[r, r, w / 2], euler=[math.pi / 2, 0, 0],
           mass=wh["mass"], friction=[wh.get("friction", 1.0), 0.005, 0.0001], rgba=[0.1, 0.1, 0.11, 0],
           **ROBOT_COLLISION)
        el(wheel, "geom", name=f"{name}_rim", type="cylinder", size=[r, w / 2], euler=[math.pi / 2, 0, 0],
           mass=0.0, contype=0, conaffinity=0, rgba=[0.1, 0.1, 0.11, 1])
        el(wheel, "geom", name=f"{name}_hub", type="box", size=[r * 0.7, w / 2 + 0.002, 0.012],
           mass=0.01, contype=0, conaffinity=0, rgba=[0.75, 0.75, 0.78, 1])
        el(actuators, "velocity", name=f"{name}_motor", joint=name, kv=motor["kv"],
           ctrlrange=[-motor["max_speed"], motor["max_speed"]],
           forcerange=[-motor["max_torque"], motor["max_torque"]])
        el(sensors, "jointpos", name=f"{name}_encoder", joint=name)
        el(sensors, "jointvel", name=f"{name}_speed", joint=name)


def _build_legged_base(spec: dict, base: ET.Element, actuators: ET.Element, sensors: ET.Element) -> None:
    b = spec["base"]
    body = b["body"]
    color, accent = body.get("color", [0.1, 0.1, 0.1, 1]), body.get("accent", [0.8, 0.1, 0.1, 1])
    for part in ("thorax", "abdomen"):
        e = body[part]
        el(base, "geom", name=part, type="ellipsoid", size=e["radii"], pos=e["center"], mass=e["mass"],
           rgba=color, **ROBOT_COLLISION)
    head = body.get("head")
    if head:
        el(base, "geom", name="head", type="sphere", size=[head["radius"]], pos=head["center"], mass=head["mass"],
           rgba=color, **ROBOT_COLLISION)
    ab = body["abdomen"]
    # Red hourglass stripe on the back: visual only.
    el(base, "geom", name="accent_stripe", type="ellipsoid", size=[ab["radii"][0] * 0.55, ab["radii"][1] * 0.3, 0.004],
       pos=[ab["center"][0], 0, ab["center"][2] + ab["radii"][2] * 0.97], rgba=accent, mass=0.001,
       contype=0, conaffinity=0)
    battery = b.get("battery")
    if battery:
        el(base, "geom", name="battery_pack", type="box", size=[0.03, 0.02, 0.008], pos=ab["center"],
           mass=battery["mass"], contype=0, conaffinity=0, group=3, rgba=[0.1, 0.1, 0.1, 1])
    lg = b["legs"]
    radius, foot = lg["radius"], lg["foot"]
    for name, (x, y, heading) in leg_mounts(spec).items():
        parent, pos = base, [x, y, 0.0]
        for i, seg in enumerate(SEGMENTS):
            j = lg["joints"][seg]
            link = el(parent, "body", name=f"{name}_{seg}_link", pos=pos,
                      euler=[0, 0, math.radians(heading)] if i == 0 else None)
            # coxa yaws about z; femur and tibia pitch about -y so a positive angle lifts the leg.
            el(link, "joint", name=f"{name}_{seg}", type="hinge", axis=[0, 0, 1] if i == 0 else [0, -1, 0],
               range=[math.radians(v) for v in j["range"]], damping=j["damping"], armature=lg.get("armature", 0.004))
            length = lg[seg]["length"]
            el(link, "geom", name=f"{name}_{seg}_geom", type="capsule", fromto=[0, 0, 0, length, 0, 0], size=[radius],
               mass=lg[seg]["mass"], rgba=color if seg != "femur" else [0.16, 0.16, 0.18, 1], **ROBOT_COLLISION)
            el(actuators, "position", name=f"{name}_{seg}_motor", joint=f"{name}_{seg}", kp=j["kp"],
               ctrlrange=[math.radians(v) for v in j["range"]], forcerange=[-j["force"], j["force"]])
            el(sensors, "jointpos", name=f"{name}_{seg}_encoder", joint=f"{name}_{seg}")
            el(sensors, "actuatorfrc", name=f"{name}_{seg}_effort", actuator=f"{name}_{seg}_motor")
            parent, pos = link, [length, 0, 0]
        el(parent, "geom", name=f"{name}_foot", type="sphere", size=[foot["radius"]], pos=pos, mass=0.005,
           friction=[foot.get("friction", 1.0), 0.005, 0.0001], rgba=accent, **ROBOT_COLLISION)
        if "foot_contacts" in spec["sensors"]:
            el(parent, "site", name=f"{name}_foot_touch", type="sphere", size=[foot["radius"] + 0.004], pos=pos,
               rgba=[0, 0, 0, 0])
            el(sensors, "touch", name=f"foot_{name}", site=f"{name}_foot_touch")


def _build_cartesian_arm(spec: dict, base: ET.Element, actuators: ET.Element, sensors: ET.Element) -> ET.Element:
    arm = spec["arm"]
    joints = arm["joints"]
    if set(joints) != {"lift", "reach", "side"}:
        raise ValueError("cartesian arm needs joints lift (z), reach (x) and side (y)")
    mast = arm["mast"]
    top = chassis_top(spec)
    mx, (mast_dx, mast_dy) = mast["x"], mast["size"]
    camera = spec["sensors"].get("head_camera")
    mast_top = mast["height"]
    el(base, "geom", name="mast", type="box", size=[mast_dx / 2, mast_dy / 2, (mast_top - top) / 2],
       pos=[mx, 0, (mast_top + top) / 2], mass=1.5, rgba=_MAST_RGBA, **ROBOT_COLLISION)
    if camera:
        head_z = camera["z"]
        el(base, "geom", name="head", type="box", size=[0.045, 0.08, 0.05],
           pos=[camera["x"] - 0.046, 0, head_z], mass=0.6, rgba=_DARK_RGBA, **ROBOT_COLLISION)
        if head_z - 0.05 > mast_top:
            el(base, "geom", name="neck", type="box", size=[0.02, 0.025, (head_z - 0.05 - mast_top) / 2],
               pos=[mx, 0, (head_z - 0.05 + mast_top) / 2], mass=0.1, rgba=_MAST_RGBA, **ROBOT_COLLISION)

    lift = joints["lift"]
    lo, hi = lift["range"]
    lift_link = el(base, "body", name="lift_link", pos=[mx + mast_dx / 2 + 0.02, 0, lo])
    # ref=lo makes the joint value equal to the beam height above the floor.
    el(lift_link, "joint", name="lift", type="slide", axis=[0, 0, 1], range=[lo, hi], ref=lo,
       damping=lift["damping"], armature=0.05)
    el(lift_link, "geom", name="lift_carriage", type="box", size=[0.02, 0.065, 0.04],
       mass=lift["mass"], rgba=_ARM_RGBA, **ROBOT_COLLISION)

    reach = joints["reach"]
    reach_max = reach["range"][1]
    arm_link = el(lift_link, "body", name="arm_link", pos=[0.0, 0, 0])
    el(arm_link, "joint", name="reach", type="slide", axis=[1, 0, 0], range=reach["range"],
       damping=reach["damping"], armature=0.02)
    beam_half = (reach_max + 0.02) / 2
    # Telescoping beam: slides through the carriage, its tail stays inside the footprint.
    el(arm_link, "geom", name="arm_beam", type="box", size=[beam_half, 0.022, 0.018],
       pos=[0.02 - beam_half, 0, 0], mass=reach["mass"], rgba=_ARM_RGBA, **ROBOT_COLLISION)
    el(arm_link, "geom", name="wrist_rail", type="box", size=[0.012, 0.13, 0.01], pos=[0.02, 0, -0.025],
       mass=0.05, rgba=_MAST_RGBA, contype=0, conaffinity=0)

    side = joints["side"]
    drop = arm["wrist_drop"]
    wrist = el(arm_link, "body", name="wrist_link", pos=[0.02, 0, 0])
    el(wrist, "joint", name="side", type="slide", axis=[0, 1, 0], range=side["range"],
       damping=side["damping"], armature=0.01)
    el(wrist, "geom", name="wrist_post", type="box", size=[0.015, 0.02, drop / 2], pos=[0, 0, -drop / 2],
       mass=side["mass"], rgba=_ARM_RGBA, **ROBOT_COLLISION)

    for name, joint in joints.items():
        el(actuators, "position", name=f"{name}_motor", joint=name, kp=joint["kp"], ctrlrange=joint["range"],
           forcerange=[-joint["force"], joint["force"]])
        el(sensors, "jointpos", name=f"{name}_encoder", joint=name)
        el(sensors, "actuatorfrc", name=f"{name}_effort", actuator=f"{name}_motor")
    return wrist


def _build_parallel_gripper(spec: dict, wrist: ET.Element, actuators: ET.Element, sensors: ET.Element,
                            equality: ET.Element) -> None:
    g = spec["gripper"]
    if g.get("kind") != "parallel_jaw":
        raise ValueError(f"unsupported gripper kind {g.get('kind')!r}")
    finger = g["finger"]
    fl, ft, fh = finger["length"], finger["thickness"], finger["height"]
    half_open = g["max_opening"] / 2
    travel = half_open - 0.002
    gripper = el(wrist, "body", name="gripper_link", pos=[0, 0, -spec["arm"]["wrist_drop"]])
    el(gripper, "geom", name="palm", type="box", size=[0.02, half_open + ft + 0.008, 0.025],
       mass=0.25, rgba=_DARK_RGBA, **ROBOT_COLLISION)
    el(gripper, "site", name="grasp_site", pos=[0.02 + fl / 2, 0, 0], size=[0.008], rgba=[1, 0.85, 0, 0.5])
    wrist_cam = spec["sensors"].get("wrist_camera")
    if wrist_cam:
        # Eye-in-hand camera on top of the palm, looking forward and down between the fingers.
        pitch = math.radians(wrist_cam.get("pitch_deg", 25))
        el(gripper, "geom", name="wrist_camera_housing", type="box", size=[0.012, 0.018, 0.01], pos=[0.0, 0, 0.034],
           mass=0.03, rgba=_MAST_RGBA, contype=0, conaffinity=0)
        el(gripper, "camera", name="wrist_camera", pos=[0.013, 0, 0.036],
           xyaxes=[0, -1, 0, math.sin(pitch), 0, math.cos(pitch)], fovy=wrist_cam["fovy_deg"])
    for side, sign in (("left", 1), ("right", -1)):
        body = el(gripper, "body", name=f"finger_{side}", pos=[0.02, sign * (half_open + ft / 2), 0])
        el(body, "joint", name=f"finger_{side}", type="slide", axis=[0, -sign, 0], range=[0, travel],
           damping=g.get("damping", 4.0), armature=0.002)
        el(body, "geom", name=f"finger_{side}_pad", type="box", size=[fl / 2, ft / 2, fh / 2], pos=[fl / 2, 0, 0],
           mass=finger["mass"], friction=[finger["friction"], 0.02, 0.001], condim=4,
           rgba=[0.2, 0.2, 0.22, 1], **ROBOT_COLLISION)
        # Force-sensing resistor on the inner face of the finger.
        el(body, "site", name=f"finger_{side}_touch", type="box", size=[fl / 2 + 0.003, 0.004, fh / 2 + 0.003],
           pos=[fl / 2, -sign * ft / 2, 0], rgba=[0, 0, 0, 0])
        el(sensors, "touch", name=f"finger_{side}_force", site=f"finger_{side}_touch")
    # Mechanical linkage: both fingers move together (one motor).
    el(equality, "joint", name="finger_coupling", joint1="finger_right", joint2="finger_left",
       polycoef=[0, 1, 0, 0, 0], solref=[0.005, 1])
    el(actuators, "position", name="gripper_motor", joint="finger_left", kp=g["kp"], ctrlrange=[0, travel],
       forcerange=[-g["force"], g["force"]])
    el(sensors, "jointpos", name="gripper_encoder", joint="finger_left")


def _build_sensors(spec: dict, base: ET.Element, sensors: ET.Element) -> None:
    s = spec["sensors"]
    legged = is_legged(spec)
    ch = spec["base"].get("chassis", {"size": [0.0, 0.0, 0.0], "ground_clearance": 0.0})
    length, width, height = ch["size"]
    zc = 0.0 if legged else ch["ground_clearance"] + height / 2
    if "imu" in s:
        el(base, "site", name="imu", pos=[0, 0, zc], size=[0.01], rgba=[1, 0, 0, 0])
        el(sensors, "gyro", name="imu_gyro", site="imu")
        el(sensors, "accelerometer", name="imu_accel", site="imu")
    if "lidar" in s:
        lidar = s["lidar"]
        el(base, "geom", name="lidar_puck", type="cylinder", size=[0.03, 0.012],
           pos=[lidar["x"], 0, lidar["z"] - 0.004], mass=0.17, rgba=_DARK_RGBA, contype=0, conaffinity=0)
        el(base, "site", name="lidar", pos=[lidar["x"], 0, lidar["z"]], size=[0.005], rgba=[0, 1, 0, 0])
    if "head_camera" in s:
        cam = s["head_camera"]
        pitch = math.radians(cam["pitch_deg"])
        if not spec.get("arm") and not legged:
            # No arm mast to carry the camera: add a slim sensor post.
            top = ch["ground_clearance"] + height
            post_x = cam["x"] - 0.05
            el(base, "geom", name="sensor_post", type="box", size=[0.015, 0.015, (cam["z"] - top) / 2],
               pos=[post_x, 0, (cam["z"] + top) / 2], mass=0.3, rgba=_MAST_RGBA, **ROBOT_COLLISION)
            el(base, "geom", name="head", type="box", size=[0.035, 0.06, 0.035], pos=[cam["x"] - 0.036, 0, cam["z"]],
               mass=0.3, rgba=_DARK_RGBA, **ROBOT_COLLISION)
        # MuJoCo cameras look along -z with +y up: x = image right, y = image up.
        el(base, "camera", name="head_camera", pos=[cam["x"], 0, cam["z"]],
           xyaxes=[0, -1, 0, math.sin(pitch), 0, math.cos(pitch)], fovy=cam["fovy_deg"])
    if "lidar_3d" in s:
        # Hemispherical multi-channel LiDAR on top of the head; rays are cast by the HAL.
        l3 = s["lidar_3d"]
        el(base, "geom", name="lidar3d_dome", type="cylinder", size=[0.04, 0.025], pos=[l3["x"], 0, l3["z"] - 0.01],
           mass=0.25, rgba=_DARK_RGBA, contype=0, conaffinity=0)
        el(base, "site", name="lidar_3d", pos=[l3["x"], 0, l3["z"]], size=[0.005], rgba=[0, 1, 0, 0])
    if "tof_ring" in s:
        # Short-range time-of-flight sensors around the chassis: see what is below the LiDAR plane.
        # Like the LiDARs, their rays are cast by the HAL (which masks hits on the robot itself).
        tof = s["tof_ring"]
        z = tof.get("z", ch["ground_clearance"] + 0.015)
        for i, (x, y, yaw_deg) in enumerate(tof["mounts"]):
            yaw = math.radians(yaw_deg)
            el(base, "site", name=f"tof_{i}", pos=[x, y, z], zaxis=[math.cos(yaw), math.sin(yaw), 0], size=[0.006],
               rgba=[1, 0.5, 0, 0])
    if "bumper" in s:
        el(base, "site", name="bumper", type="box", size=[length / 2 + 0.012, width / 2 + 0.012, height * 0.4],
           pos=[0, 0, zc], rgba=[1, 0, 0, 0])
        el(sensors, "touch", name="bumper_force", site="bumper")
