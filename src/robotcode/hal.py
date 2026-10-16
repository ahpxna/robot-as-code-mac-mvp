"""Hardware abstraction layer (HAL).

`RobotInterface` is the only door between the onboard software and a robot:
motors take setpoints, sensors return noisy, timestamped measurements. The
onboard stack never reads simulator state.

`SimRobot` implements the interface on MuJoCo. A real-robot driver (ROS 2
`ros2_control` hardware interface, serial motor board, ...) implements the same
methods; see docs/ARCHITECTURE.md for the mapping to ROS 2 topics.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .description import WHEELS, arm_joint_names, has_gripper


@dataclass
class BatteryState:
    soc: float                  # state of charge 0..1
    voltage: float
    power_w: float
    energy_used_wh: float


@dataclass
class RobotState:
    stamp: float
    wheel_position: np.ndarray  # rad, quantised encoder counts
    wheel_velocity: np.ndarray  # rad/s
    gyro: np.ndarray            # rad/s, body frame
    accel: np.ndarray           # m/s^2, body frame
    joints: dict[str, float]
    efforts: dict[str, float]
    gripper_position: float     # finger travel, 0 = open
    finger_forces: tuple[float, float]
    bumper_force: float
    battery: BatteryState
    estopped: bool


@dataclass
class LaserScan:
    stamp: float
    angles: np.ndarray          # rad, in the sensor frame
    ranges: np.ndarray          # m, inf = no return
    range_min: float
    range_max: float
    mount: tuple[float, float, float]   # sensor x, y, z in the base frame


@dataclass
class RGBDFrame:
    stamp: float
    rgb: np.ndarray             # (H, W, 3) uint8, row 0 = top
    depth: np.ndarray           # (H, W) float32 metres along the optical axis, 0 = invalid
    fx: float
    fy: float
    cx: float
    cy: float
    camera: str = "head_camera"
    meta: dict = field(default_factory=dict)


class RobotInterface(ABC):
    spec: dict

    @abstractmethod
    def time(self) -> float: ...

    @abstractmethod
    def read_state(self) -> RobotState: ...

    @abstractmethod
    def read_scan(self) -> LaserScan | None: ...

    @abstractmethod
    def capture_rgbd(self) -> RGBDFrame | None: ...

    @abstractmethod
    def command_wheels(self, left: float, right: float) -> None:
        """Wheel angular velocity setpoints (rad/s), executed by the motor controllers."""

    @abstractmethod
    def command_joints(self, targets: dict[str, float]) -> None:
        """Arm joint position setpoints, executed by the servo drives."""

    @abstractmethod
    def command_gripper(self, travel: float) -> None:
        """Finger travel setpoint (0 = open); the drive is force limited."""

    @abstractmethod
    def emergency_stop(self, engaged: bool = True) -> None: ...


class SimRobot(RobotInterface):
    """MuJoCo-backed robot: actuators, sensor noise, battery, LiDAR and RGB-D camera."""

    WATCHDOG_S = 0.3

    def __init__(self, model, data, spec: dict, *, seed: int = 0):
        import mujoco

        self._mj = mujoco
        self.model, self.data, self.spec = model, data, spec
        self.rng = np.random.default_rng(seed + 7919)
        s = spec["sensors"]
        self.arm_joints = arm_joint_names(spec)
        self._act = {model.actuator(i).name: i for i in range(model.nu)}
        self._sensor = {model.sensor(i).name: (model.sensor(i).adr[0], model.sensor(i).dim[0])
                        for i in range(model.nsensor)}
        self._base_id = model.body("base").id
        self._ctrl_range = {name: tuple(model.actuator_ctrlrange[i]) for name, i in self._act.items()}
        imu = s.get("imu", {})
        self._gyro_bias = self.rng.normal(0.0, imu.get("gyro_bias_std", 0.0), 3)
        self._last_wheel_cmd = -1.0
        self._estop = False
        self._brownout = False
        battery = spec["base"].get("battery") or {}
        self._battery_spec = battery
        self._energy_j = 0.0
        self._power = 0.0
        self._scan: LaserScan | None = None
        self._renderer = None
        self._scene_option = None
        lidar = s.get("lidar")
        if lidar:
            fov = math.radians(lidar["fov_deg"])
            self._lidar_angles = np.linspace(-fov / 2, fov / 2, int(lidar["beams"]))
        for name in self.arm_joints:
            self._set_ctrl(f"{name}_motor", float(data.joint(name).qpos[0]))

    # ---- simulation-side hooks (called by the runtime, not by the robot software) --------
    def after_physics_step(self, dt: float) -> None:
        d = self.data
        if self.time() - self._last_wheel_cmd > self.WATCHDOG_S:
            self._set_ctrl("left_wheel_motor", 0.0)
            self._set_ctrl("right_wheel_motor", 0.0)
        if self._battery_spec:
            mech = np.maximum(d.actuator_force * d.actuator_velocity, 0.0).sum()
            self._power = self._battery_spec.get("idle_power_w", 20.0) + mech / self._battery_spec.get(
                "drive_efficiency", 0.75)
            self._energy_j += self._power * dt
            if self._soc() <= 0.0 and not self._brownout:
                self._brownout = True
                self.emergency_stop(True)

    # ---- RobotInterface ----------------------------------------------------
    def time(self) -> float:
        return float(self.data.time)

    def _read(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr:adr + dim].copy()

    def _soc(self) -> float:
        cap = self._battery_spec.get("capacity_wh", 0.0) * 3600.0
        return 1.0 if cap <= 0 else max(0.0, 1.0 - self._energy_j / cap)

    def read_state(self) -> RobotState:
        s = self.spec["sensors"]
        enc = s.get("wheel_encoders", {})
        tick = 2 * math.pi / enc.get("counts_per_rev", 4096)
        wheel_pos = np.array([self._read(f"{w}_encoder")[0] for w in WHEELS])
        wheel_pos = np.round(wheel_pos / tick) * tick
        wheel_vel = np.array([self._read(f"{w}_speed")[0] for w in WHEELS])
        imu = s.get("imu", {})
        gyro = self._read("imu_gyro") + self._gyro_bias + self.rng.normal(0, imu.get("gyro_noise_std", 0.0), 3)
        accel = self._read("imu_accel") + self.rng.normal(0, imu.get("accel_noise_std", 0.0), 3)
        joints = {j: float(self._read(f"{j}_encoder")[0]) for j in self.arm_joints}
        efforts = {j: float(self._read(f"{j}_effort")[0]) for j in self.arm_joints}
        gripper = float(self._read("gripper_encoder")[0]) if has_gripper(self.spec) else 0.0
        forces = ((float(self._read("finger_left_force")[0]), float(self._read("finger_right_force")[0]))
                  if has_gripper(self.spec) else (0.0, 0.0))
        bumper = float(self._read("bumper_force")[0]) if "bumper_force" in self._sensor else 0.0
        soc = self._soc()
        nominal = self._battery_spec.get("nominal_voltage", 24.0)
        battery = BatteryState(soc, nominal * (0.82 + 0.18 * soc) if soc > 0 else 0.0, self._power,
                               self._energy_j / 3600.0)
        return RobotState(self.time(), wheel_pos, wheel_vel, gyro, accel, joints, efforts, gripper, forces,
                          bumper, battery, self._estop)

    def read_scan(self) -> LaserScan | None:
        lidar = self.spec["sensors"].get("lidar")
        if not lidar:
            return None
        if self._scan is not None and self.time() - self._scan.stamp < 1.0 / lidar["rate_hz"] - 1e-9:
            return self._scan
        site = self.data.site("lidar")
        rot = site.xmat.reshape(3, 3)
        local = np.stack([np.cos(self._lidar_angles), np.sin(self._lidar_angles),
                          np.zeros_like(self._lidar_angles)], axis=1)
        vec = (local @ rot.T).ravel()
        n = len(self._lidar_angles)
        dist = np.zeros(n)
        geomid = np.zeros(n, dtype=np.int32)
        self._mj.mj_multiRay(self.model, self.data, site.xpos.copy(), vec, None, 1, self._base_id,
                             geomid, dist, None, n, float(lidar["range_max"]))
        ranges = np.where(dist < 0, np.inf, dist)
        ranges = ranges + self.rng.normal(0, lidar.get("noise_std", 0.0), n)
        ranges[(ranges < lidar["range_min"]) | (ranges > lidar["range_max"])] = np.inf
        self._scan = LaserScan(self.time(), self._lidar_angles.copy(), ranges, lidar["range_min"],
                               lidar["range_max"], (lidar["x"], 0.0, lidar["z"]))
        return self._scan

    def capture_rgbd(self) -> RGBDFrame | None:
        cam = self.spec["sensors"].get("head_camera")
        if not cam:
            return None
        mj = self._mj
        width, height = int(cam["width"]), int(cam["height"])
        if self._renderer is None:
            self._renderer = mj.Renderer(self.model, height, width)
            opt = mj.MjvOption()
            opt.sitegroup[:] = 0                 # sites are annotations, not physical
            opt.geomgroup[3:] = 0
            self._scene_option = opt
        r = self._renderer
        r.update_scene(self.data, camera="head_camera", scene_option=self._scene_option)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera="head_camera", scene_option=self._scene_option)
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        noise = cam.get("depth_noise_std", 0.0)
        if noise:
            depth = depth + self.rng.normal(0, 1, depth.shape).astype(np.float32) * noise * depth ** 2
        depth[(depth < 0.05) | (depth > cam.get("max_depth", 6.0))] = 0.0
        if cam.get("rgb_noise_std", 2.0):
            rgb = np.clip(rgb + self.rng.normal(0, cam.get("rgb_noise_std", 2.0), rgb.shape), 0, 255).astype(
                np.uint8)
        f = (height / 2) / math.tan(math.radians(cam["fovy_deg"]) / 2)
        return RGBDFrame(self.time(), rgb, depth, f, f, width / 2, height / 2)

    def _set_ctrl(self, actuator: str, value: float) -> None:
        if actuator not in self._act:
            return
        lo, hi = self._ctrl_range[actuator]
        self.data.ctrl[self._act[actuator]] = min(max(value, lo), hi)

    def command_wheels(self, left: float, right: float) -> None:
        if self._estop:
            left = right = 0.0
        self._set_ctrl("left_wheel_motor", left)
        self._set_ctrl("right_wheel_motor", right)
        self._last_wheel_cmd = self.time()

    def command_joints(self, targets: dict[str, float]) -> None:
        if self._estop:
            return
        for name, value in targets.items():
            if name not in self.arm_joints:
                raise KeyError(f"unknown arm joint {name!r}")
            self._set_ctrl(f"{name}_motor", value)

    def command_gripper(self, travel: float) -> None:
        if not self._estop:
            self._set_ctrl("gripper_motor", travel)

    def emergency_stop(self, engaged: bool = True) -> None:
        if self._brownout:
            engaged = True
        self._estop = engaged
        if engaged:
            self._set_ctrl("left_wheel_motor", 0.0)
            self._set_ctrl("right_wheel_motor", 0.0)
            for name in self.arm_joints:          # servo brakes hold the arm where it is
                self._set_ctrl(f"{name}_motor", float(self.data.joint(name).qpos[0]))

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
