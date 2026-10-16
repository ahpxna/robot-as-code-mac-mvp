# Architecture

The system is split the way a real robot's software is split. The one rule that
matters most: **the onboard software never reads simulator state.** It sees the
robot only through the hardware abstraction layer (HAL), exactly as it would on
real hardware. Simulator ground truth is used by one component only, the
referee that scores episodes.

```
 "Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao."
        │
        ▼
 planner.py / llm_planner.py   language -> plan of skill calls, grounded on the map + object catalog
        │                      (rules by default; LLM optional, validated against the skill registry)
        ▼
 agent.py   MissionExecutor    runs steps, verifies, recovers (re-find, retry), reports honestly
        │
        ▼
 skills.py  navigate_to · find_object · pick_object · place_object · verify_object_at · look_around ...
        │
        ▼
 stack.py   RobotStack (50 Hz)  ── localization.py   wheel/gyro odometry + LiDAR scan matching (SDF)
        │                       ── navigation.py     costmap + LiDAR obstacle layer, Dijkstra, pure pursuit,
        │                                            collision forecast, stall detection, docking
        │                       ── perception.py     RGB-D colour segmentation -> 3D -> shape classification,
        │                                            object memory (WorldModel)
        │                       ── manipulation.py   joint trajectories, IK, force-sensed grasp checks
        │                       ── safety.py         bumper, tilt, battery
        │                       ── kinematics.py     FK/IK from the robot description (URDF role)
        ▼
 hal.py     RobotInterface      wheels (rad/s), joints (position), gripper (force-limited), e-stop
        │                       state: encoders, IMU, finger forces, bumper, battery; LiDAR scan; RGB-D frame
        ▼
 SimRobot (MuJoCo)  <── scene.py  <── worlds/<w>/world.xml + robots/<r>.yaml + tasks/<t>.yaml
        │
        ▼
 runtime.py GroundTruth (referee: collisions, object bumps, where objects really are)  -> evaluate.py
```

## Everything is code / configuration

| File | Role |
|---|---|
| `robots/*.yaml` | Robot description: chassis, battery, wheels + motor limits, casters, arm joints (range, speed, gain, force), gripper, sensors (pose, FOV, rates, noise), control rate. `description.py` generates the MJCF from it. |
| `worlds/<name>/world.xml` | Static environment (walls, furniture, the user). |
| `worlds/<name>/semantic.yaml` | Map annotations: rooms, surfaces (referencing geoms, so geometry is not duplicated), places, the user, aliases in Vietnamese and English. |
| `objects/catalog.yaml` | Object types (shape, size, mass, friction) and colours, plus perception thresholds. |
| `tasks/*.yaml` | Scenario: world, robot, start pose (+ jitter), objects (+ randomisation), unmapped obstacles, instruction, success criteria, time limit. |
| `build/` (generated) | `maps/*.npz/.pgm` prior map, `scenes/*.xml` exact MJCF of every run. |
| `artifacts/` (generated) | Benchmark results (JSON). |

A run is fully determined by `(task file, seed)`.

## Key design decisions

* **Mechanics that obey physics.** The first MVP balanced on two centre wheels and
  one rear caster, so the arm's weight tipped it 32° nose-down before it moved.
  The robot now has front and rear ball casters (frictionless contacts with
  `priority` so the floor does not add drag), a heavy low battery, and wheels
  that collide as thin ellipsoids (a cylinder touches at both rims and scrubs).
* **Real grasping.** No weld/attach constraint: the parallel-jaw fingers are
  force-limited position servos, objects are held by friction, and success is
  judged from finger force sensors and the measured grip width (an off-centre
  grasp closes on a chord and is rejected).
* **Real perception.** Objects are found in rendered RGB-D images: HSV colour
  segmentation, depth back-projection with the camera intrinsics, transform via
  kinematics + localisation, then height/width/taper classification
  (bottle vs cup vs can vs apple). Error is typically < 1 cm.
* **Real localisation.** The robot is told only its nominal start pose; the true
  pose is jittered. Odometry (encoders + gyro with bias) is corrected by
  aligning each LiDAR scan to a *signed* distance field of the map. (An unsigned
  field made thick obstacles a false optimum: 9 cm error next to the kitchen
  counter.)
* **Honest skills.** Every skill verifies its outcome from sensors. `place_object`
  backs away until the release point is in the camera view and only reports
  success if the object is seen there. Benchmarks count "false successes"
  (robot says done, referee disagrees) separately.
* **Generality.** Skills declare the capabilities they need (`navigate`,
  `perceive`, `manipulate`); the planner refuses plans the robot cannot execute.
  `robots/scout.yaml` (no arm) runs the same software.

## Sim-to-real path

The HAL is the boundary. A hardware driver implements `RobotInterface`; in ROS 2
terms:

| HAL method | ROS 2 equivalent |
|---|---|
| `command_wheels(l, r)` | `diff_drive_controller` wheel velocity command interfaces (`ros2_control`) |
| `command_joints({...})` | `JointTrajectoryController` / position command interfaces |
| `command_gripper(travel)` | `GripperActionController` (`control_msgs/GripperCommand`, with max effort) |
| `read_state()` | `/joint_states`, `/imu/data`, `/battery_state`, finger force topics, bumper |
| `read_scan()` | `/scan` (`sensor_msgs/LaserScan`) |
| `capture_rgbd()` | `/camera/color/image_raw` + `/camera/aligned_depth_to_color/image_raw` + `camera_info` |
| `emergency_stop()` | hardware e-stop relay / safety controller |

What stays: planner, executor, skills, world model, perception pipeline,
localisation and navigation algorithms (or swap them for AMCL/Nav2/MoveIt 2
behind the same skill interface). What must be redone for hardware: motor
drivers and tuning, sensor calibration (camera intrinsics/extrinsics, LiDAR
mount), a real map (SLAM), perception models trained on real images, and an
independent safety layer.

## Limits today (honest list)

* Colour-based detector: objects must have distinctive saturated colours from
  the catalog; no learned detector / VLM yet.
* Cartesian arm (lift, reach, wrist shift) with a parallel gripper: side grasps
  of upright objects on tables and counters; no floor picking, no 6-DoF arm,
  no lying objects.
* The prior map comes from the world geometry (a floor plan), not from SLAM.
* 2D LiDAR cannot see table tops or low objects; those come from the map, and
  dropped objects are added to the costmap by the robot itself.
* Battery, sensor noise, motor torque limits and watchdogs are modelled; motor
  heating, network latency and wheel wear are not.
* The control loop is cooperative and single-threaded at 50 Hz in simulated time.

## Next milestones

1. Learned perception: YOLO / OWL-ViT / a VLM behind `ObjectDetector.detect()`
   (open-vocabulary objects, people).
2. SLAM: build the map from LiDAR instead of from the world file.
3. 6-DoF arm description + MoveIt-style collision-aware motion planning
   (the IK in `kinematics.py` is already generic).
4. LLM agent loop: replanning from execution reports (`llm_planner.replan`),
   memory across missions, questions back to the user.
5. More worlds (warehouse, office) and robots; domain randomisation (lighting,
   textures, masses, friction) for sim-to-real robustness.
6. ROS 2 HAL driver and a physical robot.
