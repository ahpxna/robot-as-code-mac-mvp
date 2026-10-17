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
        │                       ── perception.py     RGB-D detector (colour segmentation, or OWLv2 boxes from
        │                                            perception_learned.py) -> 3D -> shape classification,
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
| `objects/catalog.yaml` | Object types (shape, size, mass, friction) and colours, plus perception settings: which detector runs (`perception.detector: color \| learned`), colour thresholds, and the learned detector's model, device, input size and thresholds (`perception.learned`). |
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
  (bottle vs cup vs can vs apple). Error is typically < 1 cm. See
  [Perception](#perception) for the learned, open-vocabulary alternative.
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

## Perception

Two detectors implement the same `detect(frame, pose, joints, exclude) -> list[Detection]`,
so skills, the world model, navigation and manipulation do not know which one runs.
Both end in the same geometric back end (`GeometricDetector` in `perception.py`):
pixels of one object + depth -> back-projection with the camera intrinsics -> map
frame (camera kinematics + localisation) -> support surface from the prior map ->
height, width across the view ray, top/middle taper -> fit against every catalog
type's dimensions -> centre from the visible top surface.

| | `color` (default) | `learned` |
|---|---|---|
| Which pixels are one object | HSV mask per catalog colour, connected components | OWLv2 box per text query; inside the box: the depth layer of its centre, points within 10 cm of it, above the support surface |
| Colour | the mask's hue | the model's best colour query (any colour the text encoder can name, e.g. white, black) |
| Type | metric size + shape fit | metric size + shape fit, with the model's type scores as a weak prior (`type_weight`) |
| Dependencies | numpy/scipy | `pip install -e '.[vision]'` (torch, transformers); ~600 MB weights, cached after first use |
| Code | `perception.ObjectDetector` | `perception_learned.LearnedObjectDetector` |

Selection, highest precedence first: `--detector` (cli, evaluate) > robot YAML
`perception.detector` > `objects/catalog.yaml` `perception.detector` > `color`.
`perception.learned` in the catalog sets the model (`google/owlv2-base-patch16-ensemble`),
device (`auto`: MPS on Apple Silicon, CUDA, else CPU), dtype (`auto`: float16 on
a GPU, float32 on CPU), input size (960 px), prompt template and thresholds.

Text queries are every catalog colour x type, filled into
`"a photo of a {color} {type}"`. Types and colours may list English `prompts`
(variants, e.g. `[cup, mug]`); the default is the entry's name. The `aliases`
are not used as queries: they mix Vietnamese and English for the language
planner, and the model's text encoder only knows English. The text embeddings
are computed once; each frame costs one image-encoder pass.

**Why metric size decides the type.** The simulated objects are flat-shaded
primitives: a red cylinder scores higher as "cup" than as "bottle", a blue cup
as "can". With the model's type label alone (`type_weight` very large) recall
drops from 100 % to 45 % on the benchmark frames below; with depth geometry
deciding and the model breaking near-ties, no frame has a wrong type. The model
contributes what geometry cannot: *that* something is an object worth
measuring, where it is in the image, and its colour, including colours with no
hue to segment by.

### Measured (Apple M1 Pro, 2026-10-16)

Frame level, `python -m robotcode.perception_bench --episodes 5`: 7 tasks x 5 seeds, the
160 frames the robot captured in `look()`; both detectors see the *same* frames and are
scored against simulator ground truth. Correct = within 10 cm of a ground-truth object
with the same type and colour; recall counts objects in view (centre in the image, not
occluded, 0.3-3 m). Error is the 3D distance of the object centre.

| | `color` | `learned` (OWLv2, fp16 on MPS) |
|---|---|---|
| Precision | 99.3 % (1 wrong label: a cup cut off at the image edge measured as an apple) | 100 % |
| Recall (objects in view) | 100 % | 100 % |
| False positives | 0 | 0 |
| Position error median / p95 / max | 6.3 / 16.4 / 27.5 mm | 6.3 / 19.8 / 27.7 mm |
| ... with the true robot pose (perception only) | 6.6 / 16.7 / 26.2 mm | 6.6 / 19.8 / 28.1 mm |
| Latency per frame, mean / p95 | 6 / 7 ms (CPU) | 411 / 424 ms (GPU) |
| First call | - | model load 1.7-6.5 s + ~1 s warm-up, once per process |
| White cup / black can (`tasks/perception/`, 19 frames) | recall 0 % (no hue) | recall 100 %, median error 7.1 mm |

The median is ~6 mm for both (the geometry is shared); the tail (p95/max) is
objects partly outside the image or 2-3 m away, where both cut out the same
partial top surface.

Variants of the learned detector on the same 160 frames:

| Variant | Recall | Precision | Latency mean / p95 | |
|---|---|---|---|---|
| default: 960 px, float16, threshold 0.03 | 100 % | 100 % | 411 / 424 ms | |
| float32 | 100 % | 100 % | 616 / 1103 ms | identical detections to float16 |
| 640 px input | 96 % | 99 % | 191 / 209 ms | misses the apple at range |
| model's type label decides (`type_weight` 1000) | 45 % | 96 % | | primitives look alike to the model |
| score threshold 0.02 | 100 % | 99 % | | first false positives |

Closed loop, `python -m robotcode.evaluate --episodes 20 --detector color|learned`
(20 seeds per task; the learned run uses 4 workers sharing the GPU, so
`det[ms]` there is 0.6-1.3 s per frame):

| Task | `color` | `learned` |
|---|---|---|
| `apple_desk_to_coffee_table` | 20/20 | 20/20 |
| `blocked_corridor` | 20/20 | 20/20 |
| `cup_to_drop_zone` | 20/20 | 20/20 |
| `deliver_bottle_to_user` | 20/20 | 20/20 |
| `fetch_red_bottle_kitchen_to_desk` | 20/20 | 20/20 |
| `go_to_kitchen` (no perception) | 20/20 | 20/20 |
| `scout_find_apple` | 20/20 | 20/20 |
| **Default benchmark** | **140/140** | **140/140** |
| Collisions / false successes | 0 / 0 | 0 / 0 |
| Wall time | 149 s (8 workers) | 455 s (4 workers) |
| `perception/white_cup_to_desk` (white cup, black can distractor) | 0/20 | 17/20 |

The three learned failures on the white cup are honest ones ("failed to pick up
the white cup after 2 attempts"): from the grasp approach pose the cup sits at
the bottom edge of the image, only a white sliver against the white floor is
visible, and OWLv2 gives it a score below threshold (0.028 at 0.05; 0.03 is the
lowest threshold without false positives in the benchmark). The colour
detector tolerates such slivers because any coloured patch is enough;
the learned one needs to recognise the object.

Limits of the learned detector today:

* Boxes, not masks: the object is cut out of its box with depth (layer, radius,
  above the support), so two objects touching in 3D inside one box can merge.
* A type still needs catalog dimensions (shape, radius, height) to get a 3D
  centre and pass the size check: open vocabulary for appearance, not yet for
  geometry of unknown objects.
* OWLv2 is trained on photographs; these renders are untextured primitives.
  Real images will look more like its training data, simulated ones less.
* Objects cut off by the image border are found less often than by the colour
  detector (see the white-cup failures above).
* Detection runs synchronously inside `look()`; the simulated clock does not
  advance meanwhile. On a robot `look()` would hold still ~0.5 s longer.

## Extra sensors on the rover, and what they taught

* **3D LiDAR** (16 channels, looking down from the head) feeds a separate 3D costmap layer, so a 2D
  beam passing *over* a low box cannot clear it. Lessons: mask returns from the robot's own body
  and from the **object it is holding** (only that object's volume: masking a wider radius blinded
  the robot to the obstacle its load was about to hit); and require two returns per cell per sweep,
  because range noise lifts the odd floor return above any small height threshold and those
  phantom cells blocked approach poses.
* **ToF ring** (8 short-range sensors at 6 cm) sees what is right beside the chassis. It made
  turning in tight spots *harder* until rotation checks covered the whole turn (not its first
  1.2 rad), used a smaller margin for slow in-place turns than for driving, and the escape
  manoeuvre drove towards open space (forwards or backwards) until a half turn was free.
* **Wrist camera** checks for neighbours before the arm sweeps over a table and refines the object
  position (3-4 mm) before grasping; it is skipped next to a same-coloured neighbour, where the two
  blobs merge at 15 cm.
* The object memory merges detections of the same colour at the same spot across *confusable*
  shapes only (cup / apple / can), and a grasp never continues if its target's type changes.

## Legged robots

`robots/spider_bot.yaml` (8 legs × coxa/femur/tibia) runs the same stack. What changes is
isolated behind the interfaces the wheeled robot already used:

* **Description** – `base.kind: legged`: body ellipsoids, leg chains with position servos,
  encoders, effort sensors and foot touch sensors are generated from the YAML. Postures are
  given as foot positions (`stand`, `fold`); joint angles always come from the leg IK.
* **Locomotion** – `legged.GaitController` implements the same drive interface as the
  differential-drive controller (`command(v, w)`, `halt()`, limits), so navigation, docking and
  skills are unchanged. Inside: alternating-tetrapod gait, stance feet slide under the body,
  swing feet arc to a Raibert-style touchdown, analytic 3-joint IK per leg.
* **State estimation** – no wheel encoders: odometry uses the body velocity the gait produces
  and the gyro; LiDAR scan matching corrects it as before. The body height above the floor
  (needed to turn camera/LiDAR points into heights) comes from the kinematics of the feet in
  contact (`RobotStack.base_height`).
* **Maps per robot** – the prior map is rasterised for each robot's height and LiDAR height,
  so table tops are obstacles for the 1.4 m rover but a roof for the 12 cm spider.
* **Transform (fold / unfold)** – a capability (`transform`) with its own skills; the body is
  lowered onto the floor with feet planted, then each tetrapod group lifts and tucks its feet.
  `tests/test_legged.py` checks the whole fold path for self-collision with `mj_geomDistance`.

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

* Perception knows only catalog types: the default colour detector needs
  saturated catalog colours; the learned detector (OWLv2, optional) handles any
  nameable colour but still needs each type's dimensions, needs most of the
  object in view, and is ~60x slower per frame (0.4 s on an M1 Pro GPU). No
  people detection, no VLM.
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

1. Learned perception, next steps: masks (SAM) instead of box + depth cut-out,
   types without catalog dimensions (size from the point cloud), people, a VLM
   for attributes; domain randomisation so simulated images look less synthetic.
   (Open-vocabulary OWLv2 boxes are in: `--detector learned`.)
2. SLAM: build the map from LiDAR instead of from the world file.
3. 6-DoF arm description + MoveIt-style collision-aware motion planning
   (the IK in `kinematics.py` is already generic).
4. LLM agent loop: replanning from execution reports (`llm_planner.replan`),
   memory across missions, questions back to the user.
5. More worlds (warehouse, office) and robots; domain randomisation (lighting,
   textures, masses, friction) for sim-to-real robustness.
6. ROS 2 HAL driver and a physical robot.
