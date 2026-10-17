# Robot-as-Code (macOS, MuJoCo)

A household mobile manipulator built entirely in software and defined as code:
robot description (YAML → generated MJCF), house, objects and missions are
configuration files; physics, sensors, motors and contacts are simulated in
MuJoCo; the onboard software (localisation, navigation, perception, grasping,
safety, task planning) runs against a hardware abstraction layer that a real
robot driver can implement later.

You give it a task in Vietnamese or English:

```bash
python -m robotcode.cli --headless --task "Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao."
```

and it plans, drives through the house, finds the bottle with its camera,
grasps it by friction, carries it to the user and checks with the camera that
it is there. An independent referee then scores the episode from simulator
ground truth.

Vietnamese quick start: [`docs/START_HERE_VI.md`](docs/START_HERE_VI.md). Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install (macOS, Python 3.11+)

```bash
bash scripts/bootstrap_mac.sh
```

Creates `.venv`, installs `mujoco numpy scipy pyyaml`, runs the test suite and
one full mission headless. No ROS, Docker, GPU, or API key needed.

## Run

```bash
source .venv/bin/activate
python -m robotcode.cli --dry-run --task "Find the red bottle in the kitchen and bring it to my desk"   # plan only
python -m robotcode.cli --smoke --headless                                                             # scene loads
python -m robotcode.cli --headless --task "Mang cái cốc xanh dương tới đích"                           # full mission
mjpython -m robotcode.cli --task "Đi vào bếp"                       # 3D viewer (macOS needs mjpython)
mjpython -m robotcode.cli --interactive                             # viewer + type tasks one after another
python -m robotcode.cli --scenario fetch_red_bottle_kitchen_to_desk --seed 9 --headless   # a task file
python -m robotcode.evaluate --episodes 20                          # benchmark all tasks
python -m unittest discover -s tests -v
```

In the viewer, blue dots are the planned path, yellow rings the objects the
robot remembers, the green arrow its believed pose.

Optional LLM planner (writes a JSON plan of skill calls that is validated
before anything moves): `pip install -e '.[llm]'`, set `OPENAI_API_KEY`, add
`--planner llm`. It costs API usage; the rule planner is the default.

Optional learned object detector (open-vocabulary OWLv2, runs on the Apple-Silicon
GPU via MPS, or CPU; no server):

```bash
pip install -e '.[vision]'                       # torch + transformers; ~600 MB of weights download on first use
python -m robotcode.cli --headless --detector learned --task "Mang chai nước màu đỏ đến cho tao"
python -m robotcode.cli --headless --detector learned --scenario tasks/perception/white_cup_to_desk.yaml
python -m robotcode.evaluate --episodes 20 --detector learned        # 4 workers: each loads the model
python -m robotcode.perception_bench --episodes 5     # both detectors on the same frames vs ground truth
```

The detector is chosen by `perception.detector: color | learned` in
`objects/catalog.yaml` (default `color`), a robot's own `perception.detector`
in `robots/*.yaml`, or `--detector` (highest precedence). See
[docs/ARCHITECTURE.md#perception](docs/ARCHITECTURE.md#perception).

## What is in the box

| Layer | Implementation |
|---|---|
| Robot | `robots/helper_rover.yaml`: 25 kg differential drive, front/rear ball casters, 150 Wh battery model, lift 0.24–1.12 m + telescoping reach 0.45 m + wrist shift, parallel gripper (25 N), RGB-D head camera, 270° LiDAR, IMU, encoders, finger force sensors, bumper. `robots/scout.yaml`: arm-less rover on the same software. |
| World | 3-room apartment (living room, kitchen, office), furniture, a seated user; unmapped obstacles per task |
| Physics | MuJoCo 3: gravity, friction cones, contacts, torque-limited motors, joint limits |
| Localisation | wheel + gyro odometry, LiDAR scan matching on a signed distance field (mm-level at rest) |
| Navigation | costmap + live LiDAR obstacle layer, Dijkstra planner, pure pursuit, collision forecast, stall detection, replanning |
| Perception | RGB-D detector → 3D points → shape classification (bottle / cup / can / apple), object memory. Detector: HSV colour segmentation (default) or open-vocabulary OWLv2 boxes (`--detector learned`) with the same depth geometry |
| Manipulation | IK from the robot description, collision-aware approach selection, force-sensed grasp check, carry monitoring, camera-verified placing |
| Agent | VN/EN rule planner (accent-insensitive), optional LLM planner, executor with recovery, honest mission report |
| Evaluation | referee: success criteria per task, collisions, object bumps, false successes, localisation error, energy |

## Benchmark (20 random seeds per task)

`python -m robotcode.evaluate --episodes 20` (object positions, robot start and
sensor noise vary with the seed):

| Task | What it exercises | Success | Collisions | False successes | Mean time |
|---|---|---|---|---|---|
| `deliver_bottle_to_user` | the Vietnamese headline task: living room table → user | 20/20 | 0 | 0 | 47 s |
| `fetch_red_bottle_kitchen_to_desk` | red bottle next to a red cup (grounding, clutter) | 20/20 | 0 | 0 | 68 s |
| `cup_to_drop_zone` | 0.9 m counter pick, floor drop-off | 20/20 | 0 | 0 | 78 s |
| `apple_desk_to_coffee_table` | sphere grasp, English instruction | 20/20 | 0 | 0 | 64 s |
| `blocked_corridor` | unmapped obstacle, LiDAR replanning | 20/20 | 0 | 0 | 74 s |
| `go_to_kitchen` | navigation | 20/20 | 0 | 0 | 10 s |
| `scout_find_apple` | arm-less robot, same software | 20/20 | 0 | 0 | 71 s |

140 episodes run in about 5 minutes on 8 cores (MuJoCo runs ~10x faster than real time).
Max localisation error over all episodes: 7 mm.

### Colour vs learned detector

Same benchmark with `--detector learned` (OWLv2 boxes + the same depth geometry), and
both detectors on the same 160 camera frames (`python -m robotcode.perception_bench --episodes 5`):

| | `color` (default) | `learned` |
|---|---|---|
| Default benchmark, 7 tasks x 20 seeds | 140/140 | 140/140 |
| Collisions / false successes | 0 / 0 | 0 / 0 |
| `tasks/perception/white_cup_to_desk` (white cup, black can) | 0/20 | 17/20 |
| Frame precision / recall | 99.3 % / 100 % | 100 % / 100 % |
| Object position error, median / p95 | 6.3 / 16.4 mm | 6.3 / 19.8 mm |
| Time per frame (M1 Pro) | 6 ms (CPU) | 411 ms (GPU, MPS, float16) |

Details, variants (input size, precision, thresholds) and failure analysis:
[docs/ARCHITECTURE.md#perception](docs/ARCHITECTURE.md#perception).

## Repository

```
robots/            robot descriptions (YAML)            worlds/apartment/  world.xml + semantic.yaml
objects/           object catalog                       tasks/             missions + success criteria
                                                        tasks/perception/  tasks only the learned detector can do
src/robotcode/
  description.py   YAML -> MJCF generator               scene.py           world + robot + objects -> model
  mapping.py       prior map (occupancy, LiDAR SDF)     hal.py             RobotInterface + SimRobot
  kinematics.py    FK / IK / footprint                  localization.py    odometry + scan matching
  navigation.py    costmap, planner, follower, docking  perception.py      RGB-D detectors + world model
  perception_learned.py  OWLv2 open-vocabulary detector  perception_bench.py  detector comparison vs ground truth
  manipulation.py  arm + gripper control                safety.py          bumper / tilt / battery
  skills.py        skill library (LLM tool list)        planner.py         VN/EN rule planner
  llm_planner.py   optional LLM planner + validation    agent.py           mission executor
  stack.py         onboard software loop                runtime.py         simulation + referee
  evaluate.py      episodes and benchmarks              cli.py             command line
tests/             unit, physics and mission tests
```

## Safety note

Nothing here is certified. Before driving real motors, add an independent
hardware e-stop, speed/force limits enforced in the motor drivers, and
collision monitoring that does not depend on this software.
