# Robot-as-Code (macOS, MuJoCo)

Robots built entirely in software and defined as code: robot descriptions
(YAML → generated MJCF), the house, objects and missions are configuration
files; physics, sensors, motors and contacts are simulated in MuJoCo; one
onboard software stack (localisation, navigation, perception, grasping, gait,
safety, task planning) runs every robot against a hardware abstraction layer
that a real robot driver can implement later.

| Robot | Body | Can |
|---|---|---|
| `helper_rover` | 25 kg wheeled mobile manipulator, lift + telescoping arm, gripper | fetch and deliver objects |
| `spider_bot` | 1.4 kg eight-legged walker, 24 servos | walk under furniture, search, fold into a compact spider "logo" |
| `scout` | arm-less wheeled rover | patrol and search |
| `humanoid_boxer` | 1.3 m, 47 kg biped, 21 servos, padded gloves | box another robot in a ring: balance, footwork, punches, camera tracking |

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
robot remembers, the green arrow its believed pose. `--camera follow` makes the
camera track the robot (default for small robots), `--camera house` shows the
whole apartment.

The spider robot (same commands, another task file or `--scenario` with `robot: spider_bot`):

```bash
mjpython -m robotcode.cli --scenario demo_spider                    # boots folded, unfolds, walks, folds again
mjpython -m robotcode.cli --scenario spider_fold_in_office          # walk to the office, fold into the logo
python -m robotcode.cli --headless --scenario spider_find_under_table   # finds an apple under the kitchen table
mjpython -m robotcode.cli --scenario spider_fold_in_office --interactive   # then type: "Mở ra rồi đi vào bếp"
```

Robot boxing (two humanoids, a referee, three rounds; details in [docs/HUMANOID_BOXING.md](docs/HUMANOID_BOXING.md)):

```bash
mjpython -m robotcode.match                          # 3D viewer with scoreboard
python -m robotcode.match --headless --rounds 1      # scorecard in the terminal
python -m robotcode.match --bench 12 --rounds 1      # 12 matches in parallel + statistics
python -m robotcode.match --rounds 1 --video build/video/match.mp4   # record (offscreen, no display needed)
pip install -e '.[rl]' && python scripts/train_boxing.py --timesteps 200000   # learn tactics with PPO
```

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
| Robots | `robots/helper_rover.yaml`: 25 kg differential drive, front/rear ball casters, 150 Wh battery model, lift 0.24–1.12 m + telescoping reach 0.45 m + wrist shift, parallel gripper (25 N), RGB-D head camera, 270° 2D LiDAR, 16-channel 3D LiDAR, 8 time-of-flight sensors, wrist camera, IMU, encoders, finger force sensors, bumper. `robots/spider_bot.yaml`: 8 legs × 3 servos, foot contact sensors, mini LiDAR, camera, IMU. `robots/scout.yaml`: arm-less rover. |
| World | 3-room apartment (living room, kitchen, office), furniture, a seated user; unmapped obstacles per task |
| Physics | MuJoCo 3: gravity, friction cones, contacts, torque-limited motors, joint limits |
| Localisation | wheel + gyro odometry, LiDAR scan matching on a signed distance field (mm-level at rest) |
| Navigation | per-robot map (a 12 cm spider walks under tables), costmap with 2D LiDAR and 3D (3D LiDAR / ToF) obstacle layers, Dijkstra planner, pure pursuit, full-sweep rotation checks, stall detection, escape manoeuvres, replanning |
| Locomotion | wheels: differential-drive controller; legs: analytic 3-joint leg IK, alternating-tetrapod gait, posture transitions (fold / unfold) with self-collision-checked paths |
| Perception | RGB-D detector → 3D points → shape classification (bottle / cup / can / apple), object memory. Detector: HSV colour segmentation (default) or open-vocabulary OWLv2 boxes (`--detector learned`) with the same depth geometry |
| Manipulation | IK from the robot description, collision-aware approach selection (angled approaches past neighbours), wrist-camera refinement of the last centimetres, force-sensed grasp check, carry monitoring, camera-verified placing |
| Agent | VN/EN rule planner (accent-insensitive), optional LLM planner, executor with recovery, honest mission report |
| Evaluation | referee: success criteria per task, collisions, object bumps, false successes, localisation error, energy |

## Benchmark (20 fresh random seeds per task)

`python -m robotcode.evaluate --episodes 20` (object positions, robot start and
sensor noise vary with the seed):

| Task | Robot | What it exercises | Success | Collisions | False successes | Mean time |
|---|---|---|---|---|---|---|
| `deliver_bottle_to_user` | rover | the Vietnamese headline task: living room table → user | 20/20 | 0 | 0 | 49 s |
| `fetch_red_bottle_kitchen_to_desk` | rover | red bottle next to a red cup (grounding, clutter) | 20/20 | 0 | 0 | 69 s |
| `cup_to_drop_zone` | rover | 0.9 m counter pick, floor drop-off | 20/20 | 0 | 0 | 74 s |
| `apple_desk_to_coffee_table` | rover | sphere grasp, English instruction | 20/20 | 0 | 0 | 65 s |
| `blocked_corridor` | rover | unmapped obstacle, LiDAR replanning | 20/20 | 0 | 0 | 78 s |
| `low_obstacles` | rover | clutter below / above the 2D scan plane (3D LiDAR, ToF) | 20/20 | 0 | 0 | 26 s |
| `go_to_kitchen` | rover | navigation | 20/20 | 0 | 0 | 10 s |
| `scout_find_apple` | scout | arm-less robot, same software | 20/20 | 0 | 0 | 75 s |
| `spider_fold_in_office` | spider | eight-legged walking, folding into the logo | 20/20 | 0 | 0 | 47 s |
| `spider_find_under_table` | spider | walking under furniture, finding an apple under a table | 20/20 | 0 | 0 | 17 s |

200 episodes (seeds 600–619, never used for tuning) in about 4.5 minutes on 7 cores; no robot
touched an object it was not asked to handle. Ablation: with the 3D LiDAR and ToF switched off
(`tasks/ablation/low_obstacles_2d_only.yaml`) the rover drives under the low console and gets stuck.
Known hard case: `fetch_red_bottle_kitchen_to_desk` seed 9 (a red cup hidden behind the red bottle
from every camera until the fingers are there) fails safely: no false success, the cup stays put.

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
worlds/boxing_ring/ arena.yaml (ring, corners)          tasks/boxing/      match rules, fighting styles
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
  legged.py        leg IK, gait, fold/unfold postures    stack.py           onboard software loop
  runtime.py       simulation + referee
  evaluate.py      episodes and benchmarks              cli.py             command line
  humanoid.py      humanoid joints, stances, leg IK      balance.py         capture-point balance + footwork
  boxing.py        punches, arm IK, opponent tracker     arena.py           ring + namespaced robots
  match.py         referee, scripted boxer, matches      rl/boxing_env.py   Gymnasium env (tactics)
scripts/train_boxing.py   PPO training / evaluation
tests/             unit, physics, sensor, legged-robot and mission tests
tasks/ablation/    the same task with sensors switched off (to measure what they buy)
```

## Safety note

Nothing here is certified. Before driving real motors, add an independent
hardware e-stop, speed/force limits enforced in the motor drivers, and
collision monitoring that does not depend on this software.
