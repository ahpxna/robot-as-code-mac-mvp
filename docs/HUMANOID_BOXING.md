# Humanoid boxing

Two copies of `robots/humanoid_boxer.yaml` fight in a ring (`worlds/boxing_ring/arena.yaml`)
under a referee. Every fighter runs the same onboard software as the other robots in this
repository: it only reads its own sensors through the HAL and only commands its own servos.

```bash
source .venv/bin/activate
mjpython -m robotcode.match                                   # 3D viewer with scoreboard (macOS needs mjpython)
python -m robotcode.match --headless                          # 3 rounds, scorecard in the terminal
python -m robotcode.match --headless --rounds 1 --seed 7      # one round, another random seed
python -m robotcode.match --bench 12 --rounds 1               # 12 matches in parallel + statistics
python -m robotcode.match --rounds 1 --video build/video/match.mp4   # MP4 via ffmpeg (GIF without), no display needed
python -m unittest tests.test_humanoid -v
```

## The robot

| | |
|---|---|
| Body | 1.3 m, ~47 kg; pelvis, torso, head; 21 position-servoed joints: 6 per leg (hip yaw/roll/pitch, knee, ankle pitch/roll), waist yaw, 4 per arm (shoulder pitch/roll/yaw, elbow) |
| Actuators | MuJoCo position servos with stiffness, damping, torque limits and rotor inertia from the YAML |
| Sensors | IMU (gyro, accelerometer, fused orientation with a mounting bias), joint encoders and speeds, heel and toe force pads on each foot, impact sensors in the head, torso and gloves, RGB-D head camera |
| Gloves | padded (soft contact), 0.45 kg |

Everything is in `robots/humanoid_boxer.yaml`: geometry, masses, joint limits and gains, the
guard and block postures, the stance, the punch definitions and the balance/gait parameters.

## Software

```
head camera (RGB-D) ─► OpponentTracker ─► opponent head / torso / nearest point
                                     │
ScriptedBoxer (style) ───────────────┤  distance & angle management, combos, blocks, counters
                                     ▼
MoveExecutor ─ arm IK, straight-line fist paths, hooks, snap-back on contact ─► arm + waist targets
BalanceController ─ capture-point stance control + footwork ─► leg targets
                                     ▼
SimRobot (HAL, namespaced 'red_' / 'blue_')  ─►  MuJoCo
Referee (simulator ground truth): hits, knockdowns, clinch breaks, scores
```

**Leg kinematics** (`humanoid.py`): analytic 6-DOF leg IK (Kajita et al.), verified against
MuJoCo's forward kinematics to 1e-9 m.

**Balance and footwork** (`balance.py`):

- State estimation: pelvis pose in a floor-fixed stance frame from the loaded feet's kinematics
  and the IMU tilt; centre of mass from the robot's own mass model; COM velocity by
  differentiating the kinematics with joint speeds and the gyro (not the noisy position).
- Kinematic-inertial fusion: while both feet stand flat, the leg kinematics give the pelvis
  tilt without the IMU's mounting error; the error is learned during a 1.2 s start-up
  calibration and refined while standing quietly. Without this, a 1° IMU bias made the robot
  fall during footwork.
- Control: the capture point (divergent component of motion) tracks a reference solved
  backwards in time from a centre-of-pressure plan (Englsberger et al. 2015). The pelvis is
  moved against the capture-point error (ankle strategy); a slow trim absorbs model error;
  a sag integrator keeps the pelvis level against servo compliance; gravity/load torques are
  fed forward to the servos. Integrators are frozen while the robot is off balance or a foot
  rolls on an edge. A foot that rolls is pushed back flat using its measured tilt.
- Footwork: boxing step-drag. A move (forward / left / turn) becomes two steps; the pressure
  shifts onto the stance foot, the other foot lifts, travels and lands; the plan continues from
  where the foot really landed, and every move re-lays the configured stance so landing errors
  never accumulate. A step does not lift off while the robot is off balance: the weight returns
  to both feet first, and the move is dropped if it does not settle.

**Punches** (`boxing.py`, `moves:` in the YAML): jab, cross, body shot (dips to stomach height
first) and hook (winds wide, then twists through). Straight punches drive the fist along a
straight line by per-tick arm IK and aim at the point of the target the opponent's guard covers
least (clearance of the punch line from the opponent's points seen by the camera). A punch snaps
back as soon as its glove feels contact, once the fist has covered 60 % of its path; earlier
contacts are glove clashes.

**Perception**: the opponent is segmented by shell colour in the head camera, back-projected
with depth and transformed with the robot's own kinematics and IMU tilt. Head error ~2 cm at
0.6–1.4 m. Up close, the nearest visible point of the opponent is used to back off.

**Scripted fighter** (`match.py`, styles in `tasks/boxing/styles.yaml`): `pressure`, `counter`,
`outboxer` differ in preferred range, aggression, combos and how they react to an incoming punch
(block or counter). Footwork happens only when clearly needed (every step is time on one foot).

**Referee**: a glove contact on the opponent's head or torso whose peak force reaches
`hit_force` is a landed punch (head 2, body 1 point); contact on gloves or arms is blocked. A
robot is down when anything but its feet touches the canvas (or its pelvis drops below 0.4 m);
ten seconds down is a knockout. Contact between the robots that lasts much longer than a punch is
a clinch: the referee calls *break* and both robots receive the signal, as from a competition
referee box.

## Measured

| Test | Result |
|---|---|
| Footwork: advance ×3, retreat ×3, side steps, turns ±20–30°, queued moves; 5 seeds each | 30 / 30 without a fall, max tilt ≤ 12° |
| Push recovery standing (0.1 s push on the torso) | 16–20 N·s in every direction (no balance control: 8–16) |
| Punching in the air: 15 punches of each type per run, 3 seeds | 15 / 15 runs standing, capture-point error < 5 cm |
| Opponent tracker vs ground truth | head ~2 cm, torso ~6 cm |
| Matches: 12 seeds × one 45 s round (`--bench 12 --rounds 1`) | 8 / 12 go the distance, 0.67 knockdowns per match, 16 punches thrown and 3.2 landed per match |

## Reinforcement learning

`src/robotcode/rl/boxing_env.py` is a Gymnasium environment for learning **tactics**: every
0.2 s the policy picks one of 12 macros (idle, advance, retreat, step left/right, turn
left/right, jab, cross, hook, body shot, block); balance, footwork and punch execution stay the
model-based controllers. Reward: points landed minus points taken, knockdown bonus, −10 for
falling, small shaping for distance and facing. The opponent is the scripted boxer.

```bash
pip install -e '.[rl]'
python scripts/train_boxing.py --timesteps 200000 --n-envs 6      # PPO, CPU, about 2 hours on 6 cores
python scripts/train_boxing.py --eval build/rl/boxing_ppo.zip --episodes 5
python scripts/train_boxing.py --random --episodes 3              # baseline
```

`privileged_opponent=True` (default) reads the opponent from the simulator for speed; `--camera`
uses the robot's own camera tracker as on hardware.

### Moving to an NVIDIA GPU

The tactics environment is CPU-bound Python (≈1× real time per environment) and is meant for a
laptop. The GPU pays off for **low-level skills** trained end to end:

- Port the robot to MuJoCo MJX (or `mujoco_playground` / Brax): the same MJCF, without the camera
  and with simplified contacts (capsule/box only, fewer contact pairs); thousands of parallel
  environments on one GPU.
- Train joint-level policies there: push recovery with stepping, walking, getting up after a
  knockdown, punches. Typical: PPO with domain randomisation (mass, friction, servo gains, IMU
  bias, latency), tens of millions of steps in under an hour.
- What does not port as is: Python-side controllers (`BalanceController`, `MoveExecutor`, the
  tracker) run per environment on the CPU; rendering for the camera is not available in MJX
  (use privileged observations or a learned state estimator for training, the camera at test
  time).
- Keep the hierarchy: a learned low-level policy replaces `BalanceController.update` behind the
  same interface; the tactics policy and the referee stay unchanged.

## Limits

- No get-up skill: a knocked-down robot stays down (a learned get-up is the obvious next RL task).
- No reactive stepping for large pushes: above ~20 N·s the robot falls; some knockdowns in
  matches come from a punch or a glove clash while on one foot.
- Footwork is quasi-static (~0.1 m per 1.9 s step pair); the robots do not circle quickly.
- The opponent is found by colour; a red or blue background object near the ring would confuse it.
