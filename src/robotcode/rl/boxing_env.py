"""Gymnasium environment: learn boxing tactics on top of the model-based humanoid controller.

    pip install -e '.[rl]'
    python scripts/train_boxing.py --timesteps 200000 --n-envs 6        # PPO on a laptop CPU
    python scripts/train_boxing.py --eval build/rl/boxing_ppo.zip --episodes 5

Hierarchy: the policy picks one of 12 tactical macros every 0.2 s (step in, slip out, jab, block,
...). The balance controller, footwork planner, arm IK and punch timing stay model based, so a
policy only has to learn *when* to do what, which is tractable on a CPU in hours. The opponent is
the scripted boxer (robotcode.match), scoring comes from the same referee as real matches.

Observation (float32, all roughly in [-1, 1]):
    own balance: capture-point error (heading frame, 2), COM velocity (2), tilt, stepping,
    punching, blocking; opponent: head position in my pelvis frame (3), distance, bearing (sin, cos),
    opponent down; round time left; recent hits landed / taken (head, body).
`privileged_opponent=True` reads the opponent from the simulator (fast: no rendering); False uses
the robot's own head-camera tracker, as on the real robot.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("robotcode.rl needs gymnasium: pip install -e '.[rl]'") from exc

from ..boxing import OpponentEstimate
from ..config import load_task
from ..match import Match, ScriptedBoxer

MACROS = ("idle", "advance", "retreat", "step_left", "step_right", "turn_left", "turn_right",
          "jab", "cross", "hook", "body", "block")
DECISION_TICKS = 20                       # 0.2 s at 100 Hz
OBS_DIM = 21


class AgentBoxer(ScriptedBoxer):
    """A fighter whose tactics come from outside (a policy); everything below them is onboard."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_truth = None          # callable -> OpponentEstimate, set by the env when privileged

    def tick(self, now: float) -> None:
        if self.ground_truth is not None:
            self.next_camera = math.inf                       # no rendering: the env supplies the opponent
            est = self.ground_truth(now)
            self.tracker.estimate = est
            self.distance, self.bearing = est.distance, est.bearing
        super().tick(now)

    def _decide(self, now: float) -> None:                    # the policy decides instead
        return

    def act(self, macro: str, now: float) -> bool:
        """Execute one macro if possible; returns False when it was not applicable."""
        bal, mv, est = self.balance, self.moves, self.tracker.estimate
        if self.fallen or not bal.ready:
            return macro == "idle"
        if macro == "idle":
            return True
        if macro == "block":
            mv.guard_name, self.block_until = "block", now + 0.6
            return True
        if macro in ("jab", "cross", "hook", "body"):
            if mv.busy or bal.stepping or est is None:
                return False
            obstacles = est.points[est.points[:, 0] < est.head[0] - 0.06] if len(est.points) else None
            return mv.start(macro, {"head": est.head, "body": est.body}, now, obstacles)
        if bal.stepping or mv.busy:
            return False
        moves = {"advance": dict(forward=0.08), "retreat": dict(forward=-0.08), "step_left": dict(left=0.08),
                 "step_right": dict(left=-0.08), "turn_left": dict(turn=0.25), "turn_right": dict(turn=-0.25)}
        bal.move(**moves[macro])
        return True


class BoxingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, match: str = "boxing/exhibition", agent: str = "red", privileged_opponent: bool = True,
                 round_s: float = 45.0, render_mode: str | None = None):
        self.cfg = load_task(match)
        self.agent = agent
        self.privileged = privileged_opponent
        self.round_s = round_s
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(MACROS))
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.match: Match | None = None
        self._renderer = None

    # ---- gym API -------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        gap = float(rng.uniform(0.9, 1.3))
        yaw = [float(rng.uniform(-10, 10)) for _ in range(2)]
        cfg = {**self.cfg, "start": {"red": [-gap / 2, 0.0, yaw[0]], "blue": [gap / 2, 0.0, 180.0 + yaw[1]]},
               "rules": {**self.cfg.get("rules", {}), "rounds": 1, "round_s": self.round_s}}
        if self.match is not None:
            self.match.close()
        self.match = Match(cfg, seed=int(rng.integers(1 << 30)), verbose=False, brains={self.agent: AgentBoxer})
        self.me = self.match.fighters[self.agent]
        self.opponent_name = next(c for c in self.match.fighters if c != self.agent)
        if self.privileged:
            self.me.ground_truth = self._truth
        self._seen_events = 0
        self._recent: list[tuple[float, str, str]] = []       # (t, 'landed'/'taken', target)
        self._knockdowns_seen = 0
        while not self.me.balance.ready:                     # start-up calibration
            self.match.step()
        return self._observe(), {}

    def step(self, action: int):
        m = self.match
        macro = MACROS[int(action)]
        valid = self.me.act(macro, m.time)
        reward = 0.0 if valid else -0.01
        for _ in range(DECISION_TICKS):
            m.step()
            if self._done():
                break
        reward += self._collect()
        est = self.me.tracker.estimate
        if est is not None:
            reward -= 0.05 * abs(est.distance - 0.64) + 0.02 * abs(est.bearing)
        terminated = bool(self.me.fallen or m.referee.knockout is not None)
        if self.me.fallen:
            reward -= 10.0
        truncated = bool(m.time >= self.round_s) and not terminated
        info = {"score": m.referee.score(), "time": m.time, "macro": macro, "valid": valid}
        return self._observe(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.match is None:
            return None
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.match.model, 360, 480)
        self._renderer.update_scene(self.match.data, camera="ringside")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.match is not None:
            self.match.close()
            self.match = None

    # ---- internals -----------------------------------------------------------------
    def _done(self) -> bool:
        m = self.match
        return self.me.fallen or m.referee.knockout is not None or m.time >= self.round_s

    def _truth(self, now: float) -> OpponentEstimate:
        """Opponent head/torso in my pelvis frame, straight from the simulator (training only)."""
        d = self.match.data
        me, other = self.me.corner.prefix, self.match.fighters[self.opponent_name].corner.prefix
        base = d.body(me + "base")
        rot, pos = base.xmat.reshape(3, 3), base.xpos
        head = rot.T @ (d.geom(other + "head").xpos - pos)
        body = rot.T @ (d.geom(other + "torso").xpos - pos)
        nearest = min(float(np.linalg.norm((rot.T @ (d.geom(other + g).xpos - pos))[:2]))
                      for g in ("L_glove", "R_glove", "torso", "head"))
        return OpponentEstimate(now, head, body, np.zeros((0, 3)), head, float(np.linalg.norm(head[:2])),
                                math.atan2(head[1], head[0]), nearest)

    def _collect(self) -> float:
        """Reward from new referee events (points landed minus points taken)."""
        m = self.match
        ref, pts = m.referee, m.referee.points
        reward = 0.0
        for e in ref.events[self._seen_events:]:
            if e.target == "blocked":
                continue
            if e.attacker == self.agent:
                reward += pts[e.target]
                self._recent.append((e.t, "landed", e.target))
            else:
                reward -= pts[e.target]
                self._recent.append((e.t, "taken", e.target))
        self._seen_events = len(ref.events)
        for t, c in ref.knockdowns[self._knockdowns_seen:]:
            reward += pts["knockdown"] if c != self.agent else 0.0
        self._knockdowns_seen = len(ref.knockdowns)
        return reward

    def _observe(self) -> np.ndarray:
        m, me = self.match, self.me
        bal, st = me.balance, me.balance.status
        h = bal.heading()
        rot = np.array([[math.cos(h), math.sin(h)], [-math.sin(h), math.cos(h)]])   # S -> heading frame
        err = rot @ (st.capture_point - st.capture_ref) if st is not None else np.zeros(2)
        vel = rot @ st.com_velocity if st is not None else np.zeros(2)
        est = me.tracker.estimate
        if est is not None:
            opp = [*(est.head / 1.0), est.distance, math.sin(est.bearing), math.cos(est.bearing)]
        else:
            opp = [0.0, 0.0, 0.0, 2.0, 0.0, 1.0]
        now = m.time
        self._recent = [r for r in self._recent if now - r[0] < 0.6]
        recent = [sum(1 for r in self._recent if r[1] == kind and r[2] == target)
                  for kind in ("landed", "taken") for target in ("head", "body")]
        obs = [*(err * 10), *(vel * 2), (st.tilt_deg / 30.0) if st else 0.0, float(bal.stepping),
               float(me.moves.busy), float(me.moves.guard_name == "block"), float(me.fallen),
               *opp, float(m.fighters[self.opponent_name].fallen), max(0.0, (self.round_s - now) / self.round_s),
               *recent]
        return np.clip(np.asarray(obs, dtype=np.float32), -5.0, 5.0)
