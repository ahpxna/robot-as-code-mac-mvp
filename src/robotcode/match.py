"""Robot boxing: referee, scripted fighters and the match runner.

    python -m robotcode.match --headless                      # one match, scorecard in the terminal
    mjpython -m robotcode.match                               # 3D viewer with a scoreboard (macOS)
    python -m robotcode.match --rounds 1 --seed 3 --report build/match.json

The Referee is the only part that reads simulator state (contacts, body heights). Each
ScriptedBoxer is onboard software: it sees the opponent through its own head camera, keeps its
balance from its own IMU/encoders/foot pads, and commands only its own servos.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .arena import Arena, build_arena
from .balance import BalanceController
from .boxing import MoveExecutor, OpponentTracker
from .config import TASKS_DIR, ConfigError, load_task, load_yaml
from .hal import SimRobot
from .humanoid import initial

CONTROL_HZ = 100


# ---- referee ----------------------------------------------------------------------------
@dataclass
class HitEvent:
    t: float
    attacker: str
    defender: str
    target: str               # head / body / blocked
    force: float


@dataclass
class _Impact:
    attacker: str
    defender: str
    start: float
    last: float
    peak: float = 0.0
    target: str = "blocked"


class Referee:
    """Scores from simulator ground truth: glove contacts and who is on the canvas."""

    RANK = {"blocked": 0, "body": 1, "head": 2}

    def __init__(self, model, data, arena: Arena):
        import mujoco

        self._mj = mujoco
        self.model, self.data = model, data
        rules = arena.match.get("rules", {})
        self.hit_force = float(rules.get("hit_force", 60.0))
        self.count_s = float(rules.get("count_s", 10.0))
        self.points = {"head": 2, "body": 1, "knockdown": 5, **rules.get("points", {})}
        self.corners = [c.name for c in arena.corners]
        self.floor = model.geom("floor").id
        # geom id -> (corner, part)
        self.part: dict[int, tuple[str, str]] = {}
        for c in arena.corners:
            for g in range(model.ngeom):
                name = model.geom(g).name
                if not name.startswith(c.prefix):
                    continue
                n = name[len(c.prefix):]
                if n.endswith("_glove"):
                    part = "glove"
                elif n in ("head", "neck", "visor"):
                    part = "head"
                elif n in ("torso", "pelvis"):
                    part = "body"
                elif n.endswith("foot_geom"):
                    part = "foot"
                elif "arm" in n:
                    part = "arm"
                else:
                    part = "leg"
                self.part[g] = (c.name, part)
        self.base = {c.name: model.body(c.prefix + "base").id for c in arena.corners}
        self.events: list[HitEvent] = []
        self.knockdowns: list[tuple[float, str]] = []
        self.knockout: tuple[float, str] | None = None
        self._impacts: dict[int, _Impact] = {}
        self._down_since: dict[str, float | None] = {c: None for c in self.corners}
        self._up_since: dict[str, float | None] = {c: None for c in self.corners}
        self._force = np.zeros(6)
        self.break_s = float(rules.get("clinch_break_s", 0.4))
        self.clinch_gap = float(rules.get("clinch_gap", 0.50))
        self._clinch_since: float | None = None
        self.breaks: list[float] = []                   # times the referee called "break"

    def clinch_check(self) -> bool:
        """Fighters locked up (bodies or arms pressed together, or too close, for too long): the
        referee calls 'break'. Like a competition's referee box, the call reaches both robots."""
        if len(self.corners) != 2:
            return False
        a, b = (self.data.xpos[self.base[c]][:2] for c in self.corners)
        now = float(self.data.time)
        # Any contact between the robots that lasts much longer than a punch (< 0.1 s) is a clinch:
        # gloves pressed on a body, arms locked, or guards leaning on each other.
        tangled = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pa, pb = self.part.get(con.geom1), self.part.get(con.geom2)
            if pa and pb and pa[0] != pb[0]:
                tangled = True
                break
        if tangled or float(np.linalg.norm(a - b)) < self.clinch_gap:
            if self._clinch_since is None:
                self._clinch_since = now
            elif now - self._clinch_since > self.break_s and (not self.breaks or now - self.breaks[-1] > 2.0):
                self.breaks.append(round(now, 2))
                return True
        else:
            self._clinch_since = None
        return False

    def after_step(self) -> None:
        m, d = self.model, self.data
        now = float(d.time)
        grounded = {c: False for c in self.corners}
        for i in range(d.ncon):
            con = d.contact[i]
            a, b = self.part.get(con.geom1), self.part.get(con.geom2)
            for glove_geom, mine, other in ((con.geom1, a, b), (con.geom2, b, a)):
                if mine and mine[1] == "glove" and other and other[0] != mine[0]:
                    self._mj.mj_contactForce(m, d, i, self._force)
                    imp = self._impacts.get(glove_geom)
                    if imp is None or imp.defender != other[0]:
                        imp = self._impacts[glove_geom] = _Impact(mine[0], other[0], now, now)
                    imp.last = now
                    imp.peak = max(imp.peak, abs(float(self._force[0])))
                    target = other[1] if other[1] in ("head", "body") else "blocked"
                    if self.RANK[target] > self.RANK[imp.target]:
                        imp.target = target
            for geom, part, other in ((con.geom2, b, con.geom1), (con.geom1, a, con.geom2)):
                if other == self.floor and part and part[1] != "foot":
                    grounded[part[0]] = True
        for g, imp in list(self._impacts.items()):
            if now - imp.last > 0.08:                    # contact over: one punch, one event
                if imp.peak >= self.hit_force or (imp.target == "blocked" and imp.peak >= 0.5 * self.hit_force):
                    self.events.append(HitEvent(round(imp.start, 3), imp.attacker, imp.defender, imp.target,
                                                round(imp.peak, 1)))
                del self._impacts[g]
        for c in self.corners:
            low = float(d.xpos[self.base[c]][2]) < 0.40
            if grounded[c] or low:
                self._up_since[c] = None
                if self._down_since[c] is None:
                    self._down_since[c] = now
                    self.knockdowns.append((round(now, 2), c))
                elif self.knockout is None and now - self._down_since[c] >= self.count_s:
                    self.knockout = (round(now, 2), c)
            elif self._down_since[c] is not None:
                if self._up_since[c] is None:
                    self._up_since[c] = now
                elif now - self._up_since[c] > 0.5 and float(d.xpos[self.base[c]][2]) > 0.55:
                    self._down_since[c] = None             # beat the count

    def down(self, corner: str) -> bool:
        return self._down_since[corner] is not None

    def score(self, until: float | None = None, since: float = 0.0) -> dict[str, float]:
        pts = {c: 0.0 for c in self.corners}
        for e in self.events:
            if e.target in ("head", "body") and since <= e.t < (until if until is not None else math.inf):
                pts[e.attacker] += self.points[e.target]
        for t, c in self.knockdowns:
            if since <= t < (until if until is not None else math.inf):
                for other in self.corners:
                    if other != c:
                        pts[other] += self.points["knockdown"]
        return pts


# ---- scripted fighter -------------------------------------------------------------------
class ScriptedBoxer:
    """One fighter's onboard software: track the opponent, manage distance, throw combos."""

    CAMERA_HZ = 15

    def __init__(self, corner, robot: SimRobot, opponent_colour, style: dict, seed: int = 0, log=None):
        self.corner, self.robot, self.style = corner, robot, style
        self.name = corner.name
        spec = corner.spec
        self.balance = BalanceController(spec, CONTROL_HZ)
        self.balance.set_stance(*initial(spec))
        self.moves = MoveExecutor(spec, self.balance.body, lead=self.balance.lead)
        self.tracker = OpponentTracker(spec, self.balance.body, opponent_colour)
        self.rng = np.random.default_rng(seed)
        self.log = log or (lambda msg: None)
        self.combo: list[str] = []
        self.next_camera = 0.0
        self.next_decision = 0.0
        self.cooldown_until = 0.0
        self.block_until = 0.0
        self.fallen = False
        self.search = 1.0
        self._prev_gap: tuple[float, float] | None = None   # (time, nearest opponent point)
        self.incoming = False
        self.distance: float | None = None                 # filtered opponent distance / bearing
        self.bearing = 0.0
        self.break_until = 0.0

    def signal(self, name: str, now: float) -> None:
        """Referee signals (competition referee box)."""
        if name == "break":
            self.break_until = now + 2.0
            self.combo = []
            self.balance.stop_stepping()

    def tick(self, now: float) -> None:
        state = self.robot.read_state()
        if self.fallen:
            return
        if now >= self.next_camera:
            self.next_camera = now + 1.0 / self.CAMERA_HZ
            frame = self.robot.capture_rgbd("head_camera")
            est = self.tracker.update(frame, state, self.balance) if frame is not None else None
            self._watch(est, now)
        if now >= self.next_decision:
            self.next_decision = now + 0.1
            self._decide(now)
        if self.block_until and now >= self.block_until:
            self.moves.guard_name, self.block_until = "guard", 0.0
        upper = self.moves.update(now, state)
        targets = self.balance.update(state, upper)
        st = self.balance.status
        if st is not None and st.fallen:
            self.fallen = True
            self.robot.emergency_stop(True)              # fall protection: brakes hold the joints
            self.log(f"{self.name} is down")
            return
        self.robot.command_joints(targets)

    def _watch(self, est, now: float) -> None:
        """Filter distance/bearing; flag an incoming punch (opponent's nearest point closing fast)."""
        self.incoming = False
        if est is None or len(est.points) == 0:
            return
        if self.distance is None:
            self.distance, self.bearing = est.distance, est.bearing
        else:
            jump = float(np.clip(est.distance - self.distance, -0.08, 0.08))   # one frame cannot move much
            self.distance += 0.5 * jump
            self.bearing += 0.4 * (est.bearing - self.bearing)
        my_head = np.array([0.0, 0.0, 0.55])
        gap = float(np.min(np.linalg.norm(est.points - my_head, axis=1)))
        if self._prev_gap is not None:
            t0, g0 = self._prev_gap
            speed = (g0 - gap) / max(1e-3, now - t0)
            self.incoming = speed > 0.9 and gap < 0.45
        self._prev_gap = (now, gap)

    def _decide(self, now: float) -> None:
        bal, mv, style = self.balance, self.moves, self.style
        if not bal.ready:                                # IMU calibration: stand still first
            return
        est = self.tracker.estimate
        if est is None or now - est.stamp > 0.6:
            if not bal.stepping and not mv.busy:
                bal.move(turn=0.3 * self.search)
            return
        if est.nearest < 0.20 and not bal.stepping and not mv.busy:   # already body to body: get out first
            bal.move(forward=-0.10)
            return
        if self.incoming and not mv.busy:
            r = self.rng.random()
            if r < style.get("counter", 0.3) and not bal.stepping:
                self._throw("cross", est, now)
                return
            if r < style.get("counter", 0.3) + style.get("block", 0.3):
                mv.guard_name, self.block_until = "block", now + 0.5
                return
        if mv.busy:
            return
        if now < self.break_until:                       # referee broke a clinch: step back, no punches
            st = bal.status
            if not bal.stepping and (self.distance is None or self.distance < 0.75) and (
                    st is None or float(np.linalg.norm(st.capture_point - st.capture_ref)) < 0.05):
                bal.move(forward=-0.10)
            return
        if self.combo and not bal.stepping:
            self._throw(self.combo.pop(0), est, now)
            return
        dist, bearing = self.distance, self.bearing
        self.search = 1.0 if bearing >= 0 else -1.0
        want = style.get("range", 0.62)
        st = bal.status
        if bal.stepping or (st is not None and float(np.linalg.norm(st.capture_point - st.capture_ref)) > 0.05):
            return                                       # finish the footwork / regain balance first
        # Footwork only when clearly needed: every step is a moment on one foot.
        # Clinch range: guards lock and the robots shove each other. Up close the head can leave the
        # camera view, so the nearest visible point of the opponent is checked too.
        if dist < 0.53 or est.nearest < 0.26:
            bal.move(forward=-0.10)
            return
        if abs(bearing) > 0.45:
            bal.move(turn=float(np.clip(0.8 * bearing, -0.35, 0.35)))
            return
        if dist > want + 0.13:
            # Half the gap per move: the opponent is probably closing in too.
            bal.move(forward=float(np.clip(0.5 * (dist - want), 0.04, 0.10)))
            return
        if now >= self.cooldown_until and self.rng.random() < style.get("aggression", 1.0) * 0.1:
            combos = style.get("combos", [["jab"]])
            self.combo = list(combos[self.rng.integers(len(combos))])
            self._throw(self.combo.pop(0), est, now)
        elif self.rng.random() < 0.01:                    # now and then: change the angle
            bal.move(left=float(self.rng.choice([-0.08, 0.08])))

    def _throw(self, name: str, est, now: float) -> None:
        if (self.distance or est.distance) > self.style.get("range", 0.62) + 0.16:
            self.combo = []
            return
        guard_line = est.head[0] - 0.06
        obstacles = est.points[est.points[:, 0] < guard_line]
        if self.moves.start(name, {"head": est.head, "body": est.body}, now, obstacles):
            self.log(f"{self.name} {name}")
        if not self.combo:
            self.cooldown_until = now + 0.6 + 0.6 * self.rng.random()


# ---- match ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    winner: str | None
    method: str                              # KO / decision / draw
    score: dict[str, float]
    cards: list[dict[str, float]]            # per-round points
    events: list[dict] = field(default_factory=list)
    knockdowns: list = field(default_factory=list)
    punches_thrown: dict[str, int] = field(default_factory=dict)
    sim_time: float = 0.0
    wall_time: float = 0.0

    def summary(self) -> str:
        landed = {c: sum(1 for e in self.events if e["attacker"] == c and e["target"] != "blocked")
                  for c in self.score}
        blocked = {c: sum(1 for e in self.events if e["attacker"] == c and e["target"] == "blocked")
                   for c in self.score}
        lines = [f"RESULT: {self.winner.upper() + ' wins by ' + self.method if self.winner else 'DRAW'}"]
        for c in self.score:
            lines.append(f"  {c:5s} points {self.score[c]:5.1f}  thrown {self.punches_thrown.get(c, 0):3d}  "
                         f"landed {landed[c]:3d}  blocked {blocked[c]:3d}")
        for i, card in enumerate(self.cards, start=1):
            lines.append(f"  round {i}: " + "  ".join(f"{c} {v:.0f}" for c, v in card.items()))
        if self.knockdowns:
            lines.append("  knockdowns: " + ", ".join(f"{c} at {t:.1f}s" for t, c in self.knockdowns))
        lines.append(f"  sim {self.sim_time:.1f} s, wall {self.wall_time:.1f} s")
        return "\n".join(lines)


class Match:
    def __init__(self, match: dict, *, seed: int = 0, viewer: bool = False, realtime: bool = False,
                 verbose: bool = True, brains: dict | None = None):
        import mujoco

        self._mj = mujoco
        self.cfg = match
        self.verbose = verbose
        self.arena = build_arena(match)
        self.model, self.data = self.arena.compile()
        self.n_sub = max(1, round(1.0 / CONTROL_HZ / self.model.opt.timestep))
        self.referee = Referee(self.model, self.data, self.arena)
        styles = load_yaml(TASKS_DIR / "boxing" / "styles.yaml")
        colours = {c: v["rgba"] for c, v in self.arena.config["corners"].items()}
        self.fighters: dict[str, ScriptedBoxer] = {}
        names = [c.name for c in self.arena.corners]
        for i, corner in enumerate(self.arena.corners):
            other = names[1 - i]
            robot = SimRobot(self.model, self.data, corner.spec, seed=seed * 10 + i, prefix=corner.prefix)
            style = styles.get(corner.style, styles["pressure"])
            brain = (brains or {}).get(corner.name, ScriptedBoxer)      # e.g. a learned policy's adapter
            self.fighters[corner.name] = brain(corner, robot, colours[other], style, seed * 100 + i,
                                               log=self._log if verbose else None)
        rules = match.get("rules", {})
        self.rounds = int(rules.get("rounds", 3))
        self.round_s = float(rules.get("round_s", 45.0))
        self.realtime = realtime
        self.viewer = None
        if viewer:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            cam = self.viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, 0.0, 0.9]
            cam.distance, cam.azimuth, cam.elevation = 3.2, 90.0, -12.0
        self._last_events = 0
        self._video = None

    # ---- video (offscreen; works without a display, e.g. on a headless GPU box) -----------
    def record(self, path: str, fps: int = 30, size: tuple[int, int] = (960, 540)) -> None:
        import shutil
        import subprocess

        width, height = size
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        renderer = self._mj.Renderer(self.model, height, width)
        cam = self._mj.MjvCamera()
        cam.type = self._mj.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation = 3.0, 115.0, -14.0
        ffmpeg = shutil.which("ffmpeg")
        proc = None
        if ffmpeg:
            proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                     "-pix_fmt", "yuv420p", "-crf", "23", path], stdin=subprocess.PIPE)
        self._video = {"renderer": renderer, "cam": cam, "proc": proc, "fps": fps, "next": 0.0, "path": path,
                       "frames": []}

    def _video_frame(self) -> None:
        v = self._video
        if v is None or self.time < v["next"]:
            return
        v["next"] += 1.0 / v["fps"]
        names = list(self.fighters)
        mid = np.mean([self.data.body(f"{n}_base").xpos for n in names], axis=0)
        v["cam"].lookat[:] = [mid[0], mid[1], 0.85]
        v["renderer"].update_scene(self.data, v["cam"])
        img = v["renderer"].render()
        img = self._scoreboard(img)
        if v["proc"] is not None:
            v["proc"].stdin.write(img.tobytes())
        elif len(v["frames"]) < 2000:
            v["frames"].append(img[::2, ::2])

    def _scoreboard(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        if not hasattr(self, "_font"):
            try:
                self._font = ImageFont.load_default(size=26)
            except TypeError:                                  # Pillow < 10.1
                self._font = ImageFont.load_default()
        font = self._font
        score = self.referee.score()
        rnd = min(self.rounds, int(self.time // self.round_s) + 1)
        left = self.round_s - (self.time % self.round_s)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([12, 12, 360, 86], fill=(0, 0, 0))
        draw.text((24, 18), f"ROUND {rnd}/{self.rounds}   {left:4.1f} s", fill=(255, 255, 255), font=font)
        draw.text((24, 50), "   ".join(f"{c.upper()} {v:.0f}" for c, v in score.items()), fill=(255, 220, 80), font=font)
        recent = [e for e in self.referee.events if e.target != "blocked" and self.time - e.t < 1.0]
        if recent:
            e = recent[-1]
            draw.text((24, 96), f"{e.attacker.upper()} lands to the {e.target}  ({e.force:.0f} N)", fill=(255, 90, 90),
                      font=font)
        if any(self.time - t < 2.0 for t in self.referee.breaks):
            draw.text((24, 128), "referee: BREAK", fill=(255, 255, 255), font=font)
        return np.asarray(pil)

    def _close_video(self) -> None:
        v, self._video = self._video, None
        if v is None:
            return
        v["renderer"].close()
        if v["proc"] is not None:
            v["proc"].stdin.close()
            v["proc"].wait()
        elif v["frames"]:
            from PIL import Image
            frames = [Image.fromarray(f) for f in v["frames"][::2]]
            frames[0].save(v["path"], save_all=True, append_images=frames[1:], duration=int(2000 / v["fps"]), loop=0)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def _log(self, msg: str) -> None:
        print(f"[{self.time:6.2f}] {msg}", flush=True)

    def step(self) -> bool:
        now = self.time
        for f in self.fighters.values():
            f.tick(now)
        if self.referee.clinch_check():
            if self.verbose:
                self._log("referee: BREAK")
            for f in self.fighters.values():
                f.signal("break", now)
        for _ in range(self.n_sub):
            self._mj.mj_step(self.model, self.data)
            self.referee.after_step()
            for f in self.fighters.values():
                f.robot.after_physics_step(self.model.opt.timestep)
        self._video_frame()
        if self.verbose:
            for e in self.referee.events[self._last_events:]:
                if e.target != "blocked":
                    self._log(f"HIT {e.attacker} -> {e.defender} {e.target} ({e.force:.0f} N)")
            self._last_events = len(self.referee.events)
        if self.viewer is not None:
            return self._sync_viewer()
        return True

    def _sync_viewer(self) -> bool:
        if not self.viewer.is_running():
            return False
        now = time.perf_counter()
        if not hasattr(self, "_wall0"):
            self._wall0, self._sim0, self._last_sync = now, self.time, 0.0
        if now - self._last_sync > 1 / 30:
            with self.viewer.lock():
                mid = 0.5 * (self.data.body("red_base").xpos + self.data.body("blue_base").xpos) \
                    if "red" in self.fighters and "blue" in self.fighters else np.zeros(3)
                self.viewer.cam.lookat[:2] = mid[:2]
                score = self.referee.score()
                rnd = min(self.rounds, int(self.time // self.round_s) + 1)
                left = self.round_s - (self.time % self.round_s)
                text = "  ".join(f"{c.upper()} {v:.0f}" for c, v in score.items())
                self.viewer.set_texts([(None, self._mj.mjtGridPos.mjGRID_TOPLEFT,
                                        f"ROUND {rnd}/{self.rounds}   {left:4.1f} s", text)])
            self.viewer.sync()
            self._last_sync = now
        if self.realtime:
            lag = (self.time - self._sim0) - (time.perf_counter() - self._wall0)
            if lag > 0:
                time.sleep(min(lag, 0.05))
        return True

    def run(self) -> MatchResult:
        wall0 = time.perf_counter()
        total = self.rounds * self.round_s
        aborted = False
        while self.time < total and self.referee.knockout is None:
            if not self.step():
                aborted = True
                break
        ref = self.referee
        cards = [ref.score(since=r * self.round_s, until=(r + 1) * self.round_s) for r in range(self.rounds)]
        score = ref.score()
        if ref.knockout is not None:
            loser = ref.knockout[1]
            winner, method = next(c for c in self.fighters if c != loser), "KO"
        else:
            ranked = sorted(score.items(), key=lambda kv: -kv[1])
            winner = ranked[0][0] if ranked[0][1] > ranked[1][1] else None
            method = "decision" if winner else "draw"
            if aborted:
                method += " (stopped)"
        thrown = {c: len(f.moves.thrown) for c, f in self.fighters.items()}
        return MatchResult(winner, method, score, cards, [asdict(e) for e in ref.events], ref.knockdowns, thrown,
                           round(self.time, 2), round(time.perf_counter() - wall0, 1))

    def close(self) -> None:
        self._close_video()
        for f in self.fighters.values():
            f.robot.close()
        if self.viewer is not None:
            self.viewer.close()


def _bench_one(args) -> dict:
    cfg, seed = args
    m = Match(cfg, seed=seed, verbose=False)
    r = m.run()
    m.close()
    return {"seed": seed, "winner": r.winner, "method": r.method, "score": r.score, "sim_time": r.sim_time,
            "knockdowns": len(r.knockdowns), "thrown": sum(r.punches_thrown.values()),
            "landed": sum(1 for e in r.events if e["target"] != "blocked"),
            "blocked": sum(1 for e in r.events if e["target"] == "blocked")}


def bench(cfg: dict, seeds: list[int], workers: int = 6) -> list[dict]:
    """Run several headless matches in parallel (one process per match)."""
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(min(workers, len(seeds))) as pool:
        return pool.map(_bench_one, [(cfg, s) for s in seeds])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Robot boxing match (MuJoCo)")
    parser.add_argument("--match", default="boxing/exhibition", help="match file in tasks/ (default: boxing/exhibition)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, help="override the number of rounds")
    parser.add_argument("--round-s", type=float, help="override the round length (simulated seconds)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fast", action="store_true", help="viewer without real-time pacing")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", help="write the result (JSON) to this file")
    parser.add_argument("--bench", type=int, metavar="N", help="run N headless matches (seeds 1..N) and summarise")
    parser.add_argument("--video", help="record the match to this file (.mp4 with ffmpeg, else .gif); implies --headless")
    args = parser.parse_args(argv)
    try:
        cfg = load_task(args.match)
    except ConfigError as exc:
        parser.error(str(exc))
    rules = dict(cfg.get("rules", {}))
    if args.rounds:
        rules["rounds"] = args.rounds
    if args.round_s:
        rules["round_s"] = args.round_s
    cfg = {**cfg, "rules": rules}
    if args.bench:
        rows = bench(cfg, list(range(1, args.bench + 1)))
        for r in rows:
            print(f"seed {r['seed']:3d}: {(r['winner'] or 'draw'):5s} {r['method']:9s} {r['sim_time']:6.1f} s  "
                  f"knockdowns {r['knockdowns']}  thrown {r['thrown']:3d}  landed {r['landed']:3d}  blocked {r['blocked']:3d}")
        full = sum(1 for r in rows if r["method"] != "KO")
        print(f"{full}/{len(rows)} matches went the distance; knockdowns per match "
              f"{np.mean([r['knockdowns'] for r in rows]):.2f}; punches thrown {np.mean([r['thrown'] for r in rows]):.1f}, "
              f"landed {np.mean([r['landed'] for r in rows]):.1f} per match")
        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)
        return 0
    viewer = not (args.headless or args.video)
    match = Match(cfg, seed=args.seed, viewer=viewer, realtime=viewer and not args.fast, verbose=not args.quiet)
    if args.video:
        match.record(args.video)
    result = match.run()
    print(result.summary())
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2, default=str)
    if viewer and match.viewer is not None and match.viewer.is_running():
        print("Close the viewer window to exit.", flush=True)
        while match.viewer.is_running():
            match.viewer.sync()
            time.sleep(0.05)
    match.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
