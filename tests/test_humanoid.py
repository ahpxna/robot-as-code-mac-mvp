"""Humanoid boxer: kinematics, two-robot arena, balance, footwork, punches, perception, referee."""
import math
import unittest

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


def _arena(gap=0.64):
    from robotcode.arena import build_arena
    from robotcode.config import load_task
    match = dict(load_task("boxing/exhibition"))
    match["start"] = {"red": [-gap / 2, 0, 0], "blue": [gap / 2, 0, 180]}
    arena = build_arena(match, write=False)
    model, data = arena.compile()
    return arena, model, data


def _fighters(arena, model, data):
    from robotcode.balance import BalanceController
    from robotcode.boxing import MoveExecutor
    from robotcode.hal import SimRobot
    from robotcode.humanoid import initial
    out = {}
    for i, c in enumerate(arena.corners):
        robot = SimRobot(model, data, c.spec, seed=3 + i, prefix=c.prefix)
        bal = BalanceController(c.spec, 100)
        bal.set_stance(*initial(c.spec))
        out[c.name] = (c, robot, bal, MoveExecutor(c.spec, bal.body, lead=bal.lead))
    return out


def _run(model, data, fighters, seconds, on_tick=None, t0=0.0):
    for tick in range(int(seconds * 100)):
        t = t0 + tick * 0.01
        if on_tick:
            on_tick(t)
        for c, robot, bal, mv in fighters.values():
            s = robot.read_state()
            robot.command_joints(bal.update(s, mv.update(t, s)))
        for _ in range(5):
            mujoco.mj_step(model, data)
    return t0 + seconds


class LegKinematicsTests(unittest.TestCase):
    def test_ik_inverts_fk(self):
        from robotcode.config import load_robot
        from robotcode.humanoid import LEG_JOINTS, SIDES, leg_fk, leg_ik
        spec = load_robot("humanoid_boxer")
        rng = np.random.default_rng(0)
        for _ in range(100):
            for side in SIDES:
                q = {f"{side}_{j}": rng.uniform(-0.5, 0.5) for j in LEG_JOINTS}
                q[f"{side}_knee"] = rng.uniform(0.1, 1.5)
                ankle, rot = leg_fk(spec, side, q)
                sol, ok = leg_ik(spec, side, ankle, rot)
                self.assertTrue(ok)
                a2, r2 = leg_fk(spec, side, sol)
                self.assertLess(np.linalg.norm(a2 - ankle), 1e-9)
                self.assertLess(np.abs(r2 - rot).max(), 1e-9)

    @unittest.skipIf(mujoco is None, "MuJoCo not installed")
    def test_fk_matches_mujoco(self):
        from robotcode.config import load_robot
        from robotcode.description import robot_xml
        from robotcode.humanoid import stance_joints, leg_fk, SIDES
        spec = load_robot("humanoid_boxer")
        m = mujoco.MjModel.from_xml_string(robot_xml(spec, floating=False))
        d = mujoco.MjData(m)
        q = stance_joints(spec, "orthodox", "guard")
        for j, v in q.items():
            d.qpos[m.joint(j).qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        for side in SIDES:
            ankle, rot = leg_fk(spec, side, q)
            self.assertLess(np.linalg.norm(ankle - d.body(f"{side}_foot").xpos), 1e-9)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class ArenaTests(unittest.TestCase):
    def test_two_namespaced_robots_and_collision_groups(self):
        arena, m, d = _arena(1.2)
        self.assertEqual(m.nu, 42)
        for name in ("red_L_knee_motor", "blue_L_knee_motor"):
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name), 0)
        red, blue = m.geom("red_head"), m.geom("blue_L_glove")
        # Robots hit each other and the world, never themselves.
        self.assertTrue((red.contype[0] & blue.conaffinity[0]) or (blue.contype[0] & red.conaffinity[0]))
        self.assertFalse(red.contype[0] & m.geom("red_L_glove").conaffinity[0])

    def test_hal_reads_only_its_own_robot(self):
        from robotcode.hal import SimRobot
        arena, m, d = _arena(1.2)
        c = arena.corners[1]
        robot = SimRobot(m, d, c.spec, prefix=c.prefix)
        state = robot.read_state()
        self.assertEqual(len(state.joints), 21)
        self.assertIn("L_heel_force", state.touch)
        self.assertIsNotNone(state.orientation)
        # blue faces -x: its IMU yaw is ~180 deg
        w, x, y, z = state.orientation
        self.assertGreater(abs(math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))), 170)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class BalanceAndMovesTests(unittest.TestCase):
    def test_stand_step_and_punch(self):
        arena, m, d = _arena(1.6)
        f = _fighters(arena, m, d)
        c, robot, bal, mv = f["red"]
        t = _run(m, d, f, 1.5)
        start = d.body("red_base").xpos.copy()
        bal.move(forward=0.10)
        t = _run(m, d, f, 2.6, t0=t)
        self.assertGreater(d.body("red_base").xpos[0] - start[0], 0.06)       # it walked forward
        self.assertFalse(bal.stepping)
        target = {"head": np.array([0.65, 0.0, 0.55]), "body": np.array([0.65, 0.0, 0.28])}
        thrown = []
        t = _run(m, d, f, 1.5, on_tick=lambda now: thrown.append(mv.start("cross", target, now)) if not thrown else None,
                 t0=t)
        self.assertTrue(thrown and thrown[0])
        self.assertFalse(mv.busy)                                          # punched and back in guard
        for name in ("red", "blue"):
            self.assertGreater(d.body(f"{name}_base").xpos[2], 0.6, name)   # both still standing
            self.assertLess(f[name][2].status.tilt_deg, 10, name)

    def test_body_shot_lands_and_referee_scores_it(self):
        from robotcode.match import Referee
        arena, m, d = _arena(0.64)
        f = _fighters(arena, m, d)
        ref = Referee(m, d, arena)
        red, blue = f["red"], f["blue"]

        def throw(now):
            if now >= 1.5 and not red[3].busy and not red[3].thrown:
                b = d.body("red_base")
                rot, pos = b.xmat.reshape(3, 3), b.xpos
                tgt = {"head": rot.T @ (d.geom("blue_head").xpos - pos), "body": rot.T @ (d.geom("blue_torso").xpos - pos)}
                red[3].start("body", tgt, now)

        for tick in range(300):
            now = tick * 0.01
            throw(now)
            for c, robot, bal, mv in f.values():
                s = robot.read_state()
                robot.command_joints(bal.update(s, mv.update(now, s)))
            for _ in range(5):
                mujoco.mj_step(m, d)
                ref.after_step()
        body_hits = [e for e in ref.events if e.attacker == "red" and e.target == "body"]
        self.assertTrue(body_hits, ref.events)
        self.assertGreater(ref.score()["red"], 0)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class PerceptionTests(unittest.TestCase):
    def test_tracker_finds_opponent_head(self):
        from robotcode.boxing import OpponentTracker
        arena, m, d = _arena(1.0)
        f = _fighters(arena, m, d)
        c, robot, bal, mv = f["red"]
        _run(m, d, f, 0.6)
        tracker = OpponentTracker(c.spec, bal.body, arena.config["corners"]["blue"]["rgba"])
        est = tracker.update(robot.capture_rgbd("head_camera"), robot.read_state(), bal)
        b = d.body("red_base")
        truth = b.xmat.reshape(3, 3).T @ (d.geom("blue_head").xpos - b.xpos)
        self.assertLess(np.linalg.norm(est.head - truth), 0.05)
        self.assertAlmostEqual(est.distance, float(np.linalg.norm(truth[:2])), delta=0.05)
        robot.close()


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class RLEnvTests(unittest.TestCase):
    def test_env_checker(self):
        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            self.skipTest("gymnasium not installed")
        from robotcode.rl.boxing_env import BoxingEnv
        env = BoxingEnv(round_s=3.0)
        check_env(env, skip_render_check=True)
        env.close()


if __name__ == "__main__":
    unittest.main()
