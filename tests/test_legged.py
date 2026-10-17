"""Legged robots: leg IK, postures, gait, folding into the "logo" posture (spider_bot)."""
import itertools
import math
import unittest

import numpy as np

from robotcode.config import load_robot
from robotcode.legged import build_legs, posture_joints

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


class LegKinematicsTests(unittest.TestCase):
    def test_ik_inverts_fk_over_the_workspace(self):
        leg = build_legs(load_robot("spider_bot"))[0]
        rng = np.random.default_rng(0)
        tested = 0
        while tested < 200:
            q = [rng.uniform(-0.8, 0.8), rng.uniform(-0.5, 1.2), rng.uniform(-2.6, -0.6)]
            foot = leg.fk(q)
            # Working range only: the foot outside the coxa (a foot folded back under the body is never used).
            if np.hypot(*(foot - leg.mount)[:2]) < leg.coxa + 0.02:
                continue
            tested += 1
            q2, ok = leg.ik(foot)
            self.assertTrue(ok)
            self.assertLess(np.linalg.norm(leg.fk(q2) - foot), 1e-9)

    def test_postures_are_reachable(self):
        spec = load_robot("spider_bot")
        for posture in ("stand", "fold"):
            self.assertEqual(len(posture_joints(spec, posture)), 24)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class LeggedPhysicsTests(unittest.TestCase):
    def test_ik_matches_the_generated_model(self):
        from robotcode.description import robot_xml
        spec = load_robot("spider_bot")
        m = mujoco.MjModel.from_xml_string(robot_xml(spec, floating=False))
        d = mujoco.MjData(m)
        for j, v in posture_joints(spec, "stand").items():
            d.qpos[m.joint(j).qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        for leg in build_legs(spec):
            q = [d.qpos[m.joint(j).qposadr[0]] for j in leg.joints]
            self.assertLess(np.linalg.norm(d.geom(f"{leg.name}_foot").xpos - leg.fk(q)), 1e-6)   # MJCF keeps 6 digits

    def test_fold_posture_and_path_are_free_of_self_collision(self):
        from robotcode.description import robot_xml
        spec = load_robot("spider_bot")
        m = mujoco.MjModel.from_xml_string(robot_xml(spec, floating=False))
        d = mujoco.MjData(m)
        geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]

        def leg_of(g):
            name = m.body(m.geom_bodyid[g]).name
            return "body" if name == "base" else name.split("_")[0]

        pairs = [(a, b) for a, b in itertools.combinations(geoms, 2) if leg_of(a) != leg_of(b)
                 and not ("body" in (leg_of(a), leg_of(b))
                          and m.body(m.geom_bodyid[b if leg_of(a) == "body" else a]).name.endswith("coxa_link"))]
        stand, fold = posture_joints(spec, "stand"), posture_joints(spec, "fold")
        for t in np.linspace(0, 1, 11):
            for j in stand:
                d.qpos[m.joint(j).qposadr[0]] = stand[j] + t * (fold[j] - stand[j])
            mujoco.mj_kinematics(m, d)
            clearance = min(mujoco.mj_geomDistance(m, d, a, b, 0.1, None) for a, b in pairs)
            self.assertGreater(clearance, 0.002, f"self-collision at t={t:.1f}")

    def test_walks_at_commanded_speed_and_folds(self):
        from robotcode.runtime import Simulation
        from robotcode.skills import SKILLS
        sim = Simulation({"name": "spider", "robot": "spider_bot", "robot_start": {"pose": [1.0, 2.0, 0.0]},
                          "objects": []}, seed=0, verbose=False)
        sim.idle(1.0)
        self.assertAlmostEqual(float(sim.data.body("base").xpos[2]), 0.075, delta=0.006)
        start = sim.truth.robot_pose()

        def walk(ctx):                       # 4 s at 0.1 m/s
            for _ in range(200):
                ctx.drive.command(0.1, 0.0, ctx.dt)
                yield
        sim.run("walk", walk(sim.stack))
        sim.idle(1.0)
        moved = np.hypot(*(sim.truth.robot_pose()[:2] - start[:2]))
        self.assertGreater(moved, 0.35)
        self.assertLess(moved, 0.5)
        self.assertLess(np.hypot(*(sim.truth.robot_pose()[:2] - sim.stack.pose[:2])), 0.01)   # localisation
        result = sim.run("fold", SKILLS["fold_robot"].func(sim.stack), timeout=20)
        self.assertTrue(result.ok, result.message)
        self.assertLess(float(sim.data.body("base").xpos[2]), 0.03)        # body resting on the floor
        result = sim.run("unfold", SKILLS["unfold_robot"].func(sim.stack), timeout=20)
        self.assertTrue(result.ok, result.message)
        self.assertGreater(float(sim.data.body("base").xpos[2]), 0.065)
        sim.close()


if __name__ == "__main__":
    unittest.main()
