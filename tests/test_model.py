"""Robot description, map and physics checks (need MuJoCo)."""
import math
import unittest

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class DescriptionTests(unittest.TestCase):
    def test_every_robot_compiles(self):
        from robotcode.config import ROBOTS_DIR, load_robot
        from robotcode.description import robot_xml
        for path in ROBOTS_DIR.glob("*.yaml"):
            spec = load_robot(path.stem)
            model = mujoco.MjModel.from_xml_string(robot_xml(spec, floating=False))
            self.assertGreater(model.nbody, 3, path.stem)
            self.assertEqual(model.actuator("left_wheel_motor").trntype, 0)

    def test_helper_rover_interfaces(self):
        from robotcode.config import load_robot
        from robotcode.kinematics import Kinematics
        kin = Kinematics(load_robot("helper_rover"))
        m = kin.model
        for name in ("lift_motor", "reach_motor", "side_motor", "gripper_motor"):
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name), 0, name)
        for name in ("imu_gyro", "finger_left_force", "bumper_force", "lift_encoder"):
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name), 0, name)
        # IK reaches a point on a table in front of the robot.
        target = np.array([0.55, 0.05, 0.60])
        q, err = kin.ik(target, kin.default_joints())
        self.assertLess(err, 1e-3)
        self.assertLess(np.linalg.norm(kin.site("grasp_site", q)[0] - target), 1e-3)
        xmin, xmax, ymin, ymax = kin.footprint()
        self.assertLess(xmax, 0.4)
        self.assertLess(ymax, 0.25)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class MapTests(unittest.TestCase):
    def test_prior_map(self):
        from robotcode.mapping import get_prior_map
        prior = get_prior_map("apartment")
        self.assertIn("kitchen_table", prior.surfaces)
        self.assertAlmostEqual(prior.surfaces["desk"].top_z, 0.75, places=3)
        self.assertEqual(prior.surfaces["coffee_table"].room, "living_room")
        self.assertTrue(prior.occupancy.value(4.0, 0.0) == 1)        # south wall
        self.assertTrue(prior.occupancy.value(2.0, 3.6) == 0)        # kitchen doorway
        self.assertTrue(prior.occupancy.value(2.2, 5.0) == 1)        # kitchen table footprint
        self.assertEqual(prior.room_of(6.0, 5.0), "office")
        # The LiDAR field is a signed distance: ~0 on a wall surface, negative inside, positive outside.
        f = prior.lidar_distance
        self.assertLess(abs(f.bilinear(np.array([0.05]), np.array([2.0]))[0]), 0.02)
        self.assertLess(f.bilinear(np.array([0.0]), np.array([2.0]))[0], 0.0)
        self.assertGreater(f.bilinear(np.array([0.5]), np.array([2.0]))[0], 0.3)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class PhysicsTests(unittest.TestCase):
    def _sim(self, pose, objects=()):
        from robotcode.runtime import Simulation
        return Simulation({"name": "test", "robot_start": {"pose": list(pose)}, "objects": list(objects)},
                          seed=0, verbose=False)

    def test_stands_level_and_drives_at_commanded_speed(self):
        sim = self._sim((1.0, 2.0, 0.0))
        sim.idle(1.0)
        self.assertLess(abs(sim.data.body("base").xmat.reshape(3, 3)[2, 0]), 0.01)   # no pitch
        start = sim.truth.robot_pose()

        def drive_forward(ctx):              # a minimal skill: 2 s at 0.3 m/s
            for _ in range(100):
                ctx.drive.command(0.3, 0.0, ctx.dt)
                yield
        sim.run("drive", drive_forward(sim.stack))
        self.assertAlmostEqual(np.linalg.norm(sim.data.qvel[:2]), 0.3, delta=0.02)
        self.assertGreater(sim.truth.robot_pose()[0] - start[0], 0.45)
        sim.close()

    def test_localization_converges_from_wrong_initial_guess(self):
        from robotcode.runtime import Simulation
        sim = Simulation({"name": "test", "robot_start": {"pose": [1.0, 2.0, 0.0], "jitter": [0.1, 0.1, 0.3]},
                          "objects": []}, seed=3, verbose=False)
        sim.idle(1.0)
        err = sim.truth.robot_pose() - sim.stack.pose
        self.assertLess(math.hypot(err[0], err[1]), 0.02)
        self.assertLess(abs(err[2]), 0.02)
        sim.close()

    def test_camera_detects_and_classifies_objects(self):
        objects = [{"name": "red_bottle", "type": "bottle", "color": "red", "support": "kitchen_table", "at": [2.0, 4.82]},
                   {"name": "red_cup", "type": "cup", "color": "red", "support": "kitchen_table", "at": [2.5, 4.85]}]
        sim = self._sim((2.2, 4.05, math.pi / 2), objects)
        sim.idle(1.0)
        from robotcode.skills import look
        sim.run("look", look(sim.stack))
        for name, otype in (("red_bottle", "bottle"), ("red_cup", "cup")):
            beliefs = sim.stack.world.query(otype, "red")
            self.assertEqual(len(beliefs), 1, name)
            err = np.linalg.norm(beliefs[0].position[:2] - sim.truth.object_position(name)[:2])
            self.assertLess(err, 0.02, name)
        sim.close()

    def test_friction_grasp_lifts_a_bottle(self):
        objects = [{"name": "red_bottle", "type": "bottle", "color": "red", "support": "coffee_table", "at": [4.0, 1.86]}]
        sim = self._sim((4.0, 2.45, -math.pi / 2), objects)
        sim.idle(1.0)
        from robotcode.skills import SKILLS, look
        sim.run("look", look(sim.stack))
        result = sim.run("pick", SKILLS["pick_object"].func(sim.stack, object="red bottle"), timeout=60)
        self.assertTrue(result.ok, result.message)
        self.assertGreater(sim.truth.object_position("red_bottle")[2], 0.30)   # lifted off the table
        self.assertTrue(sim.stack.gripper.check().holding)
        sim.close()


if __name__ == "__main__":
    unittest.main()
