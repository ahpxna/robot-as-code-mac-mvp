"""End-to-end missions judged by the simulator referee (slow: ~5 s each)."""
import unittest

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class MissionTests(unittest.TestCase):
    def run_task(self, name, seed=0, instruction=None):
        from robotcode.config import load_task
        from robotcode.evaluate import run_episode
        result, _ = run_episode(load_task(name), seed, instruction=instruction)
        return result

    def test_headline_vietnamese_delivery(self):
        r = self.run_task("deliver_bottle_to_user")
        self.assertTrue(r.success, (r.message, r.checks))
        self.assertTrue(r.robot_success)
        self.assertEqual(r.collisions, 0)

    def test_grounding_with_a_cluttered_table(self):
        # Seed 6: the red bottle sits where only an angled approach reaches it without the fingers
        # brushing the red cup.
        r = self.run_task("fetch_red_bottle_kitchen_to_desk", seed=6)
        self.assertTrue(r.success, (r.message, r.checks))

    def test_hidden_neighbour_is_handled_safely(self):
        # Seed 9: a red cup stands right behind the red bottle, hidden from the head camera until the
        # gripper is there. A known hard case: success is not guaranteed, but the robot must never
        # switch to the wrong object, move the cup off its table, or claim a success it did not achieve.
        r = self.run_task("fetch_red_bottle_kitchen_to_desk", seed=9)
        self.assertEqual(r.robot_success, r.success)
        cup = next(c for c in r.checks if c["check"] == "object_on" and c["arg"]["object"] == "red_cup")
        self.assertTrue(cup["ok"], cup)

    def test_unmapped_obstacle_is_avoided(self):
        r = self.run_task("blocked_corridor", seed=1)
        self.assertTrue(r.success, (r.message, r.checks))
        self.assertEqual(r.collisions, 0)

    def test_3d_sensors_avoid_clutter_the_2d_lidar_misses(self):
        with_3d = self.run_task("low_obstacles")
        self.assertTrue(with_3d.success, (with_3d.message, with_3d.checks))
        self.assertEqual(with_3d.collisions, 0)
        only_2d = self.run_task("tasks/ablation/low_obstacles_2d_only.yaml")
        self.assertGreater(only_2d.collisions, 0)          # drives under the console top

    def test_spider_walks_to_the_office_and_folds_into_its_logo(self):
        r = self.run_task("spider_fold_in_office")
        self.assertTrue(r.success, (r.message, r.checks))
        self.assertEqual(r.collisions, 0)

    def test_spider_finds_an_apple_under_the_table(self):
        r = self.run_task("spider_find_under_table")
        self.assertTrue(r.success, (r.message, r.checks))

    def test_robot_without_arm_refuses_to_carry(self):
        r = self.run_task("scout_find_apple", instruction="Mang chai nước đến cho tao")
        self.assertFalse(r.robot_success)
        self.assertEqual(r.error, "unsupported")
        self.assertIn("manipulate", r.message)


if __name__ == "__main__":
    unittest.main()
