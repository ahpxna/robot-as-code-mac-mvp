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
        # Seed 9 puts a red cup right next to the red bottle: needs an angled approach.
        r = self.run_task("fetch_red_bottle_kitchen_to_desk", seed=9)
        self.assertTrue(r.success, (r.message, r.checks))

    def test_unmapped_obstacle_is_avoided(self):
        r = self.run_task("blocked_corridor", seed=1)
        self.assertTrue(r.success, (r.message, r.checks))
        self.assertEqual(r.collisions, 0)

    def test_robot_without_arm_refuses_to_carry(self):
        r = self.run_task("scout_find_apple", instruction="Mang chai nước đến cho tao")
        self.assertFalse(r.robot_success)
        self.assertEqual(r.error, "unsupported")
        self.assertIn("manipulate", r.message)


if __name__ == "__main__":
    unittest.main()
