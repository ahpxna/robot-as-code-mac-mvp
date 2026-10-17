"""Added sensors: 3D LiDAR, ToF ring, wrist camera (need MuJoCo)."""
import math
import unittest

import numpy as np

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class SensorTests(unittest.TestCase):
    def sim(self, pose, objects=(), obstacles=()):
        from robotcode.runtime import Simulation
        return Simulation({"name": "sensors", "robot_start": {"pose": list(pose)}, "objects": list(objects),
                           "obstacles": list(obstacles)}, seed=0, verbose=False)

    def test_3d_lidar_sees_low_box_and_never_itself(self):
        box = {"name": "box", "type": "box", "pos": [2.0, 2.0, 0.045], "size": [0.1, 0.1, 0.045]}
        sim = self.sim((1.0, 2.0, 0.0), obstacles=[box])
        sim.idle(0.5)
        pts = sim.stack.cloud.points
        self.assertGreater(len(pts), 1000)
        # No return from inside the robot's own footprint (self-hits are masked).
        inside = (np.abs(pts[:, 0]) < 0.3) & (np.abs(pts[:, 1]) < 0.25) & (pts[:, 2] > 0.02)
        self.assertFalse(np.any(inside))
        # The 9 cm box (below the 2D scan plane) is in the 3D costmap layer.
        cm = sim.stack.nav.costmap
        ix, iy = cm.grid.to_cell(2.0, 2.0)
        self.assertGreater(cm.seen3d[iy, ix], sim.time - 1.0)
        sim.close()

    def test_tof_ring_measures_close_obstacle(self):
        box = {"name": "box", "type": "box", "pos": [1.6, 2.0, 0.045], "size": [0.05, 0.2, 0.045]}
        sim = self.sim((1.0, 2.0, 0.0), obstacles=[box])
        sim.idle(0.3)
        prox = sim.stack.proximity
        front = prox.ranges[0]            # mount [0.25, 0, 0 deg]
        self.assertAlmostEqual(front, 1.55 - 1.25, delta=0.02)
        sim.close()

    def test_wrist_camera_corrects_a_wrong_belief(self):
        from robotcode.math_utils import to_base
        from robotcode.skills import look, wrist_locate
        objects = [{"name": "red_bottle", "type": "bottle", "color": "red", "support": "coffee_table",
                    "at": [4.0, 1.86]}]
        sim = self.sim((4.0, 2.45, -math.pi / 2), objects)
        sim.idle(1.0)
        st = sim.stack
        sim.run("look", look(st))
        obj = st.world.query("bottle", "red")[0]
        truth = sim.truth.object_position("red_bottle")[:2]
        xy = to_base(st.pose, obj.position[None, :2])[0]
        sim.run("pre", st.arm.move_to(np.array([xy[0] - 0.105, xy[1], 0.56]), order=("lift", "side")))
        obj.position[:2] = truth + np.array([0.02, 0.01])          # 22 mm wrong
        est = wrist_locate(st, obj, 0.033)
        self.assertIsNotNone(est)
        self.assertLess(np.linalg.norm(est - truth), 0.006)
        sim.close()


if __name__ == "__main__":
    unittest.main()
