"""Detector selection and the learned (open-vocabulary) detector.

The learned detector's 2D model is optional (pip install -e '.[vision]'): tests that need
torch/transformers and locally cached OWLv2 weights skip cleanly without them. The 3D part
(box -> depth points -> geometry) is tested with a stand-in model that needs neither.
"""
import importlib.util
import math
import unittest
from unittest import mock

import numpy as np

from robotcode.config import ConfigError, load_catalog
from robotcode.math_utils import to_base
from robotcode.perception import DETECTORS, detector_kind
from robotcode.perception_learned import DEFAULTS, build_queries, nms, require_vision

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


def _vision_ready() -> bool:
    """torch + transformers installed and the default weights already cached (tests never download)."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(DEFAULTS["model"], "config.json"), str)


class SelectionTests(unittest.TestCase):
    def test_precedence(self):
        catalog = {"perception": {"detector": "learned"}}
        self.assertEqual(detector_kind({}, {}), "color")
        self.assertEqual(detector_kind({}, catalog), "learned")
        self.assertEqual(detector_kind({"perception": {"detector": "color"}}, catalog), "color")
        self.assertEqual(detector_kind({"perception": {"detector": "color"}}, catalog, "learned"), "learned")
        self.assertEqual(set(DETECTORS), {"color", "learned"})

    def test_unknown_detector_is_a_config_error(self):
        with self.assertRaises(ConfigError):
            detector_kind({}, {}, "yolo9000")

    def test_shipped_default_is_color(self):
        self.assertEqual(detector_kind({}, load_catalog()), "color")

    def test_missing_vision_extra_names_the_install_command(self):
        with mock.patch.dict("sys.modules", {"torch": None}):
            with self.assertRaises(ImportError) as cm:
                require_vision()
        self.assertIn(".[vision]", str(cm.exception))


class QueryTests(unittest.TestCase):
    def test_queries_cover_every_colour_and_type(self):
        catalog = load_catalog()
        texts, keys = build_queries(catalog, "a photo of a {color} {type}")
        self.assertIn("a photo of a red bottle", texts)
        self.assertEqual({k for k in keys}, {(c, t) for c in catalog["colors"] for t in catalog["types"]})

    def test_prompt_variants(self):
        catalog = {"colors": {"red": {}}, "types": {"cup": {"prompts": ["cup", "mug"]}}}
        texts, keys = build_queries(catalog, "{color} {type}")
        self.assertEqual(texts, ["red cup", "red mug"])
        self.assertEqual(keys, [("red", "cup"), ("red", "cup")])

    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=float)
        self.assertEqual(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.3), [0, 2])


class GroundTruthBoxes:
    """Stand-in for OWLv2: boxes from the true object positions, scores chosen by the test."""

    def __init__(self, sim, labels: dict[str, tuple[str, str]], extra_boxes=()):
        self.sim, self.labels, self.extra = sim, labels, list(extra_boxes)
        self.queries = None

    def set_queries(self, texts):
        self.queries = texts

    def __call__(self, rgb):
        sim = self.sim
        cam = sim.scene.robot_spec["sensors"]["head_camera"]
        h, w = rgb.shape[:2]
        f = (h / 2) / math.tan(math.radians(cam["fovy_deg"]) / 2)
        cam_pos, cam_rot = sim.stack.kin.camera("head_camera", sim.stack.state.joints)
        pose = sim.truth.robot_pose()
        catalog = sim.scene.catalog
        _, keys = build_queries(catalog, DEFAULTS["template"])
        boxes, scores = [], []
        for o in sim.scene.objects:
            t = catalog["types"][o.type]
            r = t["radius"]
            half_h = r if t["shape"] == "sphere" else t["height"] / 2
            c = sim.truth.object_position(o.name)
            uv = []
            for dx in (-r, r):
                for dy in (-r, r):
                    for dz in (-half_h, half_h):
                        p = np.array([*to_base(pose, [[c[0] + dx, c[1] + dy]])[0], c[2] + dz])
                        pc = cam_rot.T @ (p - cam_pos)
                        uv.append((w / 2 + f * pc[0] / -pc[2], h / 2 - f * pc[1] / -pc[2]))
            uv = np.array(uv)
            boxes.append([*uv.min(axis=0), *uv.max(axis=0)])
            row = np.full(len(keys), 0.002)
            row[keys.index(self.labels[o.name])] = 0.6
            scores.append(row)
        for box in self.extra:
            boxes.append(box)
            scores.append(np.full(len(keys), 0.002))
            scores[-1][0] = 0.5
        return np.array(boxes, dtype=float), np.array(scores)


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
class LiftingTests(unittest.TestCase):
    """Boxes -> 3D objects, without the neural network."""

    objects = [{"name": "red_bottle", "type": "bottle", "color": "red", "support": "kitchen_table", "at": [2.0, 4.82]},
               {"name": "blue_cup", "type": "cup", "color": "blue", "support": "kitchen_table", "at": [2.5, 4.85]}]

    def _detect(self, labels, extra_boxes=()):
        from robotcode.perception_learned import LearnedObjectDetector
        from robotcode.runtime import Simulation
        sim = Simulation({"name": "test", "robot_start": {"pose": [2.2, 4.05, math.pi / 2]}, "objects": self.objects},
                         seed=0, verbose=False)
        sim.idle(1.0)
        det = LearnedObjectDetector(sim.stack, model=GroundTruthBoxes(sim, labels, extra_boxes))
        frame = sim.hal.capture_rgbd()
        out = det.detect(frame, sim.stack.pose.copy(), sim.stack.state.joints)
        truth = {o.name: sim.truth.object_position(o.name) for o in sim.scene.objects}
        sim.close()
        return out, truth, det

    def test_boxes_become_3d_objects(self):
        out, truth, _ = self._detect({"red_bottle": ("red", "bottle"), "blue_cup": ("blue", "cup")})
        self.assertEqual(sorted(d.label for d in out), ["blue cup", "red bottle"])
        for d in out:
            name = d.label.replace(" ", "_")
            self.assertLess(np.linalg.norm(d.position - truth[name]), 0.02, name)
            self.assertEqual(d.support, "kitchen_table")

    def test_metric_size_overrides_a_wrong_model_type(self):
        # The model calls the bottle a cup: 0.20 m of measured height says otherwise.
        out, _, _ = self._detect({"red_bottle": ("red", "cup"), "blue_cup": ("blue", "cup")})
        self.assertEqual(sorted(d.label for d in out), ["blue cup", "red bottle"])

    def test_box_without_an_object_is_rejected(self):
        out, _, det = self._detect({"red_bottle": ("red", "bottle"), "blue_cup": ("blue", "cup")},
                                   extra_boxes=[[20, 200, 60, 236]])
        self.assertEqual(len(out), 2)
        self.assertIn("rejected", det.last_boxes[-1])


@unittest.skipIf(mujoco is None, "MuJoCo not installed")
@unittest.skipUnless(_vision_ready(), "learned detector needs pip install -e '.[vision]' and cached OWLv2 weights")
class LearnedDetectorTests(unittest.TestCase):
    def test_owlv2_finds_and_locates_objects(self):
        from robotcode.runtime import Simulation
        from robotcode.skills import look
        sim = Simulation({"name": "test", "robot_start": {"pose": [2.2, 4.05, math.pi / 2]},
                          "objects": LiftingTests.objects}, seed=0, verbose=False, detector="learned")
        self.assertEqual(sim.stack.detector.kind, "learned")
        sim.idle(1.0)
        sim.run("look", look(sim.stack))
        for name, otype, color in (("red_bottle", "bottle", "red"), ("blue_cup", "cup", "blue")):
            beliefs = sim.stack.world.query(otype, color)
            self.assertEqual(len(beliefs), 1, name)
            err = np.linalg.norm(beliefs[0].position[:2] - sim.truth.object_position(name)[:2])
            self.assertLess(err, 0.02, name)
        sim.close()


if __name__ == "__main__":
    unittest.main()
