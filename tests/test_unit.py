"""Fast tests without physics: geometry helpers, language grounding, plan validation."""
import json
import math
import unittest

from robotcode.llm_planner import LLMPlanner, validate_plan
from robotcode.math_utils import to_base, to_map, wheel_targets, world_to_robot, wrap_angle
from robotcode.planner import RulePlanner, UnsupportedTask, Vocabulary, check_capabilities, strip_accents, fold


def steps(plan):
    return [(s.skill, s.kwargs) for s in plan.steps]


class MathTests(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi)

    def test_local_transform(self):
        x, y = world_to_robot(0, 1, math.pi / 2)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)

    def test_map_base_roundtrip(self):
        pose = (1.0, 2.0, 0.7)
        pts = [[0.3, -0.2], [1.0, 0.5]]
        back = to_base(pose, to_map(pose, pts))
        for a, b in zip(back.ravel(), [0.3, -0.2, 1.0, 0.5]):
            self.assertAlmostEqual(a, b)

    def test_wheels(self):
        left, right = wheel_targets(.3, 0)
        self.assertAlmostEqual(left, right)
        self.assertGreater(left, 0)
        left, right = wheel_targets(0, 1.0)
        self.assertAlmostEqual(left, -right)

    def test_accent_folding_keeps_positions(self):
        text = fold("Đi vào phòng khách, tìm chai nước")
        self.assertEqual(len(text), len(strip_accents(text)))
        self.assertEqual(strip_accents(text), "di vao phong khach, tim chai nuoc")


class PlannerTests(unittest.TestCase):
    planner = RulePlanner()

    def test_headline_vietnamese_task(self):
        plan = self.planner.plan("Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao.")
        self.assertEqual(steps(plan), [
            ("navigate_to", {"target": "living_room"}),
            ("find_object", {"object": "bottle", "room": "living_room", "surface": "table"}),
            ("pick_object", {"object": "bottle"}),
            ("place_object", {"target": "user"}),
        ])

    def test_same_task_without_diacritics(self):
        a = self.planner.plan("Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao.")
        b = self.planner.plan("di vao phong khach, tim chai nuoc tren ban, lay no roi mang den cho tao")
        self.assertEqual(steps(a), steps(b))

    def test_english_fetch_with_room_and_destination(self):
        plan = self.planner.plan("Find the red bottle in the kitchen and bring it to my desk.")
        self.assertEqual(steps(plan), [
            ("find_object", {"object": "red bottle", "room": "kitchen"}),
            ("pick_object", {"object": "red bottle"}),
            ("place_object", {"target": "desk"}),
        ])

    def test_source_and_destination_markers(self):
        plan = self.planner.plan("Mang cái cốc xanh dương từ bếp lên bàn làm việc")
        self.assertEqual(steps(plan)[0], ("find_object", {"object": "blue cup", "room": "kitchen"}))
        self.assertEqual(steps(plan)[-1], ("place_object", {"target": "desk"}))

    def test_legacy_command_goes_to_the_drop_zone(self):
        for text in ("Mang chai tới đích", "mang chai toi dich"):
            self.assertEqual(steps(self.planner.plan(text))[-1], ("place_object", {"target": "drop_zone"}))

    def test_pronoun_ambiguity_after_accent_folding(self):
        # 'tới' (to) and 'tôi' (me) both fold to 'toi'; 'táo' (apple) and 'tao' (me) to 'tao'.
        self.assertEqual(steps(self.planner.plan("mang chai den cho toi"))[-1], ("place_object", {"target": "user"}))
        plan = self.planner.plan("lay qua tao tren ban lam viec roi mang cho tao")
        self.assertEqual(steps(plan)[0], ("find_object", {"object": "apple", "surface": "desk"}))
        self.assertEqual(steps(plan)[-1], ("place_object", {"target": "user"}))

    def test_take_to_means_bring(self):
        self.assertEqual(steps(self.planner.plan("Take the apple to the coffee table"))[-1],
                         ("place_object", {"target": "coffee_table"}))

    def test_simple_commands(self):
        self.assertEqual(steps(self.planner.plan("Đi vào bếp")), [("navigate_to", {"target": "kitchen"})])
        self.assertEqual(steps(self.planner.plan("Về trạm sạc")), [("navigate_to", {"target": "dock"})])
        self.assertEqual(steps(self.planner.plan("Dừng lại!")), [("stop_robot", {})])
        self.assertEqual(steps(self.planner.plan("Pin còn bao nhiêu?")), [("get_robot_status", {})])

    def test_unknown_task_not_fabricated(self):
        with self.assertRaises(UnsupportedTask):
            self.planner.plan("Clean my entire house")

    def test_capabilities_are_enforced(self):
        plan = self.planner.plan("Mang chai nước đến cho tao")
        with self.assertRaises(UnsupportedTask):
            check_capabilities(plan, {"navigate", "perceive"})
        check_capabilities(plan, {"navigate", "perceive", "manipulate"})


class _FakeLLM:
    """Stands in for the OpenAI client: returns a canned JSON plan."""

    def __init__(self, reply: dict):
        self.reply = reply
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        content = json.dumps(self.reply)
        message = type("M", (), {"content": content})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


class LLMPlannerTests(unittest.TestCase):
    def setUp(self):
        base = RulePlanner()
        self.world, self.catalog = base.world, base.vocab.catalog
        self.vocab = Vocabulary(self.world, self.catalog)

    def test_valid_plan_is_grounded(self):
        reply = {"intent": "deliver", "steps": [
            {"skill": "find_object", "args": {"object": "chai đỏ", "room": "phòng bếp"}},
            {"skill": "pick_object", "args": {"object": "red bottle"}},
            {"skill": "place_object", "args": {"target": "bàn làm việc"}}]}
        plan = LLMPlanner(self.world, self.catalog, client=_FakeLLM(reply)).plan("x", {"navigate", "perceive",
                                                                                       "manipulate"})
        self.assertEqual(steps(plan), [("find_object", {"object": "red bottle", "room": "kitchen"}),
                                       ("pick_object", {"object": "red bottle"}),
                                       ("place_object", {"target": "desk"})])

    def test_invented_skill_or_place_is_rejected(self):
        for bad in ({"steps": [{"skill": "set_motor_torque", "args": {"value": 999}}]},
                    {"steps": [{"skill": "navigate_to", "args": {"target": "the moon"}}]},
                    {"steps": [{"skill": "pick_object", "args": {}}]},
                    {"steps": [], "reason": "cannot do that"}):
            with self.assertRaises(UnsupportedTask):
                validate_plan(bad, "x", self.vocab, None)


if __name__ == "__main__":
    unittest.main()
