"""Optional LLM task planner. The model writes a plan; it never touches actuators.

The LLM receives the skill registry (names, parameters, descriptions), the
world vocabulary (rooms, furniture, places, object types and colours) and the
robot's capabilities, and must answer with JSON:

    {"intent": "...", "steps": [{"skill": "find_object", "args": {"object": "red bottle", "room": "kitchen"}}, ...]}

`validate_plan` grounds every argument to the map/catalog and rejects unknown
skills, missing arguments, invented places and capability violations, so a
model mistake becomes an UnsupportedTask instead of robot motion. On a failed
mission the executor may call `replan()` with the execution report.

Requires `python -m pip install openai` and OPENAI_API_KEY (OPENAI_MODEL to pick a model).
"""
from __future__ import annotations

import json
import os

from .planner import Plan, RulePlanner, SkillCall, UnsupportedTask, Vocabulary, check_capabilities
from .skills import SKILLS

LOCATION_ARGS = {"target", "room", "surface"}


def tool_catalog(capabilities: set[str] | None = None) -> list[dict]:
    out = []
    for spec in SKILLS.values():
        if capabilities is not None and not spec.needs <= capabilities:
            continue
        out.append({"skill": spec.name, "description": spec.description, "args": spec.params,
                    "required": list(spec.required)})
    return out


def _ground_location(vocab: Vocabulary, text: str) -> str:
    names = {v for _, k, v in vocab.entries if k in ("room", "surface", "place", "user", "surface_kind")}
    if text in names:
        return text
    found = [m for m in vocab.mentions(text) if m.kind in ("room", "surface", "place", "user", "surface_kind")]
    if len(found) != 1:
        raise UnsupportedTask(f"plan refers to an unknown place {text!r}")
    return found[0].value


def _ground_object(vocab: Vocabulary, text: str) -> str:
    found = vocab.mentions(text)
    otype = next((m.value for m in found if m.kind == "type"), None)
    color = next((m.value for m in found if m.kind == "color"), None)
    if otype is None and color is None:
        raise UnsupportedTask(f"plan refers to an unknown object {text!r}")
    return " ".join(x for x in (color, otype) if x)


def validate_plan(data: dict, instruction: str, vocab: Vocabulary, capabilities: set[str] | None,
                  planner: str = "llm") -> Plan:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list) or not data["steps"]:
        raise UnsupportedTask(data.get("reason", "the planner declined the task") if isinstance(data, dict)
                              else "malformed plan")
    steps = []
    for raw in data["steps"]:
        name = raw.get("skill") if isinstance(raw, dict) else None
        spec = SKILLS.get(name)
        if spec is None:
            raise UnsupportedTask(f"plan uses an unknown skill {name!r}")
        args = raw.get("args") or {}
        unknown = set(args) - set(spec.params)
        missing = set(spec.required) - set(args)
        if unknown or missing:
            raise UnsupportedTask(f"bad arguments for {name}: unknown {sorted(unknown)}, missing {sorted(missing)}")
        grounded = {}
        for key, value in args.items():
            if value is None or value == "":
                continue
            value = str(value)
            if key in LOCATION_ARGS:
                value = _ground_location(vocab, value)
            elif key == "object":
                value = _ground_object(vocab, value)
            grounded[key] = value
        steps.append(SkillCall.make(name, **grounded))
    plan = Plan(instruction, steps, str(data.get("intent", "custom")), planner=planner)
    if capabilities is not None:
        check_capabilities(plan, capabilities)
    return plan


class LLMPlanner:
    name = "llm"

    def __init__(self, world: dict, catalog: dict, *, client=None, model: str | None = None):
        self.world, self.catalog = world, catalog
        self.vocab = Vocabulary(world, catalog)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Set OPENAI_API_KEY to use --planner llm")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the optional AI dependency: python -m pip install openai") from exc
            self._client = OpenAI(timeout=30.0)
        return self._client

    def _context(self, capabilities: set[str] | None) -> str:
        w = self.world
        vocab = {
            "rooms": {k: v.get("aliases", []) for k, v in w.get("rooms", {}).items()},
            "surfaces": {k: {"kind": v.get("kind"), "aliases": v.get("aliases", [])}
                         for k, v in w.get("surfaces", {}).items()},
            "places": {k: v.get("aliases", []) for k, v in w.get("places", {}).items()},
            "user": {"name": "user", "aliases": w.get("user", {}).get("aliases", []),
                     "note": "placing an object for the user puts it on their handover surface"},
            "object_types": {k: v.get("aliases", []) for k, v in self.catalog["types"].items()},
            "colors": {k: v.get("aliases", []) for k, v in self.catalog["colors"].items()},
        }
        return json.dumps({"skills": tool_catalog(capabilities), "world": vocab,
                           "robot_capabilities": sorted(capabilities or [])}, ensure_ascii=False)

    SYSTEM = (
        "You are the task planner of a household mobile manipulator. Turn the user's request (Vietnamese or "
        "English) into a short sequence of the given skills. Rules: use only the listed skills and argument "
        "names; places must be names from the world vocabulary (rooms, surfaces, places, or 'user'); objects "
        "are '<color> <type>' from the catalog (omit the color if the user did not give one); an object must be "
        "found with find_object before pick_object; place_object needs the object in hand; skills verify "
        "themselves, so do not add redundant checks. If the request cannot be done with these skills, answer "
        '{"steps": [], "reason": "<why>"}. Answer with JSON only: {"intent": "...", "steps": [{"skill": "...", '
        '"args": {...}}]}.')

    def _ask(self, messages: list[dict]) -> dict:
        client = self._get_client()
        response = client.chat.completions.create(model=self.model, messages=messages, temperature=0,
                                                  response_format={"type": "json_object"}, max_tokens=600)
        text = response.choices[0].message.content
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise UnsupportedTask(f"planner returned invalid JSON: {text!r}") from exc

    def plan(self, instruction: str, capabilities: set[str] | None = None) -> Plan:
        data = self._ask([{"role": "system", "content": self.SYSTEM},
                          {"role": "user", "content": f"CONTEXT: {self._context(capabilities)}\n\nREQUEST: {instruction}"}])
        return validate_plan(data, instruction, self.vocab, capabilities)

    def replan(self, plan: Plan, report, ctx) -> Plan:
        """Ask for a new plan given what happened (executed steps, failure message, current beliefs)."""
        held = ctx.world.objects[ctx.world.held].label if ctx.world.held else None
        status = {"executed": [{"skill": s.skill, "args": s.args, "ok": s.ok, "message": s.message}
                               for s in report.steps],
                  "holding": held, "known_objects": ctx.world.summary(),
                  "robot_room": ctx.prior.room_of(*ctx.pose[:2])}
        data = self._ask([
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"CONTEXT: {self._context(ctx.capabilities)}\n\nREQUEST: {plan.instruction}"},
            {"role": "assistant", "content": json.dumps(plan.to_json(), ensure_ascii=False)},
            {"role": "user", "content": "Execution failed. Plan the REMAINING steps only, from the current state: "
                                        + json.dumps(status, ensure_ascii=False)},
        ])
        return validate_plan(data, plan.instruction, self.vocab, ctx.capabilities)


def make_planner(name: str, world: dict, catalog: dict):
    if name == "llm":
        return LLMPlanner(world, catalog)
    return RulePlanner(world, catalog)
