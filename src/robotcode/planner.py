"""Natural-language task -> plan of skill calls (Vietnamese and English), without an LLM.

The rule planner grounds words onto the world's vocabulary (rooms, furniture,
places, the user, object types and colours from the catalog) and composes the
skills from robotcode.skills. It never invents destinations or skills; anything
it cannot ground raises UnsupportedTask. The optional LLM planner
(llm_planner.py) emits the same Plan structure and is validated the same way.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .config import load_catalog, load_world


class UnsupportedTask(ValueError):
    pass


@dataclass(frozen=True)
class SkillCall:
    skill: str
    args: tuple[tuple[str, str], ...] = ()

    @classmethod
    def make(cls, skill: str, **args) -> "SkillCall":
        return cls(skill, tuple((k, v) for k, v in args.items() if v is not None))

    @property
    def kwargs(self) -> dict[str, str]:
        return dict(self.args)

    def __str__(self) -> str:
        return f"{self.skill}(" + ", ".join(f"{k}={v!r}" for k, v in self.args) + ")"


@dataclass
class Plan:
    instruction: str
    steps: list[SkillCall]
    intent: str
    planner: str = "rules"
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"instruction": self.instruction, "intent": self.intent, "planner": self.planner,
                "steps": [{"skill": s.skill, "args": s.kwargs} for s in self.steps], "notes": self.notes}


# ---------------------------------------------------------------------------
# Text normalisation that keeps character positions aligned
# ---------------------------------------------------------------------------
def fold(text: str) -> str:
    return unicodedata.normalize("NFC", text.casefold())


def strip_accents(text: str) -> str:
    """Accent-free copy with the same length as `text` (which must be NFC)."""
    out = []
    for ch in text:
        if ch == "đ":
            out.append("d")
            continue
        base = "".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn")
        out.append(base[:1] or ch)
    return "".join(out)


def _marks(ch: str) -> set[str]:
    """Diacritics on one NFC character ('đ' counts as d + stroke)."""
    if ch == "đ":
        return {"stroke"}
    return {c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) == "Mn"}


def accents_agree(typed: str, alias: str) -> bool:
    """Every diacritic the user typed is also on the alias, character by character ('den', 'đen' and
    'phòng khach' agree with 'đen' / 'phòng khách'; 'đến' does not agree with 'đen'). Both strings are
    folded and equal after strip_accents."""
    return all(_marks(t) <= _marks(a) for t, a in zip(typed, alias))


@dataclass
class Mention:
    start: int
    end: int
    kind: str                   # room surface surface_kind place user type color pronoun
    value: str


VERBS = {
    "go": ["di", "di vao", "di den", "di toi", "di sang", "di qua", "chay den", "chay toi", "go", "go to", "head to",
           "navigate", "navigate to", "move to", "drive to", "ve"],
    "find": ["tim", "tim kiem", "kiem", "find", "look for", "search", "search for", "locate", "where is"],
    "pick": ["lay", "nhat", "cam", "gap", "nhac", "pick", "pick up", "grab", "take", "get"],
    "bring": ["mang", "dem", "dua", "chuyen", "bring", "carry", "deliver", "fetch"],
    "put": ["dat", "de", "bo", "put", "place", "leave", "drop", "set"],
    "stop": ["dung", "dung lai", "stop", "halt", "thoi"],
    "status": ["trang thai", "status", "pin", "battery", "bao cao", "report"],
    "look": ["nhin xung quanh", "nhin quanh", "quan sat", "look around", "what do you see", "thay gi", "co gi"],
}
DEST_MARKERS = {"den", "toi", "cho", "vao", "len", "sang", "ve", "to", "onto", "into", "for", "tren", "on", "o"}
SOURCE_MARKERS = {"trong", "tu", "o", "tren", "in", "from", "at", "inside", "on", "of"}
FILLER = {"cai", "chiec", "the", "a", "an", "my", "cua", "phong", "nha", "khu", "ngay", "mot", "nhung", "that",
          "this", "it", "no"}
PRONOUNS = ["no", "it", "them", "cai do", "that", "do"]
SURFACE_KINDS = {"table": ["table", "bàn", "cái bàn"], "counter": ["counter", "quầy"],
                 "shelf": ["shelf", "kệ", "giá"]}
CLAUSE_SPLIT = re.compile(r"[,;.!?]|\b(?:roi|sau do|xong|va sau do|then|and then|after that|and|va)\b")


class Vocabulary:
    def __init__(self, world: dict, catalog: dict):
        self.world, self.catalog = world, catalog
        entries: list[tuple[str, str, str]] = []
        for name, room in world.get("rooms", {}).items():
            entries += [(a, "room", name) for a in [name.replace("_", " ")] + room.get("aliases", [])]
        for name, s in world.get("surfaces", {}).items():
            entries += [(a, "surface", name) for a in [name.replace("_", " ")] + s.get("aliases", [])]
        for name, p in world.get("places", {}).items():
            entries += [(a, "place", name) for a in [name.replace("_", " ")] + p.get("aliases", [])]
        entries += [(a, "user", "user") for a in world.get("user", {}).get("aliases", [])]
        kinds = {s.get("kind", "table") for s in world.get("surfaces", {}).values()}
        for kind, aliases in SURFACE_KINDS.items():
            if kind in kinds:
                entries += [(a, "surface_kind", kind) for a in aliases]
        for name, t in catalog["types"].items():
            entries += [(a, "type", name) for a in [name] + t.get("aliases", [])]
        for name, c in catalog["colors"].items():
            entries += [(a, "color", name) for a in [name] + c.get("aliases", [])]
        entries += [(a, "pronoun", "it") for a in PRONOUNS]
        self.entries = sorted({(fold(a), k, v) for a, k, v in entries}, key=lambda e: -len(e[0]))

    def mentions(self, text: str) -> list[Mention]:
        """Longest match wins; on equal length an exact (accented) match beats an accent-free one.

        Accent-free matching lets people type without diacritics, but folds distinct words together
        ('đến'/'đen' -> 'den', 'tới'/'tôi' -> 'toi', 'tớ'/'to'). So an accent-free match only counts
        where the diacritics the user did type agree with the alias ('den' matches 'đen', 'đến' does
        not), an accent-free *user* pronoun only counts after 'cho' / 'for' / 'giúp'
        ('mang cho toi', 'bring it for me'), and an accent-free object word that is also a direction
        marker reads as the marker right before a destination ('mang den cho tao' = 'mang đến cho tao',
        'lon den den ban' = 'lon đen đến bàn')."""
        t1 = fold(text)
        t2 = strip_accents(t1)
        candidates = []
        for alias, kind, value in self.entries:
            for exact, pattern, hay in ((True, alias, t1), (False, strip_accents(alias), t2)):
                for m in re.finditer(rf"(?<!\w){re.escape(pattern)}(?!\w)", hay):
                    if not exact:
                        if not accents_agree(t1[m.start():m.end()], alias):
                            continue
                        if kind == "pronoun" and pattern == "do":
                            continue
                        if kind == "user" and t1[m.start():m.end()] != alias:
                            before = re.findall(r"\w+", t2[:m.start()])
                            if not before or before[-1] not in ("cho", "for", "giup", "voi"):
                                continue
                    candidates.append((-(m.end() - m.start()), 0 if exact else 1, m.start(), m.end(), kind, value))
        taken = [False] * len(t1)
        found: list[Mention] = []
        markers: list[Mention] = []
        for _, approx, start, end, kind, value in sorted(candidates):
            if any(taken[start:end]):
                continue
            for i in range(start, end):
                taken[i] = True
            found.append(Mention(start, end, kind, value))
            if approx and kind in ("type", "color") and t2[start:end] in DEST_MARKERS:
                markers.append(found[-1])
        places = {m.start for m in found if m.kind in ("room", "surface", "surface_kind", "place", "user")}
        for m in markers:
            nxt = re.match(r"\W*(\w+)", t2[m.end:])
            if nxt and (nxt.group(1) in ("cho", "for") or m.end + nxt.start(1) in places):
                found.remove(m)
        return sorted(found, key=lambda m: m.start)


def _verbs(text_ascii: str) -> list[tuple[int, str]]:
    out = []
    for verb, words in VERBS.items():
        for w in sorted(words, key=len, reverse=True):
            for m in re.finditer(rf"(?<!\w){re.escape(w)}(?!\w)", text_ascii):
                out.append((m.start(), verb))
    return sorted(out)


def _marker_before(text_ascii: str, start: int) -> str | None:
    words = re.findall(r"\w+", text_ascii[:start])
    for w in reversed(words[-3:]):
        if w in FILLER:
            continue
        return w
    return None


class RulePlanner:
    name = "rules"

    def __init__(self, world: dict | str = "apartment", catalog: dict | None = None):
        self.world = load_world(world) if isinstance(world, str) else world
        self.vocab = Vocabulary(self.world, catalog or load_catalog())

    def plan(self, instruction: str, capabilities: set[str] | None = None) -> Plan:
        text = fold(instruction)
        ascii_text = strip_accents(text)
        steps: list[SkillCall] = []
        intent = None
        current_obj: tuple[str | None, str | None] | None = None
        found = picked = False
        context_room = None
        notes = []

        clauses, last = [], 0
        for m in CLAUSE_SPLIT.finditer(ascii_text):
            clauses.append((last, m.start()))
            last = m.end()
        clauses.append((last, len(text)))
        mentions = self.vocab.mentions(instruction)

        for c0, c1 in clauses:
            ms = [m for m in mentions if c0 <= m.start < c1]
            verbs = [(p, v) for p, v in _verbs(ascii_text) if c0 <= p < c1]
            if not ms and not verbs:
                continue
            obj = self._object(ms)
            if obj is not None:
                if current_obj is not None and obj != current_obj:
                    found = picked = False
                current_obj = obj
            verb = verbs[0][1] if verbs else None
            verb_pos = verbs[0][0] if verbs else c0
            if any(v == "bring" for _, v in verbs):
                verb, verb_pos = "bring", next(p for p, v in verbs if v == "bring")
            elif any(v == "put" for _, v in verbs) and verb in (None, "pick", "go"):
                verb, verb_pos = "put", next(p for p, v in verbs if v == "put")
            locs = [m for m in ms if m.kind in ("room", "surface", "surface_kind", "place", "user")]
            if verb == "pick" and any(m.start > verb_pos and _marker_before(ascii_text, m.start) in
                                      ("to", "onto", "into", "den", "toi", "sang", "cho") for m in locs):
                verb = "bring"          # "take the apple to the table" / "lấy táo đến bàn"

            if verb == "stop":
                return Plan(instruction, [SkillCall.make("stop_robot")], "stop")
            if verb == "status":
                steps.append(SkillCall.make("get_robot_status"))
                intent = intent or "status"
                continue
            if verb == "look" and obj is None:
                steps.append(SkillCall.make("look_around"))
                intent = intent or "look"
                continue

            if verb in ("bring", "put"):
                dest, sources = None, []
                for m in locs:
                    marker = _marker_before(ascii_text, m.start)
                    after = m.start > verb_pos
                    if m.kind == "user" and after:
                        dest = m
                    elif after and marker in DEST_MARKERS and marker not in ("trong", "tu", "in", "from"):
                        dest = m if dest is None or dest.kind != "user" else dest
                    else:
                        sources.append(m)
                if dest is not None and dest in sources:
                    sources.remove(dest)
                if current_obj is None and not picked:
                    raise UnsupportedTask("Không rõ cần mang vật gì / which object should I carry?")
                if dest is None:
                    if verb == "bring":
                        dest = Mention(0, 0, "user", "user")     # "bring the bottle" = bring it to me
                    else:
                        raise UnsupportedTask("Không rõ đặt ở đâu / where should I put it?")
                if not picked:
                    room, surface = self._source(sources, context_room)
                    if not found:
                        steps.append(SkillCall.make("find_object", object=self._describe(current_obj),
                                                    room=room, surface=surface))
                    steps.append(SkillCall.make("pick_object", object=self._describe(current_obj)))
                    found = picked = True
                steps.append(SkillCall.make("place_object", target=dest.value))
                picked = False
                intent = "deliver"
                continue

            if verb == "pick" or (verb is None and obj is not None and found):
                if current_obj is None:
                    raise UnsupportedTask("Không rõ cần lấy vật gì / which object should I pick up?")
                if not found:
                    room, surface = self._source(locs, context_room)
                    steps.append(SkillCall.make("find_object", object=self._describe(current_obj), room=room,
                                                surface=surface))
                    found = True
                if not picked:
                    steps.append(SkillCall.make("pick_object", object=self._describe(current_obj)))
                    picked = True
                intent = intent or "pick"
                continue

            if verb == "find" or (verb == "look" and obj is not None) or (verb is None and obj is not None):
                if current_obj is None:
                    raise UnsupportedTask("Không rõ cần tìm gì / what should I look for?")
                room, surface = self._source(locs, context_room)
                steps.append(SkillCall.make("find_object", object=self._describe(current_obj), room=room,
                                            surface=surface))
                found = True
                intent = intent or "find"
                continue

            if verb == "go" or (verb is None and locs):
                if not locs:
                    continue
                target = locs[-1]
                steps.append(SkillCall.make("navigate_to", target=target.value))
                if target.kind == "room":
                    context_room = target.value
                intent = intent or "navigate"
                continue

        if not steps:
            raise UnsupportedTask(
                "Chưa hiểu nhiệm vụ. Ví dụ: 'Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao', "
                "'Find the red bottle in the kitchen and bring it to my desk', 'Đi vào bếp', 'Về trạm sạc'.")
        if intent == "find" and not any(s.skill == "say" for s in steps):
            notes.append("report where the object was found")
        plan = Plan(instruction, steps, intent or "custom", notes=notes)
        if capabilities is not None:
            check_capabilities(plan, capabilities)
        return plan

    @staticmethod
    def _object(ms: list[Mention]) -> tuple[str | None, str | None] | None:
        types = [m for m in ms if m.kind == "type"]
        colors = [m for m in ms if m.kind == "color"]
        if not types and not colors:
            return None
        otype = types[0].value if types else None
        color = None
        if colors:
            anchor = types[0].start if types else colors[0].start
            color = min(colors, key=lambda m: abs(m.start - anchor)).value
        return otype, color

    @staticmethod
    def _describe(obj) -> str:
        otype, color = obj
        return " ".join(x for x in (color, otype) if x)

    @staticmethod
    def _source(locs: list[Mention], context_room: str | None) -> tuple[str | None, str | None]:
        room = next((m.value for m in locs if m.kind == "room"), context_room)
        surface = next((m.value for m in locs if m.kind in ("surface", "surface_kind")), None)
        return room, surface


def check_capabilities(plan: Plan, capabilities: set[str]) -> None:
    from .skills import SKILLS

    for step in plan.steps:
        spec = SKILLS.get(step.skill)
        if spec is None:
            raise UnsupportedTask(f"unknown skill {step.skill!r}")
        missing = spec.needs - capabilities
        if missing:
            raise UnsupportedTask(f"this robot cannot {step.skill.replace('_', ' ')}: it lacks "
                                  + ", ".join(sorted(missing)) + " capability")


def plan_text(text: str, world: str = "apartment") -> Plan:
    return RulePlanner(world).plan(text)
