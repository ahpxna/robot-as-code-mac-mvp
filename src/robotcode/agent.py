"""Mission executor: runs a plan step by step, verifies, recovers, reports.

Recovery policy (deterministic, so runs are reproducible):
* a step that reports it needs an object to be found first (lost / moved
  object) gets a find_object inserted, then is retried;
* navigation and manipulation steps are retried once;
* on an unrecoverable failure the robot stops in a safe state (arm stowed,
  still holding what it holds) and explains why.
An optional `replanner` (e.g. the LLM planner) is asked for a new plan instead.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .planner import Plan, SkillCall, UnsupportedTask, check_capabilities
from .skills import SKILLS, SkillResult

RETRYABLE = {"navigate_to", "pick_object", "place_object", "verify_object_at"}


@dataclass
class StepRecord:
    skill: str
    args: dict
    ok: bool
    message: str
    started: float
    ended: float
    attempt: int = 1


@dataclass
class MissionReport:
    instruction: str
    plan: list[str]
    planner: str
    success: bool
    message: str
    steps: list[StepRecord] = field(default_factory=list)
    started: float = 0.0
    ended: float = 0.0

    @property
    def duration(self) -> float:
        return self.ended - self.started

    def to_json(self) -> dict:
        data = asdict(self)
        data["duration"] = round(self.duration, 2)
        return data

    def pretty(self) -> str:
        lines = [f"MISSION {'SUCCEEDED' if self.success else 'FAILED'}: {self.message}",
                 f"  instruction: {self.instruction}", f"  planner: {self.planner}, sim time {self.duration:.1f} s"]
        for s in self.steps:
            mark = "ok " if s.ok else "ERR"
            lines.append(f"  [{mark}] t={s.started:6.1f}-{s.ended:6.1f}s {s.skill}"
                         f"({', '.join(f'{k}={v!r}' for k, v in s.args.items())}) -> {s.message}")
        return "\n".join(lines)


class MissionExecutor:
    def __init__(self, ctx, *, replanner=None, max_retries: int = 1):
        self.ctx = ctx
        self.replanner = replanner
        self.max_retries = max_retries

    def run(self, plan: Plan):
        """Generator returning a MissionReport."""
        ctx = self.ctx
        report = MissionReport(plan.instruction, [str(s) for s in plan.steps], plan.planner, False, "",
                               started=ctx.time)
        try:
            check_capabilities(plan, ctx.capabilities)
        except UnsupportedTask as exc:
            report.message, report.ended = str(exc), ctx.time
            return report
        queue = list(plan.steps)
        attempts: dict[int, int] = {}
        replans = 0
        i = 0
        last: SkillResult | None = None
        while i < len(queue):
            step = queue[i]
            attempt = attempts.get(i, 0) + 1
            attempts[i] = attempt
            ctx.log(f"STEP {i + 1}/{len(queue)}: {step}" + (f" (attempt {attempt})" if attempt > 1 else ""))
            started = ctx.time
            try:
                result = yield from SKILLS[step.skill].func(ctx, **step.kwargs)
            except TypeError as exc:
                result = SkillResult(False, f"bad arguments for {step.skill}: {exc}")
            if ctx.safety.estopped:
                result = SkillResult(False, f"emergency stop: {ctx.safety.reason}")
            report.steps.append(StepRecord(step.skill, step.kwargs, result.ok, result.message, started, ctx.time,
                                           attempt))
            ctx.log(("  -> ok: " if result.ok else "  -> FAILED: ") + result.message)
            last = result
            if result.ok:
                i += 1
                continue
            if ctx.safety.estopped:
                break
            if result.data.get("needs") == "find_object" and attempt <= self.max_retries:
                obj = step.kwargs.get("object")
                queue.insert(i, SkillCall.make("find_object", object=obj))
                attempts = {k + (1 if k >= i else 0): v for k, v in attempts.items()}
                continue
            if step.skill in RETRYABLE and attempt <= self.max_retries:
                continue
            if self.replanner is not None and replans < 1:
                replans += 1
                try:
                    new_plan = self.replanner.replan(plan, report, ctx)
                    check_capabilities(new_plan, ctx.capabilities)
                except (UnsupportedTask, RuntimeError) as exc:
                    ctx.log(f"replanning failed: {exc}")
                    break
                ctx.log("REPLANNED: " + "; ".join(str(s) for s in new_plan.steps))
                report.plan += ["--- replanned ---"] + [str(s) for s in new_plan.steps]
                queue, attempts, i = list(new_plan.steps), {}, 0
                continue
            break
        report.ended = ctx.time
        report.success = bool(report.steps) and i >= len(queue) and not ctx.safety.estopped
        if report.success:
            report.message = last.message if last else "done"
        else:
            report.message = last.message if last else "nothing was executed"
            if ctx.arm is not None and not ctx.safety.estopped:
                yield from ctx.arm.stow(holding=ctx.world.held is not None)
        return report


def report_json(report: MissionReport) -> str:
    return json.dumps(report.to_json(), ensure_ascii=False, indent=2)
