"""Episodes and benchmarks: judge the robot by outcomes, not by animation.

    python -m robotcode.evaluate --episodes 20            # every task in tasks/, 20 seeds each
    python -m robotcode.evaluate --task tasks/cup_to_drop_zone.yaml --episodes 50 --jobs 8

Each episode: build the scene from (task, seed) -> plan the instruction -> run
the mission -> the referee (GroundTruth) checks the task's success criteria.
The robot's own claim of success is recorded separately, so "the robot said it
succeeded but did not" (a false success) is measured, not hidden.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .config import ROOT, list_tasks, load_task
from .perception import DETECTORS


@dataclass
class EpisodeResult:
    task: str
    seed: int
    instruction: str
    plan: list[str]
    robot_success: bool
    success: bool
    checks: list[dict]
    sim_time: float
    wall_time: float
    collisions: int
    object_bumps: int
    path_length: float
    loc_error_mean: float
    loc_error_max: float
    energy_wh: float
    message: str
    steps: list[dict] = field(default_factory=list)
    error: str | None = None
    detector: str = ""
    detect_frames: int = 0
    detect_ms: float = 0.0              # mean wall time per detect() call


def judge(sim, task: dict) -> list[dict]:
    truth, prior = sim.truth, sim.scene.prior
    checks = []
    for criterion in task.get("success", []):
        (kind, arg), = criterion.items()
        ok, detail = False, ""
        if kind == "object_on":
            pos = truth.object_position(arg["object"])
            surface = prior.surfaces[arg["surface"]]
            on_top = abs(pos[2] - surface.top_z) < 0.25 and pos[2] > surface.top_z
            ok = bool(surface.contains(pos[0], pos[1]) and on_top)
            detail = f"{arg['object']} at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
        elif kind == "object_in_zone":
            pos = truth.object_position(arg["object"])
            place = prior.places[arg["place"]]
            dist = float(np.linalg.norm(pos[:2] - place.xy))
            ok = dist <= place.radius + 0.05 and pos[2] < 0.2
            detail = f"{arg['object']} {dist:.2f} m from {arg['place']} centre"
        elif kind == "object_upright":
            cos = truth.object_upright(arg)
            ok = cos > math.cos(math.radians(20))
            detail = f"tilt {math.degrees(math.acos(min(1.0, cos))):.0f} deg"
        elif kind == "robot_not_holding":
            forces = sim.hal.read_state().finger_forces
            ok = max(forces) < 0.5
            detail = f"finger forces {forces[0]:.1f}/{forces[1]:.1f} N"
        elif kind == "robot_in_room":
            pose = truth.robot_pose()
            room = prior.room_of(pose[0], pose[1])
            ok = room == arg
            detail = f"robot in {room}"
        elif kind == "robot_reports_success":
            ok = bool(sim.stack.result is not None and getattr(sim.stack.result, "success", False)) == bool(arg)
            detail = f"robot reported {'success' if ok == bool(arg) and arg else 'failure'}"
        elif kind == "max_collisions":
            ok = len(truth.collisions) <= int(arg)
            detail = f"{len(truth.collisions)} collisions"
        else:
            detail = f"unknown criterion {kind!r}"
        checks.append({"check": kind, "arg": arg, "ok": bool(ok), "detail": detail})
    return checks


def run_episode(task: dict, seed: int = 0, *, instruction: str | None = None, planner: str = "rules",
                verbose: bool = False, viewer: bool = False, realtime: bool = False, log=None,
                keep_open: bool = False, detector: str | None = None, setup=None):
    """Run one mission. Returns (EpisodeResult, Simulation). `setup(sim)` runs before the mission starts."""
    from .agent import MissionExecutor
    from .llm_planner import make_planner
    from .planner import UnsupportedTask
    from .runtime import Simulation

    wall = time.perf_counter()
    instruction = instruction or task.get("instruction", "")
    sim = Simulation(task, seed=seed, verbose=verbose, viewer=viewer, realtime=realtime, log=log, detector=detector)
    stack = sim.stack
    if setup is not None:
        setup(sim)
    plan_lines: list[str] = []
    try:
        plan_obj = make_planner(planner, sim.scene.world, sim.scene.catalog)
        plan = plan_obj.plan(instruction, stack.capabilities)
        plan_lines = [str(s) for s in plan.steps]
        if verbose:
            stack.log("PLAN: " + " -> ".join(plan_lines))
        sim.idle(0.5)                       # let localisation lock on before moving
        executor = MissionExecutor(stack, replanner=plan_obj if planner == "llm" else None)
        report = sim.run("mission", executor.run(plan), timeout=float(task.get("time_limit", 300)))
        if report is None:
            robot_ok, message, steps = False, "time limit reached (or viewer closed)", []
        else:
            robot_ok, message = report.success, report.message
            steps = [s.__dict__ for s in report.steps]
            if verbose:
                print(report.pretty())
        sim.idle(1.0)                       # let things settle before judging
        checks = judge(sim, task)
        error = None
    except UnsupportedTask as exc:
        robot_ok, message, steps, checks, error = False, str(exc), [], judge(sim, task), "unsupported"
    errors = sim.truth.loc_errors or [0.0]
    result = EpisodeResult(
        task=task.get("name", "?"), seed=seed, instruction=instruction, plan=plan_lines, robot_success=robot_ok,
        success=bool(checks) and all(c["ok"] for c in checks) if task.get("success") else robot_ok,
        checks=checks, sim_time=round(sim.time, 2), wall_time=round(time.perf_counter() - wall, 2),
        collisions=len(sim.truth.collisions), object_bumps=len(sim.truth.object_disturbances), path_length=round(sim.truth.path_length, 2),
        loc_error_mean=round(float(np.mean(errors)), 4), loc_error_max=round(float(np.max(errors)), 4),
        energy_wh=round(stack.state.battery.energy_used_wh, 3) if stack.state else 0.0, message=message, steps=steps, error=error)
    det = stack.detector
    if det is not None:
        result.detector = det.kind
        result.detect_frames = det.frames
        result.detect_ms = round(1000 * det.busy_s / max(det.frames, 1), 1)
    if not keep_open:
        sim.close()
    return result, sim


def _episode_worker(args):
    path, seed, planner, detector = args
    task = load_task(path)
    try:
        result, _ = run_episode(task, seed, planner=planner, detector=detector)
        return result.__dict__
    except Exception as exc:  # keep the benchmark going; record the crash
        return {"task": task.get("name"), "seed": seed, "success": False, "robot_success": False,
                "error": f"{type(exc).__name__}: {exc}", "sim_time": 0, "wall_time": 0, "collisions": 0, "object_bumps": 0,
                "path_length": 0, "loc_error_mean": 0, "loc_error_max": 0, "checks": [], "message": "crash"}


def summarize(results: list[dict]) -> list[dict]:
    rows = []
    for name in sorted({r["task"] for r in results}):
        rs = [r for r in results if r["task"] == name]
        ok = [r for r in rs if r["success"]]
        false_success = [r for r in rs if r["robot_success"] and not r["success"]]
        rows.append({
            "task": name, "episodes": len(rs), "success_rate": len(ok) / len(rs),
            "false_success": len(false_success), "crashes": sum(1 for r in rs if r.get("error") not in (None,
                                                                                                         "unsupported")),
            "mean_time_s": float(np.mean([r["sim_time"] for r in ok])) if ok else float("nan"),
            "collisions": int(sum(r["collisions"] for r in rs)),
            "object_bumps": int(sum(r.get("object_bumps", 0) for r in rs)),
            "mean_path_m": float(np.mean([r["path_length"] for r in rs])),
            "loc_err_max_m": float(np.max([r["loc_error_max"] for r in rs])),
            "detect_ms": float(np.mean([r["detect_ms"] for r in rs if r.get("detect_frames")] or [float("nan")])),
            "failed_seeds": [r["seed"] for r in rs if not r["success"]],
        })
    return rows


def print_table(rows: list[dict]) -> None:
    head = (f"{'task':36s} {'eps':>4s} {'success':>8s} {'false+':>6s} {'crash':>5s} {'t_ok[s]':>8s} {'coll':>5s} "
            f"{'bump':>5s} {'path[m]':>8s} {'loc_max':>8s} {'det[ms]':>8s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['task']:36s} {r['episodes']:4d} {r['success_rate']:8.0%} {r['false_success']:6d} {r['crashes']:5d} "
              f"{r['mean_time_s']:8.1f} {r['collisions']:5d} {r['object_bumps']:5d} {r['mean_path_m']:8.1f} {r['loc_err_max_m']:8.3f} "
              f"{r.get('detect_ms', float('nan')):8.0f}")
        if r["failed_seeds"]:
            print(f"{'':36s} failed seeds: {r['failed_seeds']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the robot on task files over many seeds")
    parser.add_argument("--task", action="append", help="task YAML (repeatable); default: all with success criteria")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--jobs", type=int, default=None,
                        help="worker processes (default: cores - 2; at most 4 with --detector learned: one model each)")
    parser.add_argument("--planner", choices=["rules", "llm"], default="rules")
    parser.add_argument("--detector", choices=DETECTORS, default=None,
                        help="object detector (default: perception.detector in the robot/catalog YAML, else color)")
    parser.add_argument("--min-success", type=float, default=None,
                        help="exit non-zero if any task's success rate is below this (CI gate)")
    args = parser.parse_args(argv)
    paths = args.task or [str(p) for p in list_tasks() if load_task(str(p)).get("success")]
    if args.jobs is None:
        args.jobs = max(1, (os.cpu_count() or 2) - 2)
        if args.detector == "learned":
            args.jobs = min(args.jobs, 4)
    jobs = [(p, args.seed + k, args.planner, args.detector) for p in paths for k in range(args.episodes)]
    start = time.perf_counter()
    print(f"Running {len(jobs)} episodes ({len(paths)} tasks x {args.episodes} seeds) on {args.jobs} workers, "
          f"detector: {args.detector or 'from config'}...")
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(_episode_worker, jobs))
    else:
        results = [_episode_worker(j) for j in jobs]
    rows = summarize(results)
    print_table(rows)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_{args.detector}" if args.detector else ""
    out = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}{suffix}.json"
    out.write_text(json.dumps({"detector": args.detector, "summary": rows, "episodes": results},
                              ensure_ascii=False, indent=1, default=str))
    print(f"\n{len(results)} episodes in {time.perf_counter() - start:.0f} s wall. Details: {out.relative_to(ROOT)}")
    if args.min_success is not None and any(r["success_rate"] < args.min_success for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
