"""Command line entry point.

    python -m robotcode.cli --dry-run --task "Mang chai tới đích"     # plan only, no physics
    python -m robotcode.cli --smoke --headless                         # scene loads and steps
    python -m robotcode.cli --headless --task "..."                    # full mission, no window
    mjpython -m robotcode.cli --task "..."                             # 3D viewer (macOS needs mjpython)
    mjpython -m robotcode.cli --interactive                            # viewer + type tasks one after another
    python -m robotcode.cli --scenario tasks/fetch_red_bottle_kitchen_to_desk.yaml --headless
    python -m robotcode.cli --headless --detector learned --task "..."  # OWLv2 detector (pip install -e '.[vision]')
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading

from .config import ConfigError, load_task


def _make_planner(args, task):
    from .config import load_catalog, load_world
    from .llm_planner import make_planner
    return make_planner(args.planner, load_world(task.get("world", "apartment")), load_catalog())


def _capabilities(task) -> set[str]:
    from .config import load_robot
    from .description import has_gripper
    spec = load_robot(task.get("robot", "helper_rover"))
    caps = {"navigate"}
    if "head_camera" in spec["sensors"]:
        caps.add("perceive")
    if spec.get("arm") and has_gripper(spec):
        caps.add("manipulate")
    return caps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Robot-as-Code: simulated household robot (MuJoCo)")
    parser.add_argument("--task", help="natural-language instruction (Vietnamese or English)")
    parser.add_argument("--scenario", default="home", help="task/scene file in tasks/ (default: home)")
    parser.add_argument("--seed", type=int, default=0, help="randomisation seed for the scene")
    parser.add_argument("--headless", action="store_true", help="no 3D viewer")
    parser.add_argument("--fast", action="store_true", help="viewer without real-time pacing")
    parser.add_argument("--dry-run", action="store_true", help="only plan the instruction; no physics")
    parser.add_argument("--smoke", action="store_true", help="build the scene and run 100 physics steps")
    parser.add_argument("--interactive", action="store_true", help="keep the robot running and read tasks from stdin")
    parser.add_argument("--planner", choices=["rules", "llm"], default="rules")
    parser.add_argument("--detector", choices=["color", "learned"], default=None,
                        help="object detector (default: perception.detector in the robot/catalog YAML, else color)")
    parser.add_argument("--report", help="write the episode report (JSON) to this file")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--camera", choices=["house", "follow"], default=None,
                        help="viewer camera: whole apartment, or follow the robot (default: follow small robots)")
    args = parser.parse_args(argv)

    try:
        task = load_task(args.scenario)
    except ConfigError as exc:
        parser.error(str(exc))
    if args.camera:
        task = {**task, "viewer_camera": args.camera}
    instruction = args.task or task.get("instruction")
    from .planner import UnsupportedTask

    if args.dry_run:
        if not instruction:
            parser.error("--dry-run needs --task")
        try:
            plan = _make_planner(args, task).plan(instruction, _capabilities(task))
        except (UnsupportedTask, RuntimeError) as exc:
            print(f"CANNOT PLAN: {exc}")
            return 1
        print("MISSION PLAN:")
        for i, step in enumerate(plan.steps, start=1):
            print(f"  {i}. {step}")
        print("DRY RUN ONLY: no simulation was executed")
        return 0

    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        parser.error(f"Missing dependency ({exc}). Run: python -m pip install -e .")

    if args.smoke:
        from .runtime import Simulation
        sim = Simulation(task, seed=args.seed, verbose=False, detector=args.detector)
        for _ in range(10):
            sim.step()
        print(f"SMOKE OK: scene '{task['name']}' built ({sim.model.nbody} bodies, {sim.model.nu} actuators, "
              f"{sim.model.nsensor} sensors), {10 * sim.n_sub} physics steps; this is not a task test")
        sim.close()
        return 0

    viewer = not args.headless
    if args.interactive:
        return interactive(task, args, viewer)

    if not instruction:
        parser.error("--task is required (or use --interactive)")
    from .evaluate import run_episode
    result, sim = run_episode(task, args.seed, instruction=instruction, planner=args.planner,
                              verbose=not args.quiet, viewer=viewer, realtime=viewer and not args.fast,
                              keep_open=True, detector=args.detector)
    sys.stdout.flush()
    print(f"\nRESULT: robot reports {'SUCCESS' if result.robot_success else 'FAILURE'}: {result.message}")
    if result.checks:
        print("REFEREE (simulator ground truth):")
        for c in result.checks:
            print(f"  [{'pass' if c['ok'] else 'FAIL'}] {c['check']} {c['arg']}: {c['detail']}")
    print(f"sim time {result.sim_time:.1f} s, wall {result.wall_time:.1f} s, path {result.path_length:.1f} m, "
          f"collisions {result.collisions}, localisation error max {result.loc_error_max * 100:.1f} cm, "
          f"energy {result.energy_wh:.2f} Wh")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(result.__dict__, fh, ensure_ascii=False, indent=2, default=str)
    if viewer and sim.viewer is not None and sim.viewer.is_running():
        print("Close the viewer window to exit.", flush=True)
        while sim.viewer.is_running():
            if not sim.step():
                break
    sim.close()
    return 0 if result.success else 1


def interactive(task: dict, args, viewer: bool) -> int:
    from .agent import MissionExecutor
    from .runtime import Simulation

    sim = Simulation({**task, "success": []}, seed=args.seed, verbose=True, viewer=viewer,
                     realtime=viewer and not args.fast, detector=args.detector)
    planner = _make_planner(args, task)
    inbox: queue.Queue[str | None] = queue.Queue()

    def reader():
        print("\nType a task (Vietnamese or English), 'stop' to interrupt, 'quit' to exit.")
        while True:
            try:
                line = input("task> ")
            except EOFError:
                inbox.put(None)
                return
            inbox.put(line.strip())

    threading.Thread(target=reader, daemon=True).start()
    from .planner import UnsupportedTask
    while True:
        try:
            line = inbox.get_nowait()
        except queue.Empty:
            line = ""
        if line is None or line.lower() in ("quit", "exit", "thoát"):
            break
        if line:
            if line.lower() in ("stop", "dừng", "dung", "dừng lại"):
                sim.stack.cancel()
                print("stopped")
            else:
                try:
                    plan = planner.plan(line, sim.stack.capabilities)
                    print("PLAN: " + " -> ".join(str(s) for s in plan.steps))
                    sim.stack.start("mission", MissionExecutor(sim.stack).run(plan))
                except (UnsupportedTask, RuntimeError) as exc:
                    print(f"Cannot do that: {exc}")
        was_busy = sim.stack.busy
        if not sim.step():
            break
        if was_busy and not sim.stack.busy and sim.stack.result is not None:
            print(sim.stack.result.pretty())
            print("task> ", end="", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
