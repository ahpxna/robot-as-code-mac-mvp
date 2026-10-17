"""Perception benchmark: several detectors on the same camera frames, scored against simulator ground truth.

    python -m robotcode.perception_bench --episodes 5                         # color vs learned
    python -m robotcode.perception_bench --detectors color learned learned:image_size=640,dtype=float16

Episodes run as in `evaluate` (the configured detector drives the robot); every frame the robot
captures in `look()` is also given to each benchmarked detector. Per detector:

* precision / recall: a detection is correct when it lies within 10 cm of a ground-truth object
  of the same type and colour; recall counts objects that are in view (centre inside the image,
  not occluded, 0.3-3 m away, not in the gripper),
* position error of correct detections (3D, object centre): `est` uses the robot's own pose
  estimate (what the robot acts on), `true` re-expresses the detection with the true robot pose
  (perception only, localisation error removed),
* latency: wall time per detect() call (model load and first-call warm-up reported separately).

This module is a referee tool: it reads simulator ground truth, the robot stack never does.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from .config import ROOT, list_tasks, load_task
from .math_utils import to_base, to_map

MATCH_RADIUS = 0.10


def parse_detector(spec: str) -> tuple[str, dict]:
    """'learned:image_size=640,dtype=float16' -> ('learned', {'image_size': 640, 'dtype': 'float16'})."""
    kind, _, rest = spec.partition(":")
    overrides = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return kind, overrides


def build_detector(spec: str, ctx):
    kind, overrides = parse_detector(spec)
    if kind == "color":
        from .perception import ObjectDetector
        return ObjectDetector(ctx)
    if kind == "learned":
        from .perception_learned import LearnedObjectDetector
        return LearnedObjectDetector(ctx, **overrides)
    raise ValueError(f"unknown detector {spec!r}")


def visible_objects(sim, frame, joints, exclude) -> list[str]:
    """Ground-truth objects whose centre is in view and not hidden behind something."""
    cam_pos, cam_rot = sim.stack.kin.camera(frame.camera, joints)
    pose = sim.truth.robot_pose()
    h, w = frame.depth.shape
    out = []
    for o in sim.scene.objects:
        p = sim.truth.object_position(o.name)
        if exclude and any(np.linalg.norm(p[:2] - e[:2]) < 0.12 for e in exclude):
            continue
        pb = np.array([*to_base(pose, p[None, :2])[0], p[2]])
        pc = cam_rot.T @ (pb - cam_pos)
        depth = -pc[2]
        if not 0.3 < depth < 3.0:
            continue
        u = frame.cx + frame.fx * pc[0] / depth
        v = frame.cy - frame.fy * pc[1] / depth
        if not (4 <= u < w - 4 and 4 <= v < h - 4):
            continue
        seen = frame.depth[int(v), int(u)]
        if seen > 0 and seen < depth - 0.08:         # something in front of it
            continue
        out.append(o.name)
    return out


def score_frame(sim, detections, est_pose, visible) -> dict:
    """Match detections to ground truth (greedy, nearest first)."""
    true_pose = sim.truth.robot_pose()
    objs = {o.name: o for o in sim.scene.objects}
    truth = {name: sim.truth.object_position(name) for name in objs}
    pairs = []
    for i, d in enumerate(detections):
        p_true = np.array([*to_map(true_pose, to_base(est_pose, d.position[None, :2]))[0], d.position[2]])
        for name, g in truth.items():
            dist = float(np.linalg.norm(p_true[:2] - g[:2]))
            if dist < MATCH_RADIUS:
                pairs.append((dist, i, name, p_true))
    used_d, used_g = set(), set()
    out = {"tp": 0, "wrong_label": 0, "fp": 0, "err_est": [], "err_true": [], "found": set()}
    for dist, i, name, p_true in sorted(pairs, key=lambda x: x[0]):
        if i in used_d or name in used_g:
            continue
        used_d.add(i)
        used_g.add(name)
        d, o = detections[i], objs[name]
        if (d.type, d.color) == (o.type, o.color):
            out["tp"] += 1
            out["found"].add(name)
            out["err_est"].append(float(np.linalg.norm(d.position - truth[name])))
            out["err_true"].append(float(np.linalg.norm(p_true - truth[name])))
        else:
            out["wrong_label"] += 1
    out["fp"] = len(detections) - len(used_d)
    out["missed"] = sorted(set(visible) - out["found"])
    out["expected"] = len(visible)
    out["recalled"] = len(set(visible) & out["found"])
    return out


class Bench:
    def __init__(self, specs: list[str]):
        self.specs = specs
        self.stats = {s: {"frames": 0, "tp": 0, "wrong_label": 0, "fp": 0, "expected": 0, "recalled": 0,
                          "err_est": [], "err_true": [], "latency": [], "first_call": None, "build_s": []}
                      for s in specs}
        self.misses: list[dict] = []

    def setup(self, sim) -> None:
        detectors = {}
        for spec in self.specs:
            t0 = time.perf_counter()
            detectors[spec] = build_detector(spec, sim.stack)
            self.stats[spec]["build_s"].append(time.perf_counter() - t0)
        drive = sim.stack.detector
        inner = drive.detect

        def detect(frame, pose, joints, exclude=None):
            visible = visible_objects(sim, frame, joints, exclude)
            for spec, det in detectors.items():
                t0 = time.perf_counter()
                dets = det.detect(frame, pose, joints, exclude)
                dt = time.perf_counter() - t0
                st = self.stats[spec]
                if st["first_call"] is None:
                    st["first_call"] = dt
                else:
                    st["latency"].append(dt)
                r = score_frame(sim, dets, pose, visible)
                st["frames"] += 1
                for key in ("tp", "wrong_label", "fp", "expected", "recalled"):
                    st[key] += r[key]
                st["err_est"] += r["err_est"]
                st["err_true"] += r["err_true"]
                if r["missed"] or r["fp"] or r["wrong_label"]:
                    self.misses.append({"detector": spec, "task": sim.task.get("name"), "seed": sim.scene.seed,
                                        "t": round(sim.time, 2), "missed": r["missed"], "fp": r["fp"],
                                        "wrong_label": r["wrong_label"],
                                        "detections": [f"{d.label}@({d.position[0]:.2f},{d.position[1]:.2f})"
                                                       for d in dets]})
            return inner(frame, pose, joints, exclude)

        drive.detect = detect

    def summary(self) -> list[dict]:
        rows = []
        for spec, st in self.stats.items():
            n_det = st["tp"] + st["wrong_label"] + st["fp"]
            e_est, e_true = np.array(st["err_est"]) * 1000, np.array(st["err_true"]) * 1000
            lat = np.array(st["latency"]) * 1000
            pct = (lambda a, q: float(np.percentile(a, q)) if len(a) else math.nan)
            rows.append({
                "detector": spec, "frames": st["frames"], "detections": n_det, "correct": st["tp"],
                "wrong_label": st["wrong_label"], "false_pos": st["fp"],
                "precision": st["tp"] / n_det if n_det else math.nan,
                "recall": st["recalled"] / st["expected"] if st["expected"] else math.nan,
                "in_view": st["expected"],
                "err_est_mm": {"median": pct(e_est, 50), "p95": pct(e_est, 95), "max": pct(e_est, 100)},
                "err_true_mm": {"median": pct(e_true, 50), "p95": pct(e_true, 95), "max": pct(e_true, 100)},
                "latency_ms": {"mean": float(lat.mean()) if len(lat) else math.nan, "p50": pct(lat, 50),
                               "p95": pct(lat, 95)},
                "first_call_ms": 1000 * (st["first_call"] or math.nan),
                "build_s_first": st["build_s"][0] if st["build_s"] else math.nan,
            })
        return rows


def print_rows(rows: list[dict]) -> None:
    head = (f"{'detector':38s} {'frames':>6s} {'prec':>6s} {'recall':>6s} {'wrong':>5s} {'FP':>4s} "
            f"{'err med/p95/max [mm]':>21s} {'(true pose)':>17s} {'lat mean/p95 [ms]':>18s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        e, t, lat = r["err_est_mm"], r["err_true_mm"], r["latency_ms"]
        print(f"{r['detector']:38s} {r['frames']:6d} {r['precision']:6.1%} {r['recall']:6.1%} {r['wrong_label']:5d} "
              f"{r['false_pos']:4d} {e['median']:7.1f}/{e['p95']:5.1f}/{e['max']:5.1f} "
              f"{t['median']:5.1f}/{t['p95']:5.1f}/{t['max']:5.1f} {lat['mean']:9.0f}/{lat['p95']:6.0f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare object detectors on the same frames against ground truth")
    parser.add_argument("--task", action="append", help="task YAML (repeatable); default: all with success criteria")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--detectors", nargs="+", default=["color", "learned"],
                        help="detectors to compare; 'learned:key=value,...' overrides perception.learned settings")
    parser.add_argument("--drive", default=None, help="detector that drives the robot (default: from config)")
    args = parser.parse_args(argv)
    from .evaluate import run_episode

    paths = args.task or [str(p) for p in list_tasks() if load_task(str(p)).get("success")]
    bench = Bench(args.detectors)
    start = time.perf_counter()
    for path in paths:
        task = load_task(path)
        for k in range(args.episodes):
            result, _ = run_episode(task, args.seed + k, detector=args.drive, setup=bench.setup)
            print(f"{task['name']:36s} seed {args.seed + k:3d}: {'ok' if result.success else 'FAIL'}", flush=True)
    rows = bench.summary()
    print()
    print_rows(rows)
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"perception_bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"args": vars(args), "summary": rows, "misses": bench.misses}, ensure_ascii=False,
                              indent=1, default=str))
    print(f"\n{time.perf_counter() - start:.0f} s wall. Details: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
