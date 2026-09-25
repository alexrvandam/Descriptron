#!/usr/bin/env python3
"""
descriptron_joints.py — joint angles and rotational standardisation of articulated landmarks
=============================================================================================

Mandibles, antennae, legs and other articulated parts are photographed at different angles. That
pose variation is real data about the specimen's posture, not about its shape, and it swamps shape
differences in a Procrustes analysis. This program:

  angles       measures the angle at every joint (a pivot landmark between two others) per specimen —
               a character in its own right. Joints come from the COCO skeleton (every landmark with two
               or more connections gives one angle per pair of neighbours) or from --joint.
  standardise  rotates the distal part of each articulated structure about its pivot so that every
               specimen has the same joint angle (the sample's circular mean, or --target-angle), the
               equivalent of geomorph::fixed.angle (Adams 1999). The result is a NEW COCO keypoint file
               for the existing landmark GPA; the original file is not changed.

Joints: --joint PIVOT:A,B  (landmark numbers, 1-based; the angle is A-PIVOT-B).
Rotating part (standardise): --rotate PIVOT:A,B:L1,L2,...  rotates landmarks L1,L2,... about PIVOT so
that angle A-PIVOT-B becomes the target; without the landmark list, the landmarks on B's side of the
skeleton (the connected part containing B once PIVOT is removed) are rotated.

    python descriptron_joints.py angles legs.json --out-dir joints/
    python descriptron_joints.py angles legs.json --joint 3:2,4 --joint 4:3,5 --out-dir joints/
    python descriptron_joints.py standardise legs.json --rotate 3:2,4 --out legs_standardised.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def load_sets(coco: dict, category: Optional[str]):
    """-> [(annotation, image, category)] for keypoint annotations, and the (names, skeleton) of the category."""
    cats = {c["id"]: c for c in coco["categories"]}
    ims = {i["id"]: i for i in coco["images"]}
    out = []
    for a in coco["annotations"]:
        if "keypoints" not in a:
            continue
        c = cats.get(a["category_id"], {})
        if category and c.get("name") != category:
            continue
        if not category and not c.get("keypoints") and not c.get("skeleton") and c.get("name") not in (None, "keypoints"):
            continue
        out.append((a, ims.get(a["image_id"], {}), c))
    if not out:
        sys.exit("no keypoint annotations found" + (f" in category {category}" if category else ""))
    return out


def points(ann) -> Dict[int, np.ndarray]:
    kp = ann["keypoints"]
    order = ann.get("point_order") or list(range(1, len(kp) // 3 + 1))
    return {int(o): np.array(kp[3 * j:3 * j + 2], float) for j, o in enumerate(order) if kp[3 * j + 2] > 0}


def joints_from_skeleton(skel: List[List[int]]) -> List[Tuple[int, int, int]]:
    nb = defaultdict(set)
    for a, b in skel:
        nb[a].add(b); nb[b].add(a)
    out = []
    for p in sorted(nb):
        ns = sorted(nb[p])
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                out.append((p, ns[i], ns[j]))
    return out


def angle(pts: Dict[int, np.ndarray], p: int, a: int, b: int, signed: bool = False) -> Optional[float]:
    if p not in pts or a not in pts or b not in pts:
        return None
    u, v = pts[a] - pts[p], pts[b] - pts[p]
    if not np.any(u) or not np.any(v):
        return None
    ang = math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v))           # signed, image axes
    return math.degrees(ang) if signed else abs(math.degrees(ang))


def parse_joint(s: str) -> Tuple[int, int, int]:
    p, _, ab = s.partition(":")
    a, b = ab.split(",")[:2]
    return int(p), int(a), int(b)


def cmd_angles(args):
    coco = json.load(open(args.input))
    sets = load_sets(coco, args.category)
    names = next((c.get("keypoints") for _, _, c in sets if c.get("keypoints")), None)
    skel = next((c.get("skeleton") for _, _, c in sets if c.get("skeleton")), None)
    joints = [parse_joint(j) for j in args.joint] if args.joint else joints_from_skeleton(skel or [])
    if not joints:
        sys.exit("no joints: give --joint PIVOT:A,B or use keypoints with a COCO skeleton")
    lab = (lambda k: names[k - 1] if names and 0 < k <= len(names) else str(k))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cols = [f"angle_{lab(a)}-{lab(p)}-{lab(b)}_deg" for p, a, b in joints]
    rows = []
    for ann, im, c in sets:
        pts = points(ann)
        r = {"image": im.get("file_name", ann["image_id"]), "annotation_id": ann.get("id"), "category": c.get("name")}
        for (p, a, b), col in zip(joints, cols):
            v = angle(pts, p, a, b)
            r[col] = "" if v is None else round(v, 3)
        rows.append(r)
    with open(out / "joint_angles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = []
    for col in cols:
        v = np.array([r[col] for r in rows if r[col] != ""], float)
        if len(v):
            summ.append(f"  {col}: n={len(v)} mean={v.mean():.1f} sd={v.std(ddof=1) if len(v) > 1 else 0:.1f} "
                        f"range {v.min():.1f}-{v.max():.1f}")
    print(f"{len(rows)} specimens, {len(joints)} joints -> {out/'joint_angles.csv'}\n" + "\n".join(summ))


def rotating_set(skel: List[List[int]], pivot: int, b: int, n: int) -> Set[int]:
    nb = defaultdict(set)
    for x, y in skel:
        nb[x].add(y); nb[y].add(x)
    seen, stack = {b}, [b]
    while stack:
        x = stack.pop()
        for y in nb[x]:
            if y != pivot and y not in seen:
                seen.add(y); stack.append(y)
    return seen


def cmd_standardise(args):
    coco = json.load(open(args.input))
    sets = load_sets(coco, args.category)
    skel = next((c.get("skeleton") for _, _, c in sets if c.get("skeleton")), None) or []
    specs = []
    for spec in args.rotate:
        parts = spec.split(":")
        p, a, b = parse_joint(":".join(parts[:2]))
        if len(parts) > 2 and parts[2].strip():
            rot = {int(x) for x in parts[2].split(",")}
        else:
            if not skel:
                sys.exit(f"--rotate {spec}: no landmark list and no skeleton to find the distal part")
            rot = rotating_set(skel, p, b, 0)
        specs.append((p, a, b, rot))
    report = []
    for p, a, b, rot in specs:
        thetas = [math.radians(v) for ann, _, _ in sets if (v := angle(points(ann), p, a, b, signed=True)) is not None]
        if not thetas:
            sys.exit(f"joint {p}:{a},{b} is not measurable on any specimen")
        target = math.radians(args.target_angle) if args.target_angle is not None else \
            math.atan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas)))     # circular mean
        n_done = 0
        for ann, _, _ in sets:
            pts = points(ann)
            th = angle(pts, p, a, b, signed=True)
            if th is None:
                continue
            d = target - math.radians(th)
            R = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
            kp = ann["keypoints"]
            order = ann.get("point_order") or list(range(1, len(kp) // 3 + 1))
            for j, o in enumerate(order):
                if int(o) in rot and kp[3 * j + 2] > 0:
                    q = R @ (np.array(kp[3 * j:3 * j + 2], float) - pts[p]) + pts[p]
                    kp[3 * j], kp[3 * j + 1] = round(float(q[0]), 3), round(float(q[1]), 3)
            n_done += 1
        report.append({"pivot": p, "a": a, "b": b, "rotated_landmarks": sorted(rot), "specimens": n_done,
                       "target_angle_deg": round(math.degrees(target), 3),
                       "angle_sd_before_deg": round(float(np.degrees(math.sqrt(max(0.0, -2 * math.log(max(
                           math.hypot(np.mean(np.sin(thetas)), np.mean(np.cos(thetas))), 1e-12)))))), 3)})   # circular sd
    coco.setdefault("info", {})["descriptron_joint_standardisation"] = report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(args.out, "w"), indent=1)
    for r in report:
        print(f"joint {r['a']}-{r['pivot']}-{r['b']}: rotated landmarks {r['rotated_landmarks']} on {r['specimens']} "
              f"specimens to {r['target_angle_deg']} deg (sd before {r['angle_sd_before_deg']} deg)")
    print(f"wrote {args.out} (run the landmark GPA on this file; the original is unchanged)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("Joints:")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("angles"); p.add_argument("input"); p.add_argument("--out-dir", required=True)
    p.add_argument("--joint", action="append", help="PIVOT:A,B (repeat); default: all joints of the skeleton")
    p.add_argument("--category"); p.set_defaults(fn=cmd_angles)
    p = sub.add_parser("standardise", aliases=["standardize"]); p.add_argument("input"); p.add_argument("--out", required=True)
    p.add_argument("--rotate", action="append", required=True, help="PIVOT:A,B[:L1,L2,...] (repeat for several joints)")
    p.add_argument("--target-angle", type=float, help="signed target angle A-PIVOT-B in degrees (default: circular mean)")
    p.add_argument("--category"); p.set_defaults(fn=cmd_standardise)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
