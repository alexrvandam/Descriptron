#!/usr/bin/env python3
"""Centre-line length on shapes of known length (straight bar, arcs of increasing bend, an S-curve), at several
widths, against the straight measurement a length-along-the-main-axis would give. Writes a CSV and prints a summary;
the numbers quoted in the paper and README come from this script.

  python validate_centerline_lengths.py --out_csv centerline_validation.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_centerline as dc  # noqa: E402


def mask(pts, width, size=1000):
    m = np.zeros((size, size), np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1, width)
    return m.astype(bool)


def arclen(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def axis_extent(m):
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(float); P -= P.mean(0)
    pr = P @ np.linalg.svd(P, full_matrices=False)[2][0]
    return float(pr.max() - pr.min())


T = np.linspace(0, 1, 4000)
SHAPES = {"straight": np.stack([200 + 346.4 * T, 300 + 200 * T], 1)}
for deg in (60, 90, 120):
    a = np.radians(-90 - deg / 2 + deg * T)
    SHAPES[f"arc_{deg}deg"] = np.stack([500 + 300 * np.cos(a), 700 + 300 * np.sin(a)], 1)
SHAPES["s_curve"] = np.stack([150 + 700 * T, 500 + 120 * np.sin(2 * np.pi * T)], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()
    rows = []
    for name, pts in SHAPES.items():
        for w in (6, 10, 20, 30):
            m = mask(pts, w)
            true = arclen(pts) + w                     # round caps add w/2 at each end
            r = dc.centerline(m)
            rows.append(dict(shape=name, width_px=w, true_length_px=round(true, 1),
                             centerline_px=round(r["length_px"], 1),
                             centerline_err_pct=round(100 * (r["length_px"] - true) / true, 2),
                             straight_axis_px=round(axis_extent(m), 1),
                             straight_err_pct=round(100 * (axis_extent(m) - true) / true, 2)))
    with open(a.out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    ce = np.abs([r["centerline_err_pct"] for r in rows])
    print(f"centre line: |error| median {np.median(ce):.2f}%, max {ce.max():.2f}% over {len(rows)} shape x width cases")
    for name in SHAPES:
        s = [r["straight_err_pct"] for r in rows if r["shape"] == name]
        c = [r["centerline_err_pct"] for r in rows if r["shape"] == name]
        print(f"  {name:12s} straight axis {min(s):+.1f} to {max(s):+.1f}%   centre line {min(c):+.2f} to {max(c):+.2f}%")


if __name__ == "__main__":
    main()
