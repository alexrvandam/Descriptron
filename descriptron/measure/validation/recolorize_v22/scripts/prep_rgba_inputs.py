#!/usr/bin/env python3
"""Build RGBA PNG inputs for recolorize from Descriptron's own image/mask pairs, so both programs see the
same pixels and the same mask.

RGB  = the image Descriptron's colour step reads (the *_fg_* export, or a Descriptron fhs_* output)
alpha = the *_bin_* mask Descriptron uses (255 = specimen, 0 = background)

Usage:
  prep_rgba_inputs.py --pairs_dir <dir with *_fg_*/*_bin_* pairs> --out_dir <dir>
                      [--rgb_dir <dir with <prefix><fg name> images>] [--rgb_prefix fhs_]
Writes <out_dir>/<fg stem>.png and <out_dir>/manifest.csv (fg name, bin name, rgb source, n_fg_pixels).
Reads only; writes only into --out_dir.
"""
import argparse, csv, os, sys
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rgb_dir", default=None, help="take RGB from <rgb_dir>/<rgb_prefix><fg name> instead of the fg file")
    ap.add_argument("--rgb_prefix", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    fgs = sorted(f for f in os.listdir(a.pairs_dir) if "_fg_" in f and f.endswith(".png"))
    rows = []
    for fg in fgs:
        bn = fg.replace("_fg_", "_bin_", 1)
        bp = os.path.join(a.pairs_dir, bn)
        if not os.path.isfile(bp):
            print("no mask for", fg, file=sys.stderr); continue
        src = os.path.join(a.rgb_dir, a.rgb_prefix + fg) if a.rgb_dir else os.path.join(a.pairs_dir, fg)
        if not os.path.isfile(src):
            print("no rgb for", fg, src, file=sys.stderr); continue
        rgb = np.array(Image.open(src).convert("RGB"))
        m = np.array(Image.open(bp))
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape != rgb.shape[:2]:
            print("shape mismatch", fg, file=sys.stderr); continue
        alpha = np.where(m > 0, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        Image.fromarray(rgba, "RGBA").save(os.path.join(a.out_dir, os.path.splitext(fg)[0] + ".png"))
        rows.append((fg, bn, src, int((alpha > 0).sum())))
    with open(os.path.join(a.out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fg", "bin", "rgb_source", "n_fg_pixels"]); w.writerows(rows)
    print(len(rows), "RGBA images written to", a.out_dir)


if __name__ == "__main__":
    main()
