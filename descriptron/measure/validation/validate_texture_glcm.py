#!/usr/bin/env python3
"""Checks of Descriptron's texture features (texture_phenomics_homology.py).

1. Masked GLCM against scikit-image: pixels outside the mask get an extra 'outside' grey level, scikit-image builds
   the symmetric GLCM, and that level's row and column are dropped; what remains is exactly the 'both pixels in the
   mask' matrix Descriptron computes. (scikit-image names the two diagonals the other way round; Descriptron averages
   the four directions, so its features are unaffected.)
2. Grey-level invariance: the same image quantised to 8 and 32 levels; ratio of the feature values (1 = invariant),
   for Descriptron's invariant Haralick formulas and the standard ones.

  python validate_texture_glcm.py --image_glob "<dir>/*.png" --texture_script <texture_phenomics_homology.py> --out_csv out.csv
"""
import argparse
import csv
import glob
import importlib.util

import cv2
import numpy as np
from skimage.feature import graycomatrix

ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_glob", required=True); ap.add_argument("--texture_script", required=True)
    ap.add_argument("--n_images", type=int, default=15); ap.add_argument("--out_csv", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("tx", a.texture_script)
    tx = importlib.util.module_from_spec(spec); spec.loader.exec_module(tx)
    imgs = sorted(glob.glob(a.image_glob))[: a.n_images]
    rows = []
    worst, n = 0.0, 0
    for f in imgs:
        g = cv2.imread(f, 0); q = (g // 16).astype(np.uint8)
        h, w = q.shape; yy, xx = np.mgrid[:h, :w]
        for s in range(3):
            cy, cx, ry, rx = h * (0.3 + 0.2 * s), w * (0.3 + 0.2 * s), h * 0.25, w * 0.2
            m = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
            for ang in ANGLES:
                ours = tx._masked_glcm_single(q, m, 3, ang, levels=16)
                sk = {np.pi / 4: 3 * np.pi / 4, 3 * np.pi / 4: np.pi / 4}.get(ang, ang)
                R = graycomatrix(np.where(m, q, 16).astype(np.uint8), [3], [sk], levels=17,
                                 symmetric=True, normed=False)[:16, :16, 0, 0].astype(float)
                R /= R.sum()
                worst = max(worst, float(np.abs(ours - R).max())); n += 1
        full = np.ones(g.shape, bool); gf = g.astype(float); vals = {}
        for L in (8, 32):
            ql = np.clip((gf - gf.min()) / (gf.max() - gf.min()) * (L - 1), 0, L - 1).astype(np.uint8)
            G = np.mean([tx._masked_glcm_single(ql, full, 3, an, levels=L) for an in ANGLES], axis=0)
            I, J = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
            vals[L] = (tx._haralick_invariant(G, levels=L),
                       [float((G * (I - J) ** 2).sum()), float((G / (1 + (I - J) ** 2)).sum()), float((G ** 2).sum())])
        for k, name in enumerate(("contrast", "homogeneity", "energy")):
            rows.append(dict(image=f, feature=name, invariant_ratio_32_vs_8=vals[32][0][k] / vals[8][0][k],
                             standard_ratio_32_vs_8=vals[32][1][k] / vals[8][1][k]))
    with open(a.out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"masked GLCM vs scikit-image: max |difference| {worst:.2e} over {n} matrices")
    for name in ("contrast", "homogeneity", "energy"):
        r = [x for x in rows if x["feature"] == name]
        print(f"{name:12s} ratio 32/8 levels, median: invariant {np.median([x['invariant_ratio_32_vs_8'] for x in r]):.2f}"
              f"   standard {np.median([x['standard_ratio_32_vs_8'] for x in r]):.2f}")


if __name__ == "__main__":
    main()
