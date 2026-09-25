#!/usr/bin/env python3
"""
validate_semilandmarks_vs_geomorph.py — Descriptron's outline semilandmark GPA side by side with R's geomorph
=============================================================================================================

Takes the output folder of semi_landmark_and_kpts_procrustesV42_GPA.py for one closed-outline category and gives
geomorph exactly the same semilandmarks (back_transformed_coco.json: the equal-arc-length points in the order the
GPA settled on, mirror images already reflected), then compares:

  1. gpagen WITHOUT sliding (fixed semilandmarks, what Descriptron does) vs Descriptron's aligned shapes:
     Procrustes distances between all pairs of specimens — these should agree.
  2. gpagen WITH sliding along the outline (curves =, closed curve, all points semilandmarks), minimising
     Procrustes distance (ProcD = TRUE) and bending energy (ProcD = FALSE), vs V42's --slide_method procd /
     bending (slide_gpa) on the same points.

R is needed only to run this check:
    python validate_semilandmarks_vs_geomorph.py --rscript /path/to/Rscript --rlib /path/to/Rlib \\
        --category-dir v42_out/whole_wing --out-dir val_semilandmarks/
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

R_CODE = r'''
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
suppressMessages(library(geomorph))
d <- args[2]
raw <- as.matrix(read.csv(file.path(d, "semilandmarks_raw.csv")))
p <- ncol(raw) / 2
A <- arrayspecs(raw, p, 2)
pd <- function(g) { D <- as.matrix(dist(two.d.array(g$coords))); D[upper.tri(D)] }
out <- data.frame(fixed = pd(gpagen(A, Proj = FALSE, print.progress = FALSE)))
# a closed outline: every point slides between its two neighbours
sl <- cbind(c(p, 1:(p - 1)), 1:p, c(2:p, 1))
out$slid_procd <- pd(gpagen(A, curves = sl, ProcD = TRUE, Proj = FALSE, print.progress = FALSE))
out$slid_bending <- pd(gpagen(A, curves = sl, ProcD = FALSE, Proj = FALSE, print.progress = FALSE))
pc1 <- function(g) { e <- prcomp(two.d.array(g$coords))$sdev^2; e[1] / sum(e) }
write.csv(out, file.path(d, "geomorph_pdist.csv"), row.names = FALSE)
g0 <- gpagen(A, Proj = FALSE, print.progress = FALSE)
g1 <- gpagen(A, curves = sl, ProcD = TRUE, Proj = FALSE, print.progress = FALSE)
g2 <- gpagen(A, curves = sl, ProcD = FALSE, Proj = FALSE, print.progress = FALSE)
write.csv(data.frame(fixed = pc1(g0), slid_procd = pc1(g1), slid_bending = pc1(g2)),
          file.path(d, "geomorph_pc1.csv"), row.names = FALSE)
cat("geomorph done\n")
'''


def load_points(path: Path):
    coco = json.load(open(path))
    names = {i["id"]: i["file_name"] for i in coco["images"]}
    out = {}
    for a in coco["annotations"]:
        seg = a["segmentation"][0] if isinstance(a["segmentation"][0], list) else a["segmentation"]
        out[names[a["image_id"]]] = np.array(seg, float).reshape(-1, 2)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True)
    ap.add_argument("--category-dir", required=True, help="<V42 output>/<category> folder")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    cat, out = Path(a.category_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_points(cat / "back_transformed_coco.json")
    ali = load_points(cat / "aligned_coco.json")
    keys = [k for k in ali if k in raw]
    R = np.array([raw[k] for k in keys]); Y = np.array([ali[k] for k in keys])
    # back_transformed is on each specimen's own image; V42 reflected the mirror images (x -> -x) before its
    # GPA, so give geomorph the same reflected points
    rf = cat / "reflected_specimens.csv"
    if rf.exists():
        import csv as _csv
        refl = {r["file"]: int(r["reflected"]) for r in _csv.DictReader(open(rf))}
        for i, k in enumerate(keys):
            if any(k.startswith(f) or f.startswith(k) for f, v in refl.items() if v):
                R[i] = R[i] * np.array([-1.0, 1.0])
    # closed outlines repeat the first point at the end; drop it on BOTH sides for sliding (a slider needs
    # distinct neighbours) -- the fixed comparison uses the same points, so it stays like-for-like
    if np.allclose(R[:, 0], R[:, -1]):
        R, Y = R[:, :-1], Y[:, :-1]
    # re-normalise Descriptron's aligned shapes after dropping the duplicate point
    Y = Y - Y.mean(1, keepdims=True); Y /= np.sqrt((Y ** 2).sum(axis=(1, 2)))[:, None, None]
    with open(out / "semilandmarks_raw.csv", "w") as f:
        f.write(",".join(f"{c}{i + 1}" for i in range(R.shape[1]) for c in "xy") + "\n")
        for P in R:
            f.write(",".join(f"{v:.10f}" for v in P.ravel()) + "\n")
    (out / "geomorph_semilandmarks.R").write_text(R_CODE)
    subprocess.run([a.rscript, str(out / "geomorph_semilandmarks.R"), a.rlib, str(out)], check=True)

    # Descriptron distances with the same (row-major upper triangle -> R's column-major) ordering
    n = len(Y); F = Y.reshape(n, -1)
    D = np.linalg.norm(F[:, None] - F[None], axis=2)
    py = np.array([D[i, j] for j in range(n) for i in range(j)])   # R's D[upper.tri(D)] is column-major
    import csv
    import importlib.util
    g = {k: np.array([float(r[k]) for r in csv.DictReader(open(out / "geomorph_pdist.csv"))])
         for k in ("fixed", "slid_procd", "slid_bending")}
    # Descriptron's sliding (V42 slide_gpa) on the same raw points
    spec = importlib.util.spec_from_file_location(
        "v42", Path(__file__).resolve().parents[1] / "semi_landmark_and_kpts_procrustesV42_GPA.py")
    v42 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v42)
    dpy = {"fixed": None}
    for meth, key in (("procd", "slid_procd"), ("bending", "slid_bending")):
        slid, _ = v42.slide_gpa(list(R), v42.closed_sliders(R.shape[1]), meth)
        Fs = np.array(slid).reshape(len(slid), -1)
        Ds = np.linalg.norm(Fs[:, None] - Fs[None], axis=2)
        dpy[key] = np.array([Ds[i, j] for j in range(len(Fs)) for i in range(j)])
    pc = next(csv.DictReader(open(out / "geomorph_pc1.csv")))
    e = np.linalg.svd(F - F.mean(0), compute_uv=False) ** 2
    py_pc1 = e[0] / e.sum()
    rel_fixed = np.max(np.abs(py - g["fixed"]) / g["fixed"])
    dpy["fixed"] = py
    rel = {k: float(np.max(np.abs(dpy[k] - g[k]) / g[k])) for k in g}
    summary = {"specimens": n, "semilandmarks": int(Y.shape[1]),
               "max_rel_diff_vs_geomorph": rel,
               "fixed_max_rel_diff": float(rel_fixed),
               "fixed_corr": float(np.corrcoef(py, g["fixed"])[0, 1]),
               "slid_procd_corr_with_fixed": float(np.corrcoef(py, g["slid_procd"])[0, 1]),
               "slid_bending_corr_with_fixed": float(np.corrcoef(py, g["slid_bending"])[0, 1]),
               "slid_procd_median_ratio": float(np.median(g["slid_procd"] / py)),
               "slid_bending_median_ratio": float(np.median(g["slid_bending"] / py)),
               "PC1_share": {"descriptron_fixed": float(py_pc1), "geomorph_fixed": float(pc["fixed"]),
                             "geomorph_slid_procd": float(pc["slid_procd"]),
                             "geomorph_slid_bending": float(pc["slid_bending"])}}
    json.dump(summary, open(out / "semilandmarks_vs_geomorph_summary.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), dpi=150)
    for ax, (key, title, col) in zip(axes[:3], (("fixed", "geomorph gpagen, fixed semilandmarks", "#2a78d6"),
                                               ("slid_procd", "geomorph, slid (Procrustes distance)", "#1baf7a"),
                                               ("slid_bending", "geomorph, slid (bending energy)", "#8e44ad"))):
        ax.scatter(g[key], dpy[key], s=8, c=col, alpha=0.5, edgecolors="none")
        lo, hi = min(dpy[key].min(), g[key].min()), max(dpy[key].max(), g[key].max())
        ax.plot([lo, hi], [lo, hi], color="#555", lw=0.8, ls="--")
        ax.set_xlabel(f"{title}\n(Procrustes distance)")
        ax.set_ylabel({"fixed": "Descriptron V42, fixed", "slid_procd": "Descriptron V42, --slide_method procd",
                       "slid_bending": "Descriptron V42, --slide_method bending"}[key])
        ax.set_title(f"max relative difference {rel[key]:.1e}", fontsize=10)
    ax = axes[3]; ax.axis("off")
    ax.text(0, 0.98, f"{n} outlines x {Y.shape[1]} semilandmarks\n(same points into both)\n\n"
            f"{len(py)} pairs of specimens.\n\n"
            "The three methods differ from\neach other; distances correlate\nwith fixed at\n"
            f"r = {summary['slid_procd_corr_with_fixed']:.3f} (Procrustes) and\n"
            f"r = {summary['slid_bending_corr_with_fixed']:.3f} (bending energy).\n\n"
            f"PC1 share: fixed {py_pc1:.3f}\n  slid (ProcD) {float(pc['slid_procd']):.3f}\n"
            f"  slid (bending) {float(pc['slid_bending']):.3f}",
            va="top", fontsize=9.5, family="monospace")
    fig.suptitle("Descriptron outline semilandmarks vs geomorph (R): fixed, and both sliding methods", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "semilandmarks_vs_geomorph.png"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
