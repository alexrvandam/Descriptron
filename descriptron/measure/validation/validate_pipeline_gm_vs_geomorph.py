#!/usr/bin/env python3
"""
validate_pipeline_gm_vs_geomorph.py — the pipeline's other geometric-morphometric steps side by side with geomorph
=================================================================================================================

Complements validate_shape_stats_vs_geomorph.py (statistics) and validate_semilandmarks_vs_geomorph.py
(semilandmark GPA and sliding). This one checks the remaining places the pipeline does geometric morphometrics,
and shows the mirror-image problem the checks found and 2.1 fixes:

  a  landmark_gpa_V2 GPA vs gpagen, same raw keypoints (mirror images reflected before both)
  b  landmark PCA with mirror images left in (V1) ...
  c  ... and reflected (V2), coloured by mirror image
  d  semi_landmark_and_kpts_procrustesV42_GPA shape PCA scores vs geomorph gm.prcomp, same semilandmarks
  e  share of shape variance per PC: V42 (covariance, as gm.prcomp), geomorph, and the correlation PCA V34 used
  f  biorag_homology_frame_v1 GPA vs gpagen, same outline semilandmarks
  g  colour and texture measured in homologous grid cells: how well mirror-image specimens' cell patterns match
     the other specimens, with V34 outputs and with V42 outputs

R and geomorph are needed only to run this check:
    python validate_pipeline_gm_vs_geomorph.py --rscript Rscript --rlib Rlib --keypoints-json kp.json \\
        --outline-json whole_wing.json --v42-dir v42/whole_wing --v34-colour colhom_v34 --v42-colour colhom_v42 \\
        --v34-texture texhom_v34 --v42-texture texhom_v42 --out-dir out/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

R_GPA = r'''
args <- commandArgs(trailingOnly = TRUE); .libPaths(c(args[1], .libPaths())); suppressMessages(library(geomorph))
raw <- as.matrix(read.csv(args[2])); A <- arrayspecs(raw, ncol(raw) / 2, 2)
g <- gpagen(A, Proj = FALSE, print.progress = FALSE); D <- as.matrix(dist(two.d.array(g$coords)))
write(D[upper.tri(D)], args[3], ncolumns = 1)
if (length(args) > 3) { p <- gm.prcomp(g$coords); write.csv(data.frame(p$x[, 1:3]), args[4], row.names = FALSE)
  write.csv(data.frame(share = p$sdev^2 / sum(p$sdev^2)), args[5], row.names = FALSE) }
'''


def _write_raw(path, X):
    with open(path, "w") as f:
        f.write(",".join(f"{c}{i + 1}" for i in range(X.shape[1]) for c in "xy") + "\n")
        for P in X:
            f.write(",".join(f"{v:.10f}" for v in P.ravel()) + "\n")


def _pd_colmajor(Y):
    F = np.asarray(Y).reshape(len(Y), -1)
    D = np.linalg.norm(F[:, None] - F[None], axis=2)
    return np.array([D[i, j] for j in range(len(F)) for i in range(j)])


def _mirror_of(name, refl):
    return max((v for k, v in refl.items() if name.split(".tif")[0] in k or k.split(".tif")[0] in name), default=0)


def _cell_agreement(csv_path, refl, family):
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if re.match(r"^r\d+c\d+_", c) and c.endswith("_" + family)]
    X = np.nan_to_num(df[cols].to_numpy(float))
    mir = np.array([_mirror_of(str(f), refl) for f in df.iloc[:, 0]])
    r = np.array([np.corrcoef(X[i], X[(mir == 0) & (np.arange(len(X)) != i)].mean(0))[0, 1] for i in range(len(X))])
    return float(np.nanmean(r[mir == 1])), float(np.nanmean(r[mir == 0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True)
    ap.add_argument("--keypoints-json", required=True, help="COCO keypoints (landmark GPA)")
    ap.add_argument("--keypoints-category", default=None)
    ap.add_argument("--outline-json", required=True, help="COCO with one outline category (homology frame)")
    ap.add_argument("--v42-dir", required=True, help="<V42 output>/<category>")
    ap.add_argument("--v34-colour", required=True); ap.add_argument("--v42-colour", required=True)
    ap.add_argument("--v34-texture", required=True); ap.add_argument("--v42-texture", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rfile = out / "gpagen.R"; rfile.write_text(R_GPA)
    run_r = lambda *x: subprocess.run([a.rscript, str(rfile), a.rlib, *map(str, x)], check=True)
    res = {}

    # a-c: landmark GPA
    import landmark_gpa_V2 as lg
    cfgs, fns = next(iter(lg.load_keypoints(a.keypoints_json, a.keypoints_category).values()))
    mir_k = lg.mirrored_specimens(cfgs)
    X = np.array([np.column_stack([-c[:, 0], c[:, 1]]) if m else c for c, m in zip(cfgs, mir_k)])
    A2, _, _ = lg.gpa(list(X))
    _write_raw(out / "kp_raw.csv", X); run_r(out / "kp_raw.csv", out / "kp_g.txt")
    res["landmark"] = (np.loadtxt(out / "kp_g.txt"), _pd_colmajor(A2))
    A1, _, _ = lg.gpa(list(np.array(cfgs)))
    def pcs(A):
        F = A.reshape(len(A), -1); F = F - F.mean(0); u, s, vt = np.linalg.svd(F, full_matrices=False)
        return u[:, :2] * s[:2], s ** 2 / (s ** 2).sum()
    res["lm_v1"], res["lm_v2"], res["lm_mir"] = pcs(A1), pcs(A2), mir_k

    # d-e: V42 shape PCA vs gm.prcomp (same semilandmarks, mirror images reflected as V42 did)
    d42 = Path(a.v42_dir)
    bt = json.load(open(d42 / "back_transformed_coco.json")); bn = {i["id"]: i["file_name"] for i in bt["images"]}
    refl = {r["file"]: int(r["reflected"]) for r in csv.DictReader(open(d42 / "reflected_specimens.csv"))}
    cat = d42.name
    tr = {r["filename"]: r for r in csv.DictReader(open(next(d42.glob("shape_traits_phylo_*.csv"))))}
    raw, py = [], []
    for ann in bt["annotations"]:
        fn = bn[ann["image_id"]]; P = np.array(ann["segmentation"][0], float).reshape(-1, 2)
        raw.append(P * [-1, 1] if refl.get(fn, 0) else P); py.append([float(tr[fn][f"PC{i}"]) for i in (1, 2, 3)])
    _write_raw(out / "sl_raw.csv", np.array(raw))
    run_r(out / "sl_raw.csv", out / "sl_g.txt", out / "sl_gpc.csv", out / "sl_gvar.csv")
    gpc = np.loadtxt(out / "sl_gpc.csv", delimiter=",", skiprows=1)
    gvar = np.loadtxt(out / "sl_gvar.csv", delimiter=",", skiprows=1)
    v42var = np.array([float(r["variance_explained"]) for r in csv.DictReader(open(next(d42.glob("shape_pca_variance_*.csv"))))])
    al = json.load(open(d42 / "aligned_coco.json"))
    Y = np.array([np.array(an["segmentation"][0] if isinstance(an["segmentation"][0], list) else an["segmentation"], float)
                  for an in al["annotations"]])
    Z = (Y - Y.mean(0)) / Y.std(0).clip(1e-12); s = np.linalg.svd(Z, compute_uv=False); stdvar = s ** 2 / (s ** 2).sum()
    res["pca"] = (gpc, np.array(py), gvar, v42var, stdvar)

    # f: homology frame GPA vs gpagen
    import biorag_homology_frame_v1 as hf
    src = json.load(open(a.outline_json)); names = {i["id"]: i["file_name"] for i in src["images"]}
    L = [hf.semilandmarks(np.array(max(an["segmentation"], key=len), float).reshape(-1, 2), 100)
         for an in src["annotations"] if isinstance(an["segmentation"], list)]
    L = [l for l in L if l is not None]
    _, aligned, shifts = hf.gpa(L)
    Rr = np.array([np.roll(l, -int(k), axis=0) for l, k in zip(L, shifts)])
    _write_raw(out / "hf_raw.csv", Rr); run_r(out / "hf_raw.csv", out / "hf_g.txt")
    res["frame"] = (np.loadtxt(out / "hf_g.txt"), _pd_colmajor(aligned))

    # g: colour / texture homologous cells
    colcsv = lambda d: next(Path(d).glob("color_homology_features_*.csv"))
    texcsv = lambda d: next(Path(d).glob("texture_homology_features_*.csv"))
    res["cells"] = {
        "colour L*": (_cell_agreement(colcsv(a.v34_colour), refl, "L_mean_abs"), _cell_agreement(colcsv(a.v42_colour), refl, "L_mean_abs")),
        "texture contrast": (_cell_agreement(texcsv(a.v34_texture), refl, "glcm_contrast"), _cell_agreement(texcsv(a.v42_texture), refl, "glcm_contrast")),
        "texture homogeneity": (_cell_agreement(texcsv(a.v34_texture), refl, "glcm_homogeneity"), _cell_agreement(texcsv(a.v42_texture), refl, "glcm_homogeneity")),
    }

    summary = {"landmark_gpa_V2_vs_gpagen_max_rel": float(np.max(np.abs(res["landmark"][1] - res["landmark"][0]) / res["landmark"][0])),
               "homology_frame_vs_gpagen_max_rel": float(np.max(np.abs(res["frame"][1] - res["frame"][0]) / res["frame"][0])),
               "V42_PC_scores_abs_r": [float(abs(np.corrcoef(gpc[:, i], np.array(py)[:, i])[0, 1])) for i in range(3)],
               "PC1_share": {"geomorph": float(gvar[0]), "V42": float(v42var[0]), "correlation_PCA": float(stdvar[0])},
               "landmark_PC1_share": {"V1_mirrors_left_in": float(res["lm_v1"][1][0]), "V2": float(res["lm_v2"][1][0])},
               "mirror_images": {"keypoints": int(mir_k.sum()), "outlines": int(sum(refl.values()))},
               "cells_mirrored_vs_normal_r": {k: {"V34": v[0], "V42": v[1]} for k, v in res["cells"].items()}}
    json.dump(summary, open(out / "pipeline_gm_vs_geomorph_summary.json", "w"), indent=2)
    figure(res, summary, out / "pipeline_gm_vs_geomorph.png", len(cfgs), len(raw))
    print(json.dumps(summary, indent=2))


def figure(res, summary, path, n_kp, n_sl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, green, purple, orange, grey = "#2a78d6", "#1baf7a", "#8e44ad", "#e8743b", "#9aa3ad"
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.6), dpi=150); ax = ax.ravel()

    def ident(a_, x, y, col, xl, yl, title):
        a_.scatter(x, y, s=6, c=col, alpha=0.5, edgecolors="none")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); a_.plot([lo, hi], [lo, hi], color="#555", lw=0.8, ls="--")
        a_.set_xlabel(xl); a_.set_ylabel(yl); a_.set_title(title, fontsize=10)

    g, p = res["landmark"]
    ident(ax[0], g, p, blue, "geomorph gpagen", "landmark_gpa_V2",
          f"a  Landmark GPA, {n_kp} forewings x 17 landmarks\nmax relative difference {summary['landmark_gpa_V2_vs_gpagen_max_rel']:.1e}")
    for a_, key, lab in ((ax[1], "lm_v1", "b  Landmark PCA, mirror images left in (V1)"),
                         (ax[2], "lm_v2", "c  ... mirror images reflected first (V2)")):
        sc, var = res[key]; m = res["lm_mir"]
        a_.scatter(sc[~m, 0], sc[~m, 1], s=14, c=grey, label="as photographed")
        a_.scatter(sc[m, 0], sc[m, 1], s=18, c=orange, label="mirror image")
        a_.set_xlabel(f"PC1 ({100 * var[0]:.1f} %)"); a_.set_ylabel(f"PC2 ({100 * var[1]:.1f} %)"); a_.set_title(lab, fontsize=10)
        a_.legend(frameon=False, fontsize=8)
    gpc, py, gvar, v42var, stdvar = res["pca"]
    for i, col in zip(range(3), (blue, green, purple)):
        s = np.sign(np.corrcoef(gpc[:, i], py[:, i])[0, 1])
        ax[3].scatter(gpc[:, i], s * py[:, i], s=12, c=col, alpha=0.7, label=f"PC{i + 1}")
    lo, hi = gpc[:, :3].min(), gpc[:, :3].max(); ax[3].plot([lo, hi], [lo, hi], color="#555", lw=0.8, ls="--")
    ax[3].set_xlabel("geomorph gm.prcomp score"); ax[3].set_ylabel("V42 score (sign matched)")
    ax[3].set_title(f"d  Outline shape PCA, {n_sl} wings\n|r| = " + ", ".join(f"{r:.4f}" for r in summary["V42_PC_scores_abs_r"]), fontsize=10)
    ax[3].legend(frameon=False, fontsize=8)
    k = np.arange(1, 7); w = 0.27
    ax[4].bar(k - w, gvar[:6] * 100, w, color=blue, label="geomorph gm.prcomp")
    ax[4].bar(k, v42var[:6] * 100, w, color=green, label="V42 (covariance)")
    ax[4].bar(k + w, stdvar[:6] * 100, w, color=grey, label="standardised (as V34)")
    ax[4].set_xlabel("principal component"); ax[4].set_ylabel("% of shape variance"); ax[4].legend(frameon=False, fontsize=8)
    ax[4].set_title("e  Shape PCA: covariance (as geomorph) vs\nthe standardised PCA V34 used", fontsize=10)
    g, p = res["frame"]
    ident(ax[5], g, p, purple, "geomorph gpagen", "biorag_homology_frame_v1",
          f"f  Homology-frame GPA, {n_sl} outlines\nmax relative difference {summary['homology_frame_vs_gpagen_max_rel']:.1e}")
    names = list(res["cells"]); x = np.arange(len(names)); w = 0.2
    for j, (lab, col, idx, which) in enumerate((("mirror images, V34", orange, 0, 0), ("mirror images, V42", green, 1, 0),
                                                  ("other specimens", grey, 1, 1))):
        ax[6].bar(x + (j - 1) * w, [res["cells"][n][idx][which] for n in names], w, color=col, label=lab)
    ax[6].set_xticks(x); ax[6].set_xticklabels(names, fontsize=8); ax[6].set_ylim(0, 1)
    ax[6].set_ylabel("r with the other specimens'\ncell pattern"); ax[6].legend(frameon=False, fontsize=8)
    ax[6].set_title("g  Colour and texture in homologous cells:\ndo mirror images land on the same anatomy?", fontsize=10)
    ax[7].axis("off")
    ax[7].text(0, 0.97, "Same raw points into both sides.\n\n"
               "a, f  GPA: identical to gpagen.\n"
               "d, e  Shape PCA: identical to\n      gm.prcomp (V42 default).\n\n"
               "b, c  Mirror images (orange) could\n      not be superimposed and made\n"
               f"      PC1 ({100 * summary['landmark_PC1_share']['V1_mirrors_left_in']:.1f} %); reflected first:\n"
               f"      {100 * summary['landmark_PC1_share']['V2']:.1f} %.\n\n"
               "g  Their homologous colour and\n   texture cells missed the anatomy\n   (V34); with V42 they match the\n   other specimens.\n\n"
               "Diaphorina forewings and wing\noutlines; geomorph 4.1.1.",
               va="top", fontsize=9.5, family="monospace")
    fig.suptitle("Descriptron's other geometric-morphometric steps vs geomorph (R), and the mirror-image fix", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main()
