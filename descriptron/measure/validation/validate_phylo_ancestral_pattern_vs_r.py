#!/usr/bin/env python3
"""descriptron_phylo --analyses ancestral_pattern against R.

Python writes the species-mean L*, a*, b* of the homology grid cells (with the cells each specimen left empty set
missing, --cell_geometry) and the tree; R computes; Descriptron computes the same from the same inputs:

  1. ancestral states and 95% intervals of cells every species has: phytools::fastAnc(vars = TRUE, CI = TRUE) per
     cell channel (estimate, variance, interval), ape::ace(method = "REML") and ace(method = "ML") estimates (and their
     CI95 and sigma^2, reported: they are computed differently from fastAnc's)
  2. cells some species lack: fastAnc on the tree pruned to the species that have the cell (ape::drop.tip), nodes
     matched by their descendant species; the nodes the pruning removes (Descriptron gives them the conditional
     expectation) against the same formula evaluated in R from ape's node covariance (phytools::vcvPhylo(anc.nodes =
     TRUE)) and path lengths (ape::dist.nodes) - a transcription, as no installed R package reconstructs with missing
     tips (Rphylopars is not installed)
  3. per-cell signal and rate: Blomberg's K per channel against phytools::phylosig(method = "K"), Kmult of the three
     channels against an R transcription of Adams (2014) (geomorph is validated elsewhere,
     validate_phylo_real_tree.py), the REML rate against mean(ape::pic(x, tree)^2)
  4. drawing colours: lab_to_rgb against grDevices::convertColor(from = "Lab", to = "sRGB") and
     farver::convert_colour(from = "lab", to = "rgb"), on a grid of in-gamut colours and the reconstructed colours;
     optionally (--opencv_python) the round trip through OpenCV's 8-bit RGB -> L*a*b* (the encoder of the photographs)

  python validate_phylo_ancestral_pattern_vs_r.py --rscript Rscript --out-dir val_anc/ --tree tree.nwk \\
      --groups group_labels.csv --traits color_homology_features_soma_abdomen_combined_hw1.csv \\
      --cell_geometry color_homology_cells_soma_abdomen.json [--opencv_python measure_env/bin/python]
"""
import argparse, csv, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np


def _sibling(name):
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)


def load_dp():
    spec = importlib.util.spec_from_file_location("descriptron_phylo", _sibling("descriptron_phylo.py"))
    dp = importlib.util.module_from_spec(spec); sys.modules["descriptron_phylo"] = dp; spec.loader.exec_module(dp)
    return dp


R_CODE = r'''
args <- commandArgs(TRUE); d <- args[1]
suppressMessages({library(ape); library(phytools); library(jsonlite); library(farver)})
out <- list(versions = sapply(c("ape", "phytools", "farver", "grDevices"), function(p) as.character(packageVersion(p))))
tr <- read.tree(file.path(d, "tree.nwk"))
X <- read.csv(file.path(d, "tips.csv"), row.names = 1, check.names = FALSE)
X <- X[tr$tip.label, , drop = FALSE]
desc <- function(t) { n <- Ntip(t); sapply(n + 1:t$Nnode, function(v) paste(sort(t$tip.label[getDescendants(t, v)[getDescendants(t, v) <= n]]), collapse = "|")) }
cols_full <- readLines(file.path(d, "cols_full.txt")); cols_miss <- readLines(file.path(d, "cols_missing.txt"))
cols_ace <- readLines(file.path(d, "cols_ace.txt"))
res <- list()
for (cn in cols_full) {
  x <- setNames(X[[cn]], rownames(X))
  f <- fastAnc(tr, x, vars = TRUE, CI = TRUE)
  res[[cn]] <- list(nodes = desc(tr), est = unname(f$ace), var = unname(f$var), lo = unname(f$CI95[, 1]), hi = unname(f$CI95[, 2]))
}
out$fastanc_full <- res
res <- list()
for (cn in cols_ace) {
  x <- setNames(X[[cn]], rownames(X))
  a1 <- ace(x, tr, type = "continuous", method = "REML"); a2 <- ace(x, tr, type = "continuous", method = "ML")
  m <- mean(x); s <- sd(x); a3 <- ace((x - m) / s, tr, type = "continuous", method = "ML")   # ace's optimiser on standardised data
  res[[cn]] <- list(nodes = desc(tr), reml = unname(a1$ace), ml = unname(a2$ace), reml_lo = unname(a1$CI95[, 1]),
                    ml_lo = unname(a2$CI95[, 1]), reml_sig2 = a1$sigma2[1], ml_sig2 = a2$sigma2[1],
                    ml_std = unname(a3$ace * s + m), K = as.numeric(phylosig(tr, x, method = "K")[1]), pic_sig2 = mean(pic(x, tr)^2))
}
out$ace <- res
# cells some species lack
V <- vcvPhylo(tr, anc.nodes = TRUE); DN <- dist.nodes(tr); n <- Ntip(tr)
res <- list()
for (cn in cols_miss) {
  x <- setNames(X[[cn]], rownames(X)); ok <- names(x)[is.finite(x)]
  tp <- drop.tip(tr, setdiff(tr$tip.label, ok))
  f <- fastAnc(tp, x[ok], vars = TRUE, CI = TRUE)
  # conditional expectation on the full tree, transcribed: mu + C_nS C_SS^-1 (y - mu); variance sigma2_REML / (1' C_(i)^-1 1)
  si <- match(ok, tr$tip.label); y <- x[ok]
  Css <- V[ok, ok]; Ci <- solve(Css); one <- rep(1, length(si))
  mu <- sum(Ci %*% y) / sum(Ci); r <- y - mu
  inner <- n + 1:tr$Nnode
  Cns <- t(sapply(inner, function(v) if (v == n + 1) rep(0, length(si)) else V[as.character(v), ok]))   # vcvPhylo omits the root
  ce <- as.vector(mu + Cns %*% Ci %*% r)
  s2 <- as.numeric(t(r) %*% Ci %*% r) / (length(si) - 1)
  cv <- sapply(inner, function(v) { dd <- DN[v, si]; Cv <- 0.5 * (outer(dd, dd, "+") - DN[si, si]); s2 / sum(solve(Cv, one)) })
  res[[cn]] <- list(pruned_nodes = desc(tp), est = unname(f$ace), var = unname(f$var), lo = unname(f$CI95[, 1]),
                    full_nodes = desc(tr), ce = ce, cv = cv, sig2 = s2)
}
out$fastanc_missing <- res
# Kmult transcription (Adams 2014)
km <- function(Y, C) { Y <- as.matrix(Y); n <- nrow(Y); Ci <- solve(C); one <- matrix(1, n, 1)
  a <- solve(t(one) %*% Ci %*% one) %*% t(one) %*% Ci %*% Y; R <- Y - one %*% a
  obs <- sum(diag(t(R) %*% R)) / sum(diag(t(R) %*% Ci %*% R)); ex <- (sum(diag(C)) - n / sum(Ci)) / (n - 1); obs / ex }
C <- vcv(tr); cells <- readLines(file.path(d, "cells_kmult.txt"))
out$kmult <- lapply(setNames(cells, cells), function(cl) km(X[, paste0(cl, "_", c("L", "a", "b"), "_mean_abs")], C))
# colours
L <- as.matrix(read.csv(file.path(d, "lab_colours.csv")))
out$convertColor <- convertColor(L, from = "Lab", to = "sRGB", clip = NA)
out$farver <- convert_colour(L, "lab", "rgb") / 255
write(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(d, "R_results.json"))
'''

OPENCV_CODE = r'''
import sys, json, numpy as np, cv2
rgb = np.array(json.load(open(sys.argv[1])), np.uint8).reshape(-1, 1, 3)
lab = cv2.cvtColor(rgb[..., ::-1].copy(), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
lab[:, 0] *= 100.0 / 255.0; lab[:, 1:] -= 128.0                     # as color_phenomics_homology_v2_1.py
labf = cv2.cvtColor((rgb[..., ::-1].astype(np.float32) / 255.0).copy(), cv2.COLOR_BGR2LAB).reshape(-1, 3)
json.dump({"u8": lab.tolist(), "f32": labf.astype(float).tolist()}, open(sys.argv[2], "w"))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", default="Rscript"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tree", required=True); ap.add_argument("--groups", required=True)
    ap.add_argument("--traits", required=True); ap.add_argument("--cell_geometry", required=True)
    ap.add_argument("--n_full", type=int, default=120, help="complete cells checked against fastAnc (x 3 channels)")
    ap.add_argument("--n_ace", type=int, default=40, help="cell channels checked against ace / phylosig / pic")
    ap.add_argument("--opencv_python", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dp = load_dp(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    root, _ = dp.load_tree(a.tree)
    geom = dp.load_cell_geometry(a.cell_geometry)
    gmap = dp.load_group_map(a.groups)
    sp, cols, M, _ = dp.load_traits(a.traits, None, None, None, gmap, keep_missing=True, blank=geom["blank"])
    sp = [dp._norm_name(s) for s in sp]
    names = {t.name for t in dp.tips_of(root)}
    tree = dp.prune(dp._copy_tree(root), set(sp) & names); tips = dp.tips_of(tree)
    cells, cix = dp.colour_cell_index(cols)
    lab = M[[sp.index(t.name) for t in tips]][:, cix]                       # species x cells x 3
    ns = len(tips); nobs = np.isfinite(lab[..., 0]).sum(0)
    full = np.where(nobs == ns)[0]; miss = np.where((nobs < ns) & (nobs >= max(3, int(np.ceil(0.5 * ns)))))[0]
    pick_full = np.sort(rng.choice(full, min(a.n_full, len(full)), replace=False))
    ch = "Lab"
    colname = lambda c, k: f"{cells[c]}_{ch[k]}_mean_abs"
    nw = dp.to_newick(tree); (out / "tree.nwk").write_text(nw if nw.rstrip().endswith(";") else nw + ";")
    with open(out / "tips.csv", "w", newline="") as fh:
        w = csv.writer(fh); allc = [colname(c, k) for c in range(len(cells)) for k in range(3)]
        w.writerow(["species"] + allc)
        for i, t in enumerate(tips):
            w.writerow([t.name] + ["NA" if not np.isfinite(lab[i, c, k]) else repr(float(lab[i, c, k]))
                                   for c in range(len(cells)) for k in range(3)])
    cf = [colname(c, k) for c in pick_full for k in range(3)]
    cm = [colname(c, k) for c in miss for k in range(3)]
    ca = [cf[i] for i in np.sort(rng.choice(len(cf), min(a.n_ace, len(cf)), replace=False))]
    (out / "cols_full.txt").write_text("\n".join(cf) + "\n"); (out / "cols_missing.txt").write_text("\n".join(cm) + "\n")
    (out / "cols_ace.txt").write_text("\n".join(ca) + "\n")
    kcells = [cells[c] for c in pick_full[:30]]
    (out / "cells_kmult.txt").write_text("\n".join(kcells) + "\n")
    # colours: an in-gamut grid (sRGB 8-bit levels converted to Lab by the textbook formulas) + reconstructed colours
    nodes = dp.internal_nodes(tree)
    R = dp.bm_ancestral(tips, nodes, lab.reshape(ns, -1))
    est = R["est"].reshape(len(nodes), len(cells), 3)
    g = np.linspace(0, 1, 9); rgb_grid = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    lin = np.where(rgb_grid <= 0.04045, rgb_grid / 12.92, ((rgb_grid + 0.055) / 1.055) ** 2.4)
    Mx = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ Mx.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab_grid = np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]
    lab_rec = est[np.isfinite(est).all(-1)].reshape(-1, 3)[:: max(1, est.size // 3 // 2000)]
    L_all = np.r_[lab_grid, lab_rec]
    np.savetxt(out / "lab_colours.csv", L_all, delimiter=",", header="L,a,b", comments="")
    (out / "r_side.R").write_text(R_CODE)
    r = subprocess.run([a.rscript, str(out / "r_side.R"), str(out)], capture_output=True, text=True)
    if r.returncode:
        sys.exit("R failed:\n" + r.stderr[-3000:])
    RJ = json.load(open(out / "R_results.json"))
    key = lambda nd: "|".join(sorted(t.name for t in dp.tips_of(nd)))
    nkey = {key(nd): i for i, nd in enumerate(nodes)}
    nsets = [(frozenset(k.split("|")), i) for k, i in nkey.items()]
    mrca_of = lambda s_: min((len(S_), i) for S_, i in nsets if frozenset(s_.split("|")) <= S_)[1]
    se = np.sqrt(R["var"]).reshape(len(nodes), len(cells), 3)
    cidx = {colname(c, k): (c, k) for c in range(len(cells)) for k in range(3)}
    summ = {"R_versions": RJ["versions"], "species": ns, "cells": len(cells), "cells_complete": int(len(full)),
            "cells_missing_some_reconstructed": int(len(miss))}
    # 1. complete cells vs fastAnc
    de = dv = dc = 0.0
    for cn, v in RJ["fastanc_full"].items():
        c, k = cidx[cn]; ix = [nkey[s] for s in v["nodes"]]
        de = max(de, np.abs(est[ix, c, k] - np.array(v["est"])).max())
        dv = max(dv, (np.abs(se[ix, c, k] ** 2 - np.array(v["var"])) / np.array(v["var"])).max())
        dc = max(dc, np.abs(est[ix, c, k] - dp.ANC_Z95 * se[ix, c, k] - np.array(v["lo"])).max())
    summ["fastAnc_complete_cells"] = {"cell_channels": len(RJ["fastanc_full"]), "max_abs_diff_estimate": de,
                                      "max_rel_diff_variance": dv, "max_abs_diff_CI95_low": dc}
    # ace
    dr = dm = dms = dk = ds = 0.0; ci_ratio_reml, ci_ratio_ml, sig = [], [], []
    for cn, v in RJ["ace"].items():
        c, k = cidx[cn]; ix = [nkey[s] for s in v["nodes"]]
        dr = max(dr, np.abs(est[ix, c, k] - np.array(v["reml"])).max())
        dm = max(dm, np.abs(est[ix, c, k] - np.array(v["ml"])).max())
        dms = max(dms, np.abs(est[ix, c, k] - np.array(v["ml_std"])).max())
        ci_ratio_reml.append(np.median((est[ix, c, k] - np.array(v["reml_lo"])) / (dp.ANC_Z95 * se[ix, c, k])))
        ci_ratio_ml.append(np.median((est[ix, c, k] - np.array(v["ml_lo"])) / (dp.ANC_Z95 * se[ix, c, k])))
        sig.append((v["reml_sig2"] / R["sigsq"].reshape(len(cells), 3)[c, k], v["ml_sig2"] / R["sigsq"].reshape(len(cells), 3)[c, k]))
        K, _ = dp.k_by_column(lab[:, c, k][:, None], dp.vcv(tips, tips))
        dk = max(dk, abs(K[0] - v["K"]))
        ds = max(ds, abs(R["sigsq"].reshape(len(cells), 3)[c, k] - v["pic_sig2"]) / v["pic_sig2"])
    sig = np.array(sig)
    summ["ace"] = {"cell_channels": len(RJ["ace"]), "max_abs_diff_estimate_vs_REML": dr, "max_abs_diff_estimate_vs_ML": dm,
                   "max_abs_diff_estimate_vs_ML_on_standardised_data": dms,
                   "ace_REML_CI_halfwidth_over_fastAnc_median": float(np.median(ci_ratio_reml)),
                   "ace_ML_CI_halfwidth_over_fastAnc_median": float(np.median(ci_ratio_ml)),
                   "ace_REML_sigma2_over_REML_contrasts": [float(sig[:, 0].min()), float(sig[:, 0].max())],
                   "ace_ML_sigma2_over_REML_contrasts": [float(sig[:, 1].min()), float(sig[:, 1].max())]}
    summ["phylosig_K_max_abs_diff"] = dk; summ["pic_sigma2_max_rel_diff"] = ds
    # 2. cells some species lack
    de = dv = dc = dce = dcv = 0.0; nmatched = nce = 0
    for cn, v in RJ["fastanc_missing"].items():
        c, k = cidx[cn]
        ix = [mrca_of(s) for s in v["pruned_nodes"]]                       # the MRCA in the full tree of each pruned node
        de = max(de, np.abs(est[ix, c, k] - np.array(v["est"])).max())
        dv = max(dv, (np.abs(se[ix, c, k] ** 2 - np.array(v["var"])) / np.array(v["var"])).max())
        dc = max(dc, np.abs(est[ix, c, k] - dp.ANC_Z95 * se[ix, c, k] - np.array(v["lo"])).max())
        nmatched += len(ix)
        fx = [nkey[s] for s in v["full_nodes"]]
        rest = [j for j, i_ in enumerate(fx) if i_ not in set(ix)]
        dce = max(dce, np.abs(est[fx, c, k] - np.array(v["ce"])).max())
        dcv = max(dcv, (np.abs(se[fx, c, k] ** 2 - np.array(v["cv"])) / np.array(v["cv"])).max())
        nce += len(rest)
    summ["fastAnc_cells_missing_in_some_species"] = {
        "cell_channels": len(RJ["fastanc_missing"]), "nodes_compared": nmatched, "max_abs_diff_estimate": de,
        "max_rel_diff_variance": dv, "max_abs_diff_CI95_low": dc,
        "all_nodes_vs_R_transcription": {"nodes_removed_by_pruning": nce, "max_abs_diff_estimate": dce,
                                         "max_rel_diff_variance": dcv}}
    # 3. Kmult
    Kd = 0.0
    for cl, kv in RJ["kmult"].items():
        c = cells.index(cl)
        _, Kg = dp.k_by_column(lab[:, c, :], dp.vcv(tips, tips), [np.arange(3)])
        Kd = max(Kd, abs(Kg[0] - float(np.ravel(kv)[0])))
    summ["Kmult_cell_max_abs_diff"] = Kd
    # 4. colours (in-gamut grid: also the round trip to the sRGB levels)
    py = dp.lab_to_rgb(L_all)
    cc = np.array(RJ["convertColor"], float); fv = np.array(RJ["farver"], float)
    ok = np.isfinite(cc).all(1)
    summ["lab_to_rgb"] = {"colours": len(L_all), "in_gamut_by_R": int(ok.sum()),
                          "max_abs_diff_vs_convertColor": float(np.abs(py[ok] - cc[ok]).max()),
                          "max_abs_diff_vs_farver": float(np.abs(py[: len(lab_grid)] - fv[: len(lab_grid)]).max()),
                          "max_abs_diff_vs_farver_all_8bit_levels": float(np.abs(py - fv).max() * 255),
                          "grid_round_trip_max_abs_diff": float(np.abs(py[: len(lab_grid)] - rgb_grid).max())}
    if a.opencv_python:
        rgb8 = np.array(np.meshgrid(*[np.arange(0, 256, 17)] * 3)).reshape(3, -1).T
        json.dump(rgb8.tolist(), open(out / "rgb8.json", "w")); (out / "opencv_side.py").write_text(OPENCV_CODE)
        r = subprocess.run([a.opencv_python, "-I", str(out / "opencv_side.py"), str(out / "rgb8.json"), str(out / "lab_cv.json")],
                           capture_output=True, text=True)
        if r.returncode == 0:
            J = json.load(open(out / "lab_cv.json"))
            back = dp.lab_to_rgb(np.array(J["u8"])) * 255; e = np.abs(back - rgb8).max(1)
            backf = dp.lab_to_rgb(np.array(J["f32"])) * 255
            summ["opencv_round_trip_8bit_levels"] = {
                "float32_lab_max_abs": float(np.abs(backf - rgb8).max()),
                "uint8_lab_median": float(np.median(e)), "uint8_lab_share_within_2": float((e <= 2).mean()),
                "uint8_lab_max_abs": float(e.max()),
                "note": "uint8 L*a*b* (the photographs' encoding) rounds a*, b* to integers; the large errors are "
                        "saturated colours with a channel near 0, where sRGB is steep"}
    (out / "validation_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
