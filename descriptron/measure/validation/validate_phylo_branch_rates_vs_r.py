#!/usr/bin/env python3
"""descriptron_phylo rate-coloured ancestral patterns and the comparison of rates between parts, against R.

Python writes the species-mean L*, a*, b* of the homology grid cells every species has, for each trait set, and the
tree; R computes; Descriptron computes the same from the same inputs:

  1. --anc_branch_colour rate_de00 (branch_pattern_changes): every branch's mean CIEDE2000 over the cells between the
     parent's and the child's pattern, divided by the branch length, with the patterns from phytools::fastAnc per cell
     channel (the tips: the species means) and the per-cell difference from farver::compare_colour(method =
     "cie2000"); branches matched by their descendant species (first trait set)
  2. --anc_branch_colour bm_rate (pattern_bm_branch_rates): phytools::multirateBM(lambda = 1) on each of the first k
     principal components of the standardised pattern features (prcomp(scale = TRUE) scores, signs matched), node
     rates and branch rates (the log-linear mean of the two node rates, as phytools draws them), and their mean over
     the k PCs (first trait set)
  3. --part_sets (multi_evol_rates, part_rates_section): geomorph::compare.multi.evol.rates(Subset = TRUE) on the
     parts' complete cells side by side: sigma^2_mult per part, the largest / smallest ratio and the pairwise ratios
     (exact), geomorph's makePD of the equal-rate matrix and its statistic on R's own simulated data (exact, on every
     --sub_cells-th cell), and the P values: geomorph's (its null divides each part by the total number of
     variables), the same simulations with each part divided by its own number of variables (R transcription), and
     Descriptron's two nulls (Monte Carlo agreement only: different random numbers); Kmult per part against
     geomorph::physignal

  python validate_phylo_branch_rates_vs_r.py --rscript Rscript --rlib /path/to/Rlib_with_geomorph --out-dir val/ \\
      --tree tree.nwk --groups group_labels.csv \\
      --traits head=fine_grid_head/color_homology_features_head_combined_hw1.csv \\
      --cell_geometry head=fine_grid_head/color_homology_cells_head.json  [... one pair per part]
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
args <- commandArgs(TRUE); if (nzchar(args[1])) .libPaths(c(args[1], .libPaths())); d <- args[2]
iter <- as.integer(args[3])
suppressMessages({library(ape); library(phytools); library(farver); library(jsonlite); library(geomorph)})
out <- list(versions = sapply(c("ape", "phytools", "farver", "geomorph"), function(p) as.character(packageVersion(p))))
tr <- read.tree(file.path(d, "tree.nwk")); n <- Ntip(tr)
desc <- function(v) paste(sort(tr$tip.label[getDescendants(tr, v)[getDescendants(tr, v) <= n]]), collapse = "|")
key <- sapply(1:(n + tr$Nnode), function(v) if (v <= n) tr$tip.label[v] else desc(v))
# 1. branch Delta E00 from fastAnc + farver
X <- as.matrix(read.csv(file.path(d, "set1_lab.csv"), row.names = 1, check.names = FALSE))[tr$tip.label, ]
p <- ncol(X) / 3
st <- matrix(NA, n + tr$Nnode, ncol(X)); st[1:n, ] <- X
for (j in 1:ncol(X)) st[n + 1:tr$Nnode, j] <- fastAnc(tr, setNames(X[, j], rownames(X)))
lab_of <- function(v) matrix(st[v, ], ncol = 3, byrow = TRUE)          # columns c1L c1a c1b c2L ...
de <- sapply(1:nrow(tr$edge), function(e) {
  A <- lab_of(tr$edge[e, 1]); B <- lab_of(tr$edge[e, 2])
  mean(sapply(1:p, function(c) compare_colour(A[c, , drop = FALSE], B[c, , drop = FALSE], from_space = "lab", method = "cie2000")))
})
out$branches <- list(child = key[tr$edge[, 2]], length = tr$edge.length, dE00 = de, rate = de / tr$edge.length)
# 2. multirateBM on the PCs
F <- as.matrix(read.csv(file.path(d, "set1_features.csv"), row.names = 1, check.names = FALSE))[tr$tip.label, ]
k <- as.integer(readLines(file.path(d, "k.txt")))
pcs <- prcomp(F, scale. = TRUE)$x[, 1:k, drop = FALSE]
lnm <- function(a, b) ifelse(abs(a - b) < 1e-12, exp(a), (exp(a) - exp(b)) / (a - b))
mr <- list()
for (i in 1:k) {
  f <- multirateBM(tr, pcs[, i], lambda = 1, quiet = TRUE)
  ls <- log(f$sig2)
  mr[[i]] <- list(node = key, sig2 = unname(f$sig2), edge_child = key[tr$edge[, 2]],
                  edge_rate = lnm(ls[tr$edge[, 1]], ls[tr$edge[, 2]]), scores = unname(pcs[, i]), tips = rownames(pcs),
                  convergence = f$convergence)
}
out$multirate <- mr
# 3. geomorph compare.multi.evol.rates + physignal
P <- as.matrix(read.csv(file.path(d, "parts_lab.csv"), row.names = 1, check.names = FALSE))[tr$tip.label, ]
gp <- factor(readLines(file.path(d, "parts_gp.txt")), levels = readLines(file.path(d, "parts_names.txt")))
t0 <- Sys.time()
cm <- compare.multi.evol.rates(P, gp, tr, Subset = TRUE, iter = iter, seed = 1, print.progress = FALSE)
out$cmer <- list(rates = unname(cm$sigma.d.gp), ratio = cm$sigma.d.ratio, P = cm$P.value,
                 pair_ratio = as.vector(cm$sigma.d.gp.ratio), pair_P = as.vector(cm$pairwise.pvalue),
                 seconds = as.numeric(Sys.time() - t0, units = "secs"))
# the same simulations, each part divided by its own number of variables (transcription of geomorph's code)
pp <- geomorph:::phylo.mat(P, tr); obs <- geomorph:::sigma.d.multi(P, pp$invC, pp$D.mat, gp, TRUE)
Rm <- obs$R; diag(Rm) <- obs$rate.global; Rm <- geomorph:::makePD(Rm)
xs <- simplify2array(geomorph:::sim.char.BM(tr, Rm, nsim = iter, seed = 1))
N <- n; Xadj <- diag(N) - crossprod(matrix(1, N, N), pp$invC) / sum(pp$invC); Pt <- pp$D.mat %*% Xadj
g <- as.integer(gp); cmb <- combn(nlevels(gp), 2)
rr <- function(x, own) { r <- sapply(1:nlevels(gp), function(j) sum((Pt %*% x[, g == j, drop = FALSE])^2) / N /
                                  (if (own) sum(g == j) else ncol(x))); sapply(1:ncol(cmb), function(c) max(r[cmb[, c]]) / min(r[cmb[, c]])) }
obs_pair <- as.vector(obs$sigma.d.gp.ratio)
sim_own <- t(sapply(1:iter, function(j) rr(xs[, , j], TRUE)))
out$own_p <- list(P = geomorph:::pval(c(obs$sigma.d.ratio, apply(sim_own, 1, max))),
                  pair_P = sapply(1:ncol(cmb), function(c) geomorph:::pval(c(obs_pair[c], sim_own[, c]))))
# exact pieces on a subset of the cells: makePD and the statistic on R's simulated data
sub <- readLines(file.path(d, "parts_sub_cols.txt")); Ps <- P[, sub]; gs <- factor(readLines(file.path(d, "parts_sub_gp.txt")), levels = levels(gp))
ps <- geomorph:::phylo.mat(Ps, tr); os <- geomorph:::sigma.d.multi(Ps, ps$invC, ps$D.mat, gs, TRUE)
Rs <- os$R; diag(Rs) <- os$rate.global; R0 <- geomorph:::makePD(Rs)
write.csv(R0, file.path(d, "R_makePD_sub.csv"), row.names = FALSE)
x3 <- simplify2array(geomorph:::sim.char.BM(tr, R0, nsim = 3, seed = 7))
for (j in 1:3) write.csv(x3[, , j], file.path(d, sprintf("R_sim_sub_%d.csv", j)))
gi <- lapply(1:nlevels(gs), function(j) which(as.integer(gs) == j))
Xadj <- diag(N) - crossprod(matrix(1, N, N), ps$invC) / sum(ps$invC); Pts <- ps$D.mat %*% Xadj
out$sim_sub_ratios <- lapply(1:3, function(j) geomorph:::fast.sigma.d.multi(x3[, , j], Pts, TRUE, gi, nlevels(gs), combn(nlevels(gs), 2), N, ncol(Ps)))
out$sub_rates <- os$rate.gps
ks <- list()
for (nm in levels(gp)) { ph <- physignal(P[, g == which(levels(gp) == nm)], tr, iter = 99, print.progress = FALSE); ks[[nm]] <- ph$phy.signal }
out$Kmult <- ks
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), file.path(d, "R_results.json"))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", default="Rscript"); ap.add_argument("--rlib", default="")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--tree", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--traits", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--cell_geometry", action="append", required=True, metavar="NAME=JSON")
    ap.add_argument("--iter", type=int, default=999, help="simulations of the rate test (R and Descriptron)")
    ap.add_argument("--sub_cells", type=int, default=6, help="every n-th complete cell for the exact makePD check")
    ap.add_argument("--k", default="auto", help="pattern PCs for multirateBM (as --model_pcs; auto = up to 10)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dp = load_dp(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    root, _ = dp.load_tree(a.tree)
    gmap = dp.load_group_map(a.groups)
    geoms = dict(x.split("=", 1) for x in a.cell_geometry)
    sets = []
    for spec in a.traits:
        nm, path = spec.split("=", 1)
        geom = dp.load_cell_geometry(geoms[nm])
        sp, cols, M, _ = dp.load_traits(path, None, None, None, gmap, keep_missing=True, blank=geom["blank"])
        sets.append((nm, [dp._norm_name(s) for s in sp], cols, M))
    names = {t.name for t in dp.tips_of(root)}
    common = set.intersection(*[set(s[1]) for s in sets]) & names
    tree = dp.prune(dp._copy_tree(root), common); tips = dp.tips_of(tree); ns = len(tips)
    nw = dp.to_newick(tree); (out / "tree.nwk").write_text(nw if nw.rstrip().endswith(";") else nw + ";")
    labs, featm = [], None
    for k_, (nm, sp, cols, M) in enumerate(sets):
        cells, cix = dp.colour_cell_index(cols)
        lab = M[[sp.index(t.name) for t in tips]][:, cix]
        comp = np.isfinite(lab[..., 0]).all(0)
        labs.append((nm, [c for c, ok in zip(cells, comp) if ok], lab[:, comp]))
        if k_ == 0:
            comp_cols = ~np.isnan(M).any(axis=0)
            featm = (M[[sp.index(t.name) for t in tips]][:, comp_cols], [c for c, ok in zip(cols, comp_cols) if ok])
    # 1. set 1: complete cells, Lab
    nm1, cells1, lab1 = labs[0]
    with open(out / "set1_lab.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["species"] + [f"{c}_{ch}" for c in cells1 for ch in "Lab"])
        for i, t in enumerate(tips):
            w.writerow([t.name] + [repr(float(v)) for v in lab1[i].ravel()])
    F, fcols = featm
    keep = [j for j, c in enumerate(fcols) if not __import__("re").search(r"centroid|csize", c, __import__("re").I)]
    F = F[:, keep]; fcols = [fcols[j] for j in keep]
    with open(out / "set1_features.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["species"] + fcols)
        for i, t in enumerate(tips):
            w.writerow([t.name] + [repr(float(v)) for v in F[i]])
    S = {"raw": F, "idx": list(range(ns))}
    sc, ev, _ = dp.pattern_pc_matrix(S)
    k = dp.n_axes(ev, a.k, 10, ns)
    (out / "k.txt").write_text(f"{k}\n")
    # 3. parts side by side
    pcols, gp = [], []
    for nm, cells, lab in labs:
        pcols += [f"{nm}:{c}_{ch}" for c in cells for ch in "Lab"]; gp += [nm] * (3 * len(cells))
    PX = np.hstack([lab.reshape(ns, -1) for _, _, lab in labs])
    with open(out / "parts_lab.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["species"] + pcols)
        for i, t in enumerate(tips):
            w.writerow([t.name] + [repr(float(v)) for v in PX[i]])
    (out / "parts_gp.txt").write_text("\n".join(gp) + "\n")
    (out / "parts_names.txt").write_text("\n".join(nm for nm, _, _ in labs) + "\n")
    sub = []
    for nm, cells, lab in labs:
        sub += [f"{nm}:{c}_{ch}" for c in cells[::a.sub_cells] for ch in "Lab"]
    (out / "parts_sub_cols.txt").write_text("\n".join(sub) + "\n")
    (out / "parts_sub_gp.txt").write_text("\n".join(s.split(":")[0] for s in sub) + "\n")
    (out / "r_side.R").write_text(R_CODE)
    r = subprocess.run([a.rscript, str(out / "r_side.R"), a.rlib, str(out), str(a.iter)], capture_output=True, text=True)
    if r.returncode:
        sys.exit("R failed:\n" + r.stderr[-3000:])
    RJ = json.load(open(out / "R_results.json"))
    key = lambda nd: nd.name if not nd.kids else "|".join(sorted(t.name for t in dp.tips_of(nd)))
    summ = {"R_versions": RJ["versions"], "species": ns, "sets": [nm for nm, _, _ in labs],
            "set1_complete_cells": len(cells1)}
    # 1
    nodes = dp.internal_nodes(tree)
    B = dp.bm_ancestral(tips, nodes, lab1.reshape(ns, -1))
    est = B["est"].reshape(len(nodes), len(cells1), 3)
    br = dp.branch_pattern_changes(tree, tips, nodes, est, lab1, np.ones((len(nodes), len(cells1)), bool))
    mine = {key(b["child"]): b for b in br}
    Rb = RJ["branches"]
    dd = [abs(mine[c]["mean"] - v) for c, v in zip(Rb["child"], Rb["dE00"])]
    drr = [abs(mine[c]["rate"] - v) / v for c, v in zip(Rb["child"], Rb["rate"])]
    dl = [abs(mine[c]["length"] - v) for c, v in zip(Rb["child"], Rb["length"])]
    summ["rate_de00_vs_fastAnc_farver"] = {"branches": len(dd), "max_abs_diff_dE00": float(max(dd)),
                                           "max_rel_diff_rate": float(max(drr)), "max_abs_diff_length": float(max(dl)),
                                           "R_dE00_range": [float(min(Rb["dE00"])), float(max(Rb["dE00"]))]}
    # 2
    bm = dp.pattern_bm_branch_rates(S, tree, tips, a.k, 10, 1.0)
    nodes_ = bm["nodes"]; ix = {key(n): i for i, n in enumerate(nodes_)}
    worst_node, worst_edge, sc_d, corr = 0.0, 0.0, 0.0, []
    mean_r = np.zeros(len(nodes_)); mean_d = np.zeros(len(nodes_))
    for i, m in enumerate(RJ["multirate"]):
        rs = dict(zip(m["tips"], m["scores"]))
        sgn = np.sign(np.dot([rs[t.name] for t in tips], sc[:, i]))
        sc_d = max(sc_d, float(np.abs(sgn * np.array([rs[t.name] for t in tips]) - sc[:, i]).max()))
        fit = dp.multirate_bm(tree, tips, sc[:, i], 1.0)
        a_ = np.log([fit["node_rate"][ix[n]] for n in m["node"]]); b_ = np.log(m["sig2"])
        worst_node = max(worst_node, float(np.abs(a_ - b_).max())); corr.append(float(np.corrcoef(a_, b_)[0, 1]))
        er = np.array([fit["edge_rate"][ix[c]] for c in m["edge_child"]])
        worst_edge = max(worst_edge, float(np.abs(np.log(er) - np.log(m["edge_rate"])).max()))
        for c, v in zip(m["edge_child"], m["edge_rate"]):
            mean_r[ix[c]] += v / len(RJ["multirate"])
    mine_mean = np.array([bm["rate"][dp.branch_key(nodes_[i])] for i in range(1, len(nodes_))])
    summ["bm_rate_vs_multirateBM"] = {"k_pcs": k, "max_abs_diff_pc_scores": sc_d, "max_abs_diff_log_node_rate": worst_node,
                                      "min_corr_log_node_rate": float(min(corr)), "max_abs_diff_log_edge_rate": worst_edge,
                                      "max_rel_diff_mean_edge_rate": float(np.abs(mine_mean / mean_r[1:] - 1).max()),
                                      "R_convergence": [m["convergence"] for m in RJ["multirate"]]}
    # 3
    Ys = [lab.reshape(ns, -1) for _, _, lab in labs]
    C = dp.vcv(tips, tips)
    T = dp.multi_evol_rates(Ys, C, a.iter, rng)
    Tg = dp.multi_evol_rates(Ys, C, a.iter, rng, geomorph_null=True)
    cm = RJ["cmer"]
    # R's pairwise order: combn over levels = (1,2),(1,3),...; Descriptron's pairs the same order
    summ["compare_multi_evol_rates"] = {
        "parts": [nm for nm, _, _ in labs], "n_variables": [Y.shape[1] for Y in Ys],
        "max_rel_diff_rates": float(np.abs(T["rates"] / np.array(cm["rates"]) - 1).max()),
        "rel_diff_ratio": float(abs(T["ratio"] / cm["ratio"] - 1)),
        "max_rel_diff_pair_ratio": float(np.abs(T["pair_ratio"] / np.array(cm["pair_ratio"]) - 1).max()),
        "rates": [float(v) for v in T["rates"]], "ratio": T["ratio"],
        "P_geomorph_R": cm["P"], "P_geomorph_null_Descriptron": float(Tg["P"]),
        "P_own_p_R_same_sims": RJ["own_p"]["P"], "P_Descriptron": float(T["P"]),
        "pair_P_geomorph_R": cm["pair_P"], "pair_P_geomorph_null_Descriptron": [float(v) for v in Tg["pair_P"]],
        "pair_P_own_p_R": RJ["own_p"]["pair_P"], "pair_P_Descriptron": [float(v) for v in T["pair_P"]],
        "simulations": a.iter, "R_seconds": cm["seconds"]}
    # exact sub-checks
    subcols = (out / "parts_sub_cols.txt").read_text().split()
    allcols = [f"{nm}:{c}_{ch}" for nm, cells, _ in labs for c in cells for ch in "Lab"]
    si = [allcols.index(c) for c in subcols]
    X = np.hstack(Ys)[:, si]; gsub = [c.split(":")[0] for c in subcols]
    names_ = [nm for nm, _, _ in labs]
    groups = [np.array([j for j, g in enumerate(gsub) if g == nm]) for nm in names_]
    Ci = np.linalg.inv(C)
    rates_sub = dp._subset_rates(X, Ci, groups)
    glob = float(dp._subset_rates(X, Ci, [np.arange(X.shape[1])])[0])
    w = Ci.sum(1); Xc = X - (w @ X) / w.sum(); Rm = Xc.T @ Ci @ Xc / ns; np.fill_diagonal(Rm, glob)
    R0 = dp.make_pd(Rm)
    R0r = np.loadtxt(out / "R_makePD_sub.csv", delimiter=",", skiprows=1)
    pr = []
    for j in range(1, 4):
        rows = list(csv.reader(open(out / f"R_sim_sub_{j}.csv")))[1:]
        order = [[r[0] for r in rows].index(t.name) for t in tips]
        Xs = np.array([[float(v) for v in r[1:]] for r in rows])[order]
        rr = dp._subset_rates(Xs, Ci, groups, X.shape[1])
        pairs = [(i, k_) for i in range(len(groups)) for k_ in range(i + 1, len(groups))]
        pr.append(float(np.abs(np.array([max(rr[i], rr[k_]) / min(rr[i], rr[k_]) for i, k_ in pairs])
                               / np.array(RJ["sim_sub_ratios"][j - 1]) - 1).max()))
    summ["exact_subset_checks"] = {"variables": X.shape[1],
                                   "max_rel_diff_rates": float(np.abs(rates_sub / np.array(RJ["sub_rates"]) - 1).max()),
                                   "makePD_max_abs_diff": float(np.abs(R0 - R0r).max()),
                                   "makePD_max_abs": float(np.abs(R0r).max()),
                                   "stat_on_R_sims_max_rel_diff": max(pr)}
    kd = {}
    for (nm, _, _), Y in zip(labs, Ys):
        kd[nm] = {"Descriptron": dp.kmult(Y, C), "geomorph_physignal": RJ["Kmult"][nm]}
    summ["Kmult_per_part"] = kd
    summ["Kmult_max_abs_diff"] = float(max(abs(v["Descriptron"] - v["geomorph_physignal"]) for v in kd.values()))
    (out / "validation_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
