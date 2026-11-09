#!/usr/bin/env python3
"""descriptron_phylo's colour-pattern trees, tanglegram and convergence tests against the reference R implementations.

Python writes the inputs (colour tables, distance matrices, the bootstrap's resampled cell indices); R (farver 2.1.2,
phangorn 2.12.1, ape 5.8.1, phytools 2.4.4, vegan 2.7.5, convevol 2.2.1, windex 2.1.0) simulates trees and data, and
computes; Descriptron computes the same things from the same inputs:

  1. CIEDE2000 against the 34 test pairs published by Sharma, Wu & Dalal (2005, Color Res. Appl. 30: 21, Table 1;
     embedded below, 4 decimals) and against farver::compare_colour(method = "cie2000") on 2000 random CIELAB pairs
  2. UPGMA and neighbour joining against phangorn::upgma and ape::nj on the same distance matrices: topology
     (rooted clades for UPGMA, unrooted bipartitions for NJ) and every branch length
  3. bootstrap support of the colour tree: R rebuilds the colour distances itself (per-cell CIEDE2000 with farver,
     mean over the cells present in both species) for the SAME resampled cell-index matrices Python used, builds the
     trees and counts clades with ape::prop.clades; the counts must be identical
  4. Mantel r (Pearson and Spearman) against vegan::mantel (r exact, P within Monte Carlo error)
  5. Robinson-Foulds distance against phangorn::RF.dist; tanglegram crossings against phytools::cophylo(rotate = TRUE)
     (Descriptron's must be no more)
  6. Stayton's C1-C4 (convevol::calcConv) and Ct1-Ct4 (convevol::calcConvCt with groups; for a two-species group,
     where convevol 2.2.1's calcConvCt(groups = ) stops with an error, without groups: the same single pair) exactly, and the significance
     of convSig / convSigCt within Monte Carlo error: a 40-tip tree with Brownian-motion data (negative control) and
     the same data with four distant species pulled to one phenotype (positive control), and the real tree
  7. the Wheatsheaf index against windex::windex (index and jackknife interval exact) and test.windex (P within Monte
     Carlo error), on the same data

  python validate_phylo_colour_convergence_vs_r.py --rscript Rscript --out-dir val_colour/ \\
      [--tree real.nwk --colour_table color_homology_features_X.csv --groups group_labels.csv]
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


# Sharma, Wu & Dalal (2005) Table 1: L1 a1 b1 L2 a2 b2 dE00
SHARMA = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0010, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0012, 7.2195),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0009, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0010, -2.4900, 4.8045),
    (50.0000, -0.0010, 2.4900, 50.0000, 0.0011, -2.4900, 4.7461),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.1736, 0.5854, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2972, 0.0000, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 1.8634, 0.5757, 1.0000),
    (50.0000, 2.5000, 0.0000, 50.0000, 3.2592, 0.3350, 1.0000),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (61.2901, 3.7196, -5.3901, 61.4292, 2.2480, -4.9620, 1.8731),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
    (2.0776, 0.0795, -1.1350, 0.9033, -0.0636, -0.5514, 0.9082),
]

R_CODE = r'''
args <- commandArgs(TRUE); d <- args[1]; NS <- as.integer(args[2]); NB <- as.integer(args[3]); NP <- as.integer(args[4])
suppressMessages({library(ape); library(phytools); library(phangorn); library(vegan); library(convevol); library(windex)
                  library(farver); library(jsonlite); library(parallel)})
set.seed(20261106)
out <- list(versions = sapply(c("ape", "phytools", "phangorn", "vegan", "convevol", "windex", "farver"),
                              function(p) as.character(packageVersion(p))))
cores <- max(1, min(16, detectCores() - 2))
# 1. farver CIEDE2000
lp <- as.matrix(read.csv(file.path(d, "lab_pairs.csv")))
out$farver <- sapply(seq_len(nrow(lp)), function(i)
  compare_colour(lp[i, 1:3, drop = FALSE], lp[i, 4:6, drop = FALSE], from_space = "lab", method = "cie2000")[1, 1])
# 2. UPGMA / NJ
for (f in list.files(d, "^dm_.*[.]csv$")) {
  D <- as.matrix(read.csv(file.path(d, f), row.names = 1, check.names = FALSE)); tag <- sub("[.]csv$", "", f)
  write.tree(phangorn::upgma(as.dist(D)), file.path(d, paste0("R_upgma_", tag, ".nwk")), digits = 17)
  write.tree(ape::nj(as.dist(D)), file.path(d, paste0("R_nj_", tag, ".nwk")), digits = 17)
}
# 3. bootstrap from the same resampled cells, with R's own colour distances
desc_tips <- function(tr, nd) { g <- getDescendants(tr, nd); paste(sort(tr$tip.label[g[g <= Ntip(tr)]]), collapse = ",") }
Dreal <- NULL
for (tag in gsub("^species_|[.]txt$", "", list.files(d, "^species_.*[.]txt$"))) {
  sp <- readLines(file.path(d, paste0("species_", tag, ".txt"))); cells <- readLines(file.path(d, paste0("cells_", tag, ".txt")))
  L <- read.csv(file.path(d, paste0("lab_", tag, ".csv")))
  n <- length(sp); nc <- length(cells); A <- array(NA_real_, c(nc, n, n))
  for (k in seq_len(nc)) {
    s <- L[L$cell == cells[k], ]; m <- as.matrix(s[match(sp, s$species), c("L", "a", "b")])
    ok <- complete.cases(m); M <- matrix(NA_real_, n, n)
    M[ok, ok] <- compare_colour(m[ok, , drop = FALSE], m[ok, , drop = FALSE], from_space = "lab", method = "cie2000")
    A[k, , ] <- M
  }
  dmean <- function(ix) { S <- matrix(0, n, n); C <- matrix(0, n, n)
    for (k in ix) { M <- A[k, , ]; ok <- !is.na(M); S[ok] <- S[ok] + M[ok]; C <- C + ok }
    D <- S / C; diag(D) <- 0; dimnames(D) <- list(sp, sp); D }
  D0 <- dmean(seq_len(nc)); write.csv(D0, file.path(d, paste0("R_D_", tag, ".csv")))
  if (tag == "real") Dreal <- D0
  idx <- as.matrix(read.csv(file.path(d, paste0("idx_", tag, ".csv")), header = FALSE))
  for (meth in c("upgma", "nj")) {
    build <- if (meth == "upgma") function(D) phangorn::upgma(as.dist(D)) else function(D) ape::nj(as.dist(D))
    ref <- build(D0)
    trees <- mclapply(seq_len(nrow(idx)), function(b) build(dmean(idx[b, ])), mc.cores = cores); class(trees) <- "multiPhylo"
    cnt <- prop.clades(ref, trees, rooted = (meth == "upgma"))
    nodes <- Ntip(ref) + seq_len(ref$Nnode)
    write.csv(data.frame(clade = sapply(nodes, function(nd) desc_tips(ref, nd)), count = cnt),
              file.path(d, paste0("R_support_", meth, "_", tag, ".csv")), row.names = FALSE)
  }
}
# 4./5. Mantel, RF, cophylo
ord <- function(t) { t <- reorder(t, "cladewise"); t$tip.label[t$edge[t$edge[, 2] <= Ntip(t), 2]] }
ncross <- function(o1, o2) { p <- match(o1, o2); s <- 0; for (i in 1:(length(p) - 1)) s <- s + sum(p[i] > p[(i + 1):length(p)]); s }
tm <- pbtree(n = 40, scale = 1); tm$tip.label <- paste0("m", 1:40); tm <- read.tree(text = write.tree(tm))
xm <- fastBM(tm, nsim = 3); Dm <- as.matrix(dist(xm)); Pm <- cophenetic(tm)[rownames(Dm), rownames(Dm)]
write.tree(tm, file.path(d, "mantel_tree.nwk"), digits = 17); write.csv(Dm, file.path(d, "mantel_D.csv"))
tie <- function(M) { x <- M[lower.tri(M)]; o <- order(x); xs <- x[o]; tol <- 1e-8 * max(abs(x))
  if (length(xs) > 1) for (k in 2:length(xs)) if (xs[k] - xs[k - 1] <= tol || abs(x[o[k]] - xs[k - 1]) <= tol) xs[k] <- xs[k - 1]
  x[o] <- xs; M[lower.tri(M)] <- x; M[upper.tri(M)] <- t(M)[upper.tri(M)]; M }
mm <- list(sim = list(P = Pm, D = Dm))
if (!is.null(Dreal)) { tr <- read.tree(file.path(d, "real_tree.nwk")); mm$real <- list(P = cophenetic(tr)[rownames(Dreal), rownames(Dreal)], D = Dreal) }
out$mantel <- lapply(mm, function(x) {
  a <- mantel(as.dist(x$P), as.dist(x$D), method = "pearson", permutations = NP)
  # ties: distances equal on the tree must be equal numbers (Descriptron's _tie: sorted neighbours within
  # 1e-8 x the largest take the first value of their run)
  b <- mantel(as.dist(tie(x$P)), as.dist(tie(x$D)), method = "spearman", permutations = NP)
  list(pearson_r = a$statistic, pearson_P = a$signif, spearman_r = b$statistic, spearman_P = b$signif) })
pairs <- list(sim = list(tm, phangorn::upgma(as.dist(Dm))))
if (!is.null(Dreal)) pairs$real <- list(tr, phangorn::upgma(as.dist(Dreal)))
for (k in 1:10) { a <- rtree(30); b <- rtree(30, tip.label = sample(a$tip.label)); pairs[[paste0("random", k)]] <- list(a, b) }
out$rf_cophylo <- lapply(names(pairs), function(nm) {
  a <- pairs[[nm]][[1]]; b <- pairs[[nm]][[2]]
  write.tree(a, file.path(d, paste0("pair_", nm, "_1.nwk")), digits = 17); write.tree(b, file.path(d, paste0("pair_", nm, "_2.nwk")), digits = 17)
  invisible(capture.output(cp <- cophylo(a, b, rotate = TRUE)))
  list(name = nm, RF = as.numeric(RF.dist(a, b)), crossings_unrotated = ncross(ord(a), ord(b)),
       crossings_cophylo = ncross(ord(cp$trees[[1]]), ord(cp$trees[[2]]))) })
# 6./7. convevol and windex
tc <- pbtree(n = 40, scale = 1); tc$tip.label <- paste0("c", 1:40); tc <- read.tree(text = write.tree(tc, digits = 17))
X <- fastBM(tc, nsim = 3); colnames(X) <- paste0("x", 1:3); X <- X[tc$tip.label, ]
o <- ord(tc); focal <- o[c(2, 13, 26, 38)]                   # spread over the tree
Xp <- X; Xp[focal, ] <- matrix(c(4, -3, 3), 4, 3, byrow = TRUE) + rnorm(12, sd = 0.1)
write.tree(tc, file.path(d, "conv_tree.nwk"), digits = 17)
cases <- list(neg = list(tree = tc, X = X, focal = focal, groups = setNames(seq_along(focal), focal)),
              pos = list(tree = tc, X = Xp, focal = focal, groups = setNames(seq_along(focal), focal)))
for (nm in c("neg", "pos")) write.csv(cases[[nm]]$X, file.path(d, paste0("conv_X_", nm, ".csv")))
writeLines(focal, file.path(d, "conv_focal.txt"))
if (file.exists(file.path(d, "real_groups.csv"))) {
  tr <- read.tree(file.path(d, "real_tree.nwk")); Xr <- as.matrix(read.csv(file.path(d, "real_pcs.csv"), row.names = 1))
  Xr <- Xr[tr$tip.label, , drop = FALSE]; G <- read.csv(file.path(d, "real_groups.csv"))
  for (g in unique(G$group)) { s <- G[G$group == g, ]
    cases[[paste0("real_", g)]] <- list(tree = tr, X = Xr, focal = s$species, groups = setNames(as.character(s$lineage), s$species)) }
}
out$conv <- mclapply(names(cases), function(nm) {
  cs <- cases[[nm]]; tree <- cs$tree; Xc <- cs$X
  cc <- calcConv(tree, Xc, cs$focal)
  # convevol 2.2.1's calcConvCt(groups = ) fails for two species (its pair matrix drops to a vector): their one pair
  # is the same with groups = NULL
  gg <- if (length(cs$focal) > 2) cs$groups else NULL
  ct <- if (length(unique(cs$groups)) > 1) calcConvCt(tree, Xc, cs$focal, groups = gg)$mean else rep(NA, 4)
  sg <- convSig(tree, Xc, cs$focal, nsim = NS)
  st <- if (length(unique(cs$groups)) > 1) { invisible(capture.output(z <- convSigCt(tree, Xc, cs$focal, groups = gg, nsim = NS))); z$pvals[, 2] } else rep(NA, 4)
  # Ct4 with convevol's edge indexing corrected (C4.edge instead of edge), for Descriptron's default Ct4
  ctfix <- if (length(unique(cs$groups)) > 1) { e <- environment(calcConvCt); old <- get("calcCsCt", envir = e)
    fx <- old; body(fx) <- parse(text = gsub("lapply(edge, function(y) y[x,", "lapply(C4.edge, function(y) y[x,",
                                            deparse(body(old)), fixed = TRUE))[[1]]
    assignInNamespace("calcCsCt", fx, "convevol"); on.exit(assignInNamespace("calcCsCt", old, "convevol"))
    r <- calcConvCt(tree, Xc, cs$focal, groups = gg)$mean; assignInNamespace("calcCsCt", old, "convevol"); r } else rep(NA, 4)
  dat <- data.frame(species = tree$tip.label, focal = as.integer(tree$tip.label %in% cs$focal), Xc[tree$tip.label, , drop = FALSE])
  tv <- colnames(Xc)
  w <- windex(dat, tree, traits = tv, focal = dat$focal)
  invisible(capture.output(tw <- test.windex(dat, tree, traits = tv, focal = dat$focal, reps = NS, plot = FALSE)))
  list(name = nm, C = unname(cc), P_C = unname(sg[, 2]), Ct = unname(ct), P_Ct = unname(st), Ct_fixed = unname(ctfix),
       w = w$w, low95 = w$low95, up95 = w$up95, P_w = tw$P) }, mc.cores = cores, mc.set.seed = TRUE)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), file.path(d, "R_results.json"))
'''


def mc_ok(p1, p2, n1, n2):
    """|P1 - P2| against its Monte Carlo standard error (two independent estimates); z, and whether z <= 4 or the
    difference is within 2 simulation steps."""
    p = (p1 + p2) / 2
    se = np.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    z = abs(p1 - p2) / se
    return float(z), bool(z <= 4 or abs(p1 - p2) <= 2 / min(n1, n2))


def edge_lengths(root, dp, rooted):
    """branch length per edge, keyed by the clade below it (rooted) or by the bipartition (unrooted)."""
    names = frozenset(t.name for t in dp.tips_of(root))
    out = {}
    stack = [root]
    while stack:
        n = stack.pop(); stack += n.kids
        if n.parent is None:
            continue
        s = frozenset(t.name for t in dp.tips_of(n))
        if not rooted and min(names) in s:
            s = names - s
        out[s] = out.get(s, 0.0) + n.length          # NJ's unrooted root edges are summed if ape roots elsewhere
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", default="Rscript"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tree", default=None, help="real tree (ultrametric) for the real-data comparisons")
    ap.add_argument("--colour_table", default=None, help="a colour-homology table (color_homology_features_*.csv)")
    ap.add_argument("--groups", default=None, help="group labels CSV giving each specimen's species")
    ap.add_argument("--nsim", type=int, default=500, help="simulations / resamplings on both sides (convevol, windex)")
    ap.add_argument("--nboot", type=int, default=1000, help="bootstrap replicates of the colour tree")
    ap.add_argument("--nperm", type=int, default=9999, help="Mantel permutations")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--reuse_r", action="store_true", help="compare against R_results.json already in --out-dir")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dp = load_dp()
    rng = np.random.default_rng(a.seed)
    summary = {}

    # ------------------------------------------------------------------ inputs
    S = np.array(SHARMA)
    de = dp.ciede2000(S[:, :3], S[:, 3:6])
    summary["ciede2000_sharma2005"] = {"pairs": len(S), "max_abs_diff_vs_published": float(np.abs(de - S[:, 6]).max()),
                                       "pass_1e-4": bool(np.abs(de - S[:, 6]).max() < 1e-4)}
    lp = np.column_stack([rng.uniform(0, 100, 2000), rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000),
                          rng.uniform(0, 100, 2000), rng.uniform(-100, 100, 2000), rng.uniform(-100, 100, 2000)])
    lp[:200, 3:6] = lp[:200, :3] + rng.normal(0, 2, (200, 3))               # near pairs
    lp[:, [0, 3]] = np.clip(lp[:, [0, 3]], 0, 100)                           # farver clamps L* to [0, 100]
    lp[200:300, 1:3] = rng.normal(0, 0.5, (100, 2))                          # near-neutral
    lp = np.vstack([lp, S[:, :6]])
    np.savetxt(out / "lab_pairs.csv", lp, delimiter=",", header="L1,a1,b1,L2,a2,b2", comments="", fmt="%.17g")

    def write_dm(tag, D, names):
        with open(out / f"dm_{tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow([""] + names)
            for i, nm in enumerate(names):
                w.writerow([nm] + [repr(float(v)) for v in D[i]])
    dms = {}
    for k in range(5):
        P = rng.normal(size=(30, 5)); names = [f"s{i}" for i in range(30)]
        D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); dms[f"rand{k}"] = (D, names); write_dm(f"rand{k}", D, names)

    # simulated colour pattern (40 species, 60 cells, 5% of species x cells missing) and the real table
    labsets = {}
    ns, nc = 40, 60
    base = np.column_stack([rng.uniform(20, 70, nc), rng.normal(5, 5, nc), rng.normal(10, 8, nc)])
    groupfx = rng.normal(0, 6, (5, nc, 3)); g = rng.integers(0, 5, ns)
    lab = base[None] + groupfx[g] + rng.normal(0, 3, (ns, nc, 3))
    lab[rng.random((ns, nc)) < 0.05] = np.nan
    labsets["sim"] = ([f"sp{i:02d}" for i in range(ns)], [f"r{1 + i // 10}c{1 + i % 10}" for i in range(nc)], lab)
    real = None
    if a.tree and a.colour_table and a.groups:
        root, _ = dp.load_tree(a.tree)
        gmap = dp.load_group_map(a.groups)
        sp, cols, M, counts = dp.load_traits(a.colour_table, None, None, None, gmap, keep_missing=True)
        sp = [dp._norm_name(s) for s in sp]
        tipn = {t.name for t in dp.tips_of(root)}
        common = [s for s in sp if s in tipn]
        tree = dp.prune(dp._copy_tree(root), set(common)); tips = dp.tips_of(tree); names = [t.name for t in tips]
        cells, cix = dp.colour_cell_index(cols, dp.COLOUR_CELLS_DEFAULT)
        lab_r = M[[sp.index(s) for s in names]][:, cix]
        labsets["real"] = (names, cells, lab_r)
        (out / "real_tree.nwk").write_text(dp.to_newick(tree, fmt="{!r}") + ";\n")
        real = {"tree": tree, "names": names, "cells": cells, "lab": lab_r}
    idxs, Dcs = {}, {}
    for tag, (names, cells, lab) in labsets.items():
        (out / f"species_{tag}.txt").write_text("\n".join(names) + "\n")
        (out / f"cells_{tag}.txt").write_text("\n".join(cells) + "\n")
        with open(out / f"lab_{tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["species", "cell", "L", "a", "b"])
            for i, s in enumerate(names):
                for k, c in enumerate(cells):
                    w.writerow([s, c] + (["NA"] * 3 if not np.isfinite(lab[i, k]).all() else [repr(float(v)) for v in lab[i, k]]))
        Dcs[tag] = dp.per_cell_distances(lab, "de2000")
        idxs[tag] = rng.integers(0, len(cells), size=(a.nboot, len(cells)))
        np.savetxt(out / f"idx_{tag}.csv", idxs[tag] + 1, delimiter=",", fmt="%d")
        D = dp.colour_distance(Dcs[tag]); dms[f"colour_{tag}"] = (D, names); write_dm(f"colour_{tag}", D, names)

    # real-data convergence inputs: the PCs and candidate groups the program would use
    if real is not None:
        tree, names, lab_r = real["tree"], real["names"], real["lab"]
        Dc = Dcs["real"]
        ctree, sup, _, _ = dp.bootstrap_support(Dc, names, "upgma", a.nboot, np.random.default_rng(1))
        cands = dp.candidate_groups(ctree, sup, tree, 0.7, len(names) // 2)
        ok = np.isfinite(lab_r).all(axis=(0, 2)); Xc = lab_r[:, ok, :].reshape(len(names), -1)
        Z = Xc - Xc.mean(0); U, sv, Vt = np.linalg.svd(Z, full_matrices=False); ev = sv ** 2 / (sv ** 2).sum()
        k = max(1, min(int(np.searchsorted(np.cumsum(ev), 0.9 - 1e-12) + 1), 10, len(names) - 2))
        Xr = Z @ Vt[:k].T
        with open(out / "real_pcs.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["species"] + [f"PC{i + 1}" for i in range(k)])
            for i, s in enumerate(names):
                w.writerow([s] + [repr(float(v)) for v in Xr[i]])
        with open(out / "real_groups.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["group", "species", "lineage"])
            for gi, c in enumerate(cands):
                for s in c["members"]:
                    w.writerow([f"G{gi + 1}", s, c["lineages"][s]])
        real.update(X=Xr, cands=cands, k=k)

    # ------------------------------------------------------------------ R
    (out / "r_side.R").write_text(R_CODE)
    if not (a.reuse_r and (out / "R_results.json").exists()):
        subprocess.run([a.rscript, str(out / "r_side.R"), str(out), str(a.nsim), str(a.nboot), str(a.nperm)], check=True)
    R = json.loads((out / "R_results.json").read_text())
    summary["R_versions"] = R["versions"]

    # 1. farver
    fv = np.array(R["farver"], float); ours = dp.ciede2000(lp[:, :3], lp[:, 3:6])
    summary["ciede2000_vs_farver"] = {"pairs": len(lp), "max_abs_diff": float(np.abs(ours - fv).max()),
                                      "farver_vs_sharma_max_abs_diff": float(np.abs(fv[-len(S):] - S[:, 6]).max())}

    # 2. UPGMA / NJ
    worst = {"upgma": [0, 0.0], "nj": [0, 0.0]}
    for tag, (D, names) in dms.items():
        for meth, rooted in (("upgma", True), ("nj", False)):
            rt = dp.parse_newick((out / f"R_{meth}_dm_{tag}.nwk").read_text())
            pt = dp.upgma(D, names) if meth == "upgma" else dp.nj(D, names)
            rf = len(dp.tree_keys(rt, rooted) ^ dp.tree_keys(pt, rooted))
            el_r, el_p = edge_lengths(rt, dp, rooted), edge_lengths(pt, dp, rooted)
            dl = max(abs(el_r[k] - el_p.get(k, np.inf)) for k in el_r) if set(el_r) == set(el_p) else np.inf
            worst[meth][0] = max(worst[meth][0], rf); worst[meth][1] = max(worst[meth][1], dl)
    summary["upgma_vs_phangorn"] = {"matrices": len(dms), "max_RF_rooted_clades": worst["upgma"][0],
                                    "max_abs_diff_branch_length": worst["upgma"][1]}
    summary["nj_vs_ape"] = {"matrices": len(dms), "max_RF_unrooted": worst["nj"][0],
                            "max_abs_diff_branch_length": worst["nj"][1]}

    # 3. bootstrap counts
    boot = {}
    for tag, (names, cells, lab) in labsets.items():
        RD = np.array([[float(v) for v in r[1:]] for r in list(csv.reader(open(out / f"R_D_{tag}.csv")))[1:]])
        dD = float(np.abs(RD - dp.colour_distance(Dcs[tag])).max())
        for meth in ("upgma", "nj"):
            ref = dp.build_colour_tree(dp.colour_distance(Dcs[tag]), names, meth)
            cnt = dp.support_counts(Dcs[tag], names, meth, idxs[tag], ref); cnt.pop("__filled__")
            ours = {",".join(sorted(s)): cnt[id(n)] for n, s in dp.clade_sets(ref)}
            rc = {r["clade"]: int(r["count"]) for r in csv.DictReader(open(out / f"R_support_{meth}_{tag}.csv"))
                  if r["count"] not in ("NA", "")}
            shared = [k for k in ours if k in rc]
            diff = [abs(ours[k] - rc[k]) for k in shared]
            boot[f"{tag}_{meth}"] = {"replicates": a.nboot, "nodes_compared": len(shared), "nodes": len(ours),
                                     "max_abs_count_diff": int(max(diff)) if diff else None,
                                     "colour_distance_max_abs_diff_vs_R": dD}
    summary["bootstrap_support_vs_ape_prop_clades"] = boot

    # 4. Mantel
    mres = {}
    tm = dp.parse_newick((out / "mantel_tree.nwk").read_text())
    Dm_rows = list(csv.reader(open(out / "mantel_D.csv")))
    mn = Dm_rows[0][1:]; Dm = np.array([[float(v) for v in r[1:]] for r in Dm_rows[1:]])
    tipm = {t.name: t for t in dp.tips_of(tm)}
    sets = {"sim": (dp.patristic([tipm[s] for s in mn]), Dm)}
    if real is not None:
        tipsr = {t.name: t for t in dp.tips_of(real["tree"])}
        sets["real"] = (dp.patristic([tipsr[s] for s in real["names"]]), dp.colour_distance(Dcs["real"]))
    for nm, (P, D) in sets.items():
        r = {}
        for meth in ("pearson", "spearman"):
            m = dp.mantel(P, D, meth, a.nperm, rng)
            rr, rp = float(R["mantel"][nm][f"{meth}_r"]), float(R["mantel"][nm][f"{meth}_P"])
            z, ok = mc_ok(m["P"], rp, a.nperm, a.nperm)
            r[meth] = {"r_descriptron": m["r"], "r_vegan": rr, "abs_diff_r": abs(m["r"] - rr),
                       "P_descriptron": m["P"], "P_vegan": rp, "P_z": z, "P_within_mc": ok}
        mres[nm] = r
    summary["mantel_vs_vegan"] = mres

    # 5. RF and crossings
    rows = []
    for x in R["rf_cophylo"]:
        t1 = dp.parse_newick((out / f"pair_{x['name']}_1.nwk").read_text())
        t2 = dp.parse_newick((out / f"pair_{x['name']}_2.nwk").read_text())
        rf = dp.rf_distance(t1, t2)["RF"]
        c = dp.untangle(t1, t2)
        rows.append({"pair": x["name"], "RF_descriptron": rf, "RF_phangorn": x["RF"], "crossings_descriptron": c,
                     "crossings_cophylo": x["crossings_cophylo"], "crossings_unrotated": x["crossings_unrotated"]})
    summary["rf_vs_phangorn"] = {"pairs": len(rows), "max_abs_diff": max(abs(r["RF_descriptron"] - r["RF_phangorn"]) for r in rows)}
    summary["crossings_vs_cophylo"] = {"pairs": len(rows),
                                       "descriptron_never_more": all(r["crossings_descriptron"] <= r["crossings_cophylo"] for r in rows),
                                       "per_pair": [{k: r[k] for k in ("pair", "crossings_descriptron", "crossings_cophylo",
                                                                       "crossings_unrotated")} for r in rows]}

    # 6./7. convergence
    tc = dp.parse_newick((out / "conv_tree.nwk").read_text())
    focal = (out / "conv_focal.txt").read_text().split()
    conv_rows = []
    for x in R["conv"]:
        nm = x["name"]
        if nm in ("neg", "pos"):
            tree = tc
            Xd = {r[0]: [float(v) for v in r[1:]] for r in list(csv.reader(open(out / f"conv_X_{nm}.csv")))[1:]}
            fo, lin = focal, {s: str(i) for i, s in enumerate(focal)}
        else:
            tree = real["tree"]
            Xd = {s: list(real["X"][i]) for i, s in enumerate(real["names"])}
            gi = int(nm.split("_G")[1]) - 1
            fo, lin = real["cands"][gi]["members"], real["cands"][gi]["lineages"]
        T = dp.ConvTree(tree)
        X = np.array([Xd[s] for s in T.names])
        multi = len(set(lin.values())) > 1
        ct = dp.convergence_test(T, X, fo, lin if multi else None, a.nsim, rng, ct4="convevol")
        ctd = dp.conv_stats(T, T.all_values(X), dp._pairs(T, fo, lin if multi else None), "clade")[1]
        wi = dp.test_wheatsheaf(X, T.C, np.array([s in fo for s in T.names]), a.nsim, rng)
        r = {"case": nm, "n_focal": len(fo), "pairs_C": ct["n_pairs_C"], "pairs_Ct": ct["n_pairs_Ct"] if multi else 0}
        rc = np.array(x["C"], float)
        r["max_abs_diff_C"] = float(np.abs(ct["C"] - rc).max())
        r["max_rel_diff_C"] = float((np.abs(ct["C"] - rc) / np.maximum(np.abs(rc), 1e-300)).max())
        zc = [mc_ok(ct["P_C"][i], float(x["P_C"][i]), a.nsim, a.nsim) for i in range(4)]
        r["P_C_descriptron"], r["P_C_convevol"] = [float(v) for v in ct["P_C"]], [float(v) for v in x["P_C"]]
        r["P_C_max_z"], r["P_C_within_mc"] = max(z for z, _ in zc), all(o for _, o in zc)
        if multi:
            rt = np.array(x["Ct"], float)
            fin = np.isfinite(rt) & np.isfinite(ct["Ct"])
            r["Ct_same_nonfinite"] = bool(np.array_equal(np.isfinite(rt), np.isfinite(ct["Ct"])))
            r["max_abs_diff_Ct"] = float(np.abs(ct["Ct"][fin] - rt[fin]).max()) if fin.any() else None
            r["max_rel_diff_Ct"] = float((np.abs(ct["Ct"][fin] - rt[fin]) / np.maximum(np.abs(rt[fin]), 1e-300)).max()) if fin.any() else None
            zt = [mc_ok(ct["P_Ct"][i], float(x["P_Ct"][i]), a.nsim, a.nsim) for i in range(4)]
            r["P_Ct_descriptron"], r["P_Ct_convevol"] = [float(v) for v in ct["P_Ct"]], [float(v) for v in x["P_Ct"]]
            r["P_Ct_max_z"], r["P_Ct_within_mc"] = max(z for z, _ in zt), all(o for _, o in zt)
            rfx = np.array(x["Ct_fixed"], float)
            r["Ct4_default_descriptron"], r["Ct4_convevol_corrected"] = float(ctd[3]), float(rfx[3])
            r["max_abs_diff_Ct_default_vs_corrected_convevol"] = float(np.abs(ctd - rfx).max())
        r["C_descriptron"], r["C_convevol"] = [float(v) for v in ct["C"]], [float(v) for v in rc]
        if multi:
            r["Ct_descriptron"], r["Ct_convevol"] = [float(v) for v in ct["Ct"]], [float(v) for v in rt]
        for k_, rk in (("w", "w"), ("low95", "low95"), ("up95", "up95")):
            rv = float("inf") if x[rk] in ("Inf", None) and k_ == "up95" else float(x[rk])
            r[f"wheatsheaf_{k_}_descriptron"], r[f"wheatsheaf_{k_}_windex"] = wi[k_], rv
        r["wheatsheaf_max_abs_diff"] = max(abs(wi[k_] - r[f"wheatsheaf_{k_}_windex"]) if np.isfinite(wi[k_])
                                           else (0.0 if wi[k_] == r[f"wheatsheaf_{k_}_windex"] else np.inf)
                                           for k_ in ("w", "low95", "up95"))
        z, ok = mc_ok(wi["P"], float(x["P_w"]), a.nsim, a.nsim)
        r.update(P_wheatsheaf_descriptron=wi["P"], P_wheatsheaf_windex=float(x["P_w"]), P_wheatsheaf_z=z,
                 P_wheatsheaf_within_mc=ok)
        conv_rows.append(r)
    summary["convergence_vs_convevol_windex"] = {
        "cases": len(conv_rows), "simulations_each_side": a.nsim,
        "max_abs_diff_C1_C4": max(r["max_abs_diff_C"] for r in conv_rows),
        "max_rel_diff_C1_C4": max(r["max_rel_diff_C"] for r in conv_rows),
        "max_abs_diff_Ct1_Ct4": max((r["max_abs_diff_Ct"] for r in conv_rows if r.get("max_abs_diff_Ct") is not None), default=None),
        "max_rel_diff_Ct1_Ct4": max((r["max_rel_diff_Ct"] for r in conv_rows if r.get("max_rel_diff_Ct") is not None), default=None),
        "Ct_nonfinite_agree": all(r.get("Ct_same_nonfinite", True) for r in conv_rows),
        "note_Ct": "Ct1-Ct4 compared with ct4='convevol' (convevol 2.2.1's edge indexing reproduced); Descriptron's "
                   "default Ct4 (the clade's branches) compared with calcCsCt corrected to use C4.edge",
        "max_abs_diff_Ct_default_vs_corrected_convevol": max((r["max_abs_diff_Ct_default_vs_corrected_convevol"]
                                                             for r in conv_rows if "max_abs_diff_Ct_default_vs_corrected_convevol" in r), default=None),
        "max_abs_diff_wheatsheaf_w_and_interval": max(r["wheatsheaf_max_abs_diff"] for r in conv_rows),
        "P_C_all_within_mc": all(r["P_C_within_mc"] for r in conv_rows),
        "P_Ct_all_within_mc": all(r.get("P_Ct_within_mc", True) for r in conv_rows),
        "P_wheatsheaf_all_within_mc": all(r["P_wheatsheaf_within_mc"] for r in conv_rows),
        "max_P_z": max(max(r["P_C_max_z"], r.get("P_Ct_max_z", 0), r["P_wheatsheaf_z"]) for r in conv_rows),
        "per_case": conv_rows}
    if real is not None:
        summary["real_data"] = {"species": len(real["names"]), "cells": len(real["cells"]), "convergence_pcs": real["k"],
                                "candidate_groups": len(real["cands"])}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    brief = {k: v for k, v in summary.items() if k != "convergence_vs_convevol_windex"}
    brief["convergence"] = {k: v for k, v in summary["convergence_vs_convevol_windex"].items() if k != "per_case"}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
