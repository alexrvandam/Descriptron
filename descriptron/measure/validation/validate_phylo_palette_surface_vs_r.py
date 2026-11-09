#!/usr/bin/env python3
"""descriptron_phylo's palette distance, PCA distance, PCoA, OU-with-regimes likelihood and SURFACE against R.

Python writes the inputs (palettes, specimen-EMD matrix, resampled specimen and feature indices, features, trees,
regime paintings); R (emdist 0.3-3, ape, phangorn, stats, ouch 2.20, surface 0.6, geiger) computes; Descriptron
computes the same things from the same inputs:

  1. earth mover's distance against emdist::emd(cbind(w, L, a, b), ...) (Euclidean ground distance, weights
     normalised to 1 as validation/recolorize_v22/scripts/trees.R does) on 300 random palette pairs (1-15 colours)
     and on every pair of real specimen palettes. emdist works in single precision (its C code stores flows and
     costs as float): the comparison reports the difference in units of float32 spacing at R's value. Descriptron's
     exactness is checked separately against scipy.stats.wasserstein_distance on one-dimensional palettes (where the
     EMD has a closed form), to 1e-12.
  2. species palette distance = mean EMD over specimen pairs, as trees.R's species_mat: R's own (emdist) matrix,
     and trees.R's species_mat applied to Descriptron's specimen matrix (aggregation exact)
  3. principal coordinates against stats::cmdscale(eig = TRUE) and ape::pcoa (eigenvalues; coordinates up to sign)
  4. the PCA distance against prcomp(features) + dist on the first k PCs
  5. bootstrap support: palette_emd (specimens resampled within species) and pca (features resampled) rebuilt in R
     from the SAME resampled indices (species_mat / prcomp, phangorn::upgma, ape::prop.clades): counts identical
  6. the Hansen OU likelihood with several optima against ouch::hansen (log-likelihood, alpha, sigma^2, every
     optimum) for several regime paintings and starting values, simulated and real trees
  7. SURFACE against surface::runSurface: data simulated with planted convergent regimes, Brownian-motion data, and
     the first colour-pattern PCs of the real data: the AICc of every forward and backward step, the final shifts and
     regimes (as tip sets), and the final alpha, sigma^2 and optima per trait
  8. models on a colour-pattern PC: geiger::fitContinuous (BM, OU, EB, lambda, white) on PC1 of the real set

  python validate_phylo_palette_surface_vs_r.py --rscript Rscript --out-dir val_palette_surface/ \\
      --tree real.nwk --colour_table color_homology_features_X.csv --groups group_labels.csv \\
      --palettes 'color_extraction/X/*_palette.csv'
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
suppressMessages({library(emdist); library(ape); library(phangorn); library(ouch); library(surface); library(geiger)
                  library(phytools); library(jsonlite); library(parallel)})
set.seed(20261106)
cores <- max(1, min(12, detectCores() - 2))
out <- list(versions = sapply(c("emdist", "ape", "phangorn", "ouch", "surface", "geiger"),
                              function(p) as.character(packageVersion(p))))
# 1. EMD, random pairs: palettes in long format (pair, side, w, L, a, b)
rp <- read.csv(file.path(d, "emd_random.csv"))
out$emd_random <- sapply(sort(unique(rp$pair)), function(k) {
  A <- rp[rp$pair == k & rp$side == 1, ]; B <- rp[rp$pair == k & rp$side == 2, ]
  emd(as.matrix(cbind(A$w / sum(A$w), A$L, A$a, A$b)), as.matrix(cbind(B$w / sum(B$w), B$L, B$a, B$b))) })
# 1./2. real palettes, as trees.R
spec_emd <- function(pal) {
  ims <- unique(pal$image)
  P <- lapply(ims, function(i) { p <- pal[pal$image == i, ]; as.matrix(cbind(p$frac / sum(p$frac), p$L, p$a, p$b)) })
  n <- length(ims); D <- matrix(0, n, n, dimnames = list(ims, ims))
  for (i in 1:(n - 1)) for (j in (i + 1):n) D[i, j] <- D[j, i] <- emd(P[[i]], P[[j]])
  D }
species_mat <- function(D, sp, S) {
  M <- matrix(0, length(S), length(S), dimnames = list(S, S))
  for (i in seq_along(S)) for (j in seq_along(S)) if (i != j) {
    ii <- which(sp == S[i]); jj <- which(sp == S[j]); M[i, j] <- mean(D[ii, jj, drop = FALSE]) }
  M }
if (file.exists(file.path(d, "palettes_real.csv"))) {
  pal <- read.csv(file.path(d, "palettes_real.csv"))
  Dsp <- spec_emd(pal); write.csv(Dsp, file.path(d, "R_specimen_emd.csv"))
  slots <- read.csv(file.path(d, "slots.csv")); S <- readLines(file.path(d, "species.txt"))
  write.csv(species_mat(Dsp, slots$species, S), file.path(d, "R_species_emd_R_specimens.csv"))
  Dpy <- as.matrix(read.csv(file.path(d, "py_specimen_emd.csv"), header = FALSE))
  write.csv(species_mat(Dpy, slots$species, S), file.path(d, "R_species_emd_py_specimens.csv"))
  Dsp_species <- as.matrix(read.csv(file.path(d, "py_species_emd.csv"), header = FALSE)); dimnames(Dsp_species) <- list(S, S)
  # 3. PCoA
  cm <- cmdscale(as.dist(Dsp_species), k = nrow(Dsp_species) - 1, eig = TRUE)
  pc <- ape::pcoa(as.dist(Dsp_species))
  write.csv(cm$points, file.path(d, "R_cmdscale_points.csv"), row.names = FALSE)
  writeLines(format(cm$eig, digits = 17), file.path(d, "R_cmdscale_eig.txt"))
  write.csv(pc$vectors, file.path(d, "R_pcoa_vectors.csv"), row.names = FALSE)
  writeLines(format(pc$values$Eigenvalues, digits = 17), file.path(d, "R_pcoa_eig.txt"))
  # 5. bootstrap support, palette_emd, from the same resampled specimens
  idx <- as.matrix(read.csv(file.path(d, "idx_palette.csv"), header = FALSE))
  ref <- phangorn::upgma(as.dist(Dsp_species))
  trees <- mclapply(seq_len(nrow(idx)), function(b) { ix <- idx[b, ]
    M <- species_mat(Dpy[ix, ix], slots$species, S); phangorn::upgma(as.dist(M)) }, mc.cores = cores)
  class(trees) <- "multiPhylo"
  desc_tips <- function(tr, nd) { g <- getDescendants(tr, nd); paste(sort(tr$tip.label[g[g <= Ntip(tr)]]), collapse = ",") }
  nodes <- Ntip(ref) + seq_len(ref$Nnode)
  write.csv(data.frame(clade = sapply(nodes, function(nd) desc_tips(ref, nd)), count = prop.clades(ref, trees, rooted = TRUE)),
            file.path(d, "R_support_palette.csv"), row.names = FALSE)
}
# 4./5. PCA distance and its bootstrap
if (file.exists(file.path(d, "features.csv"))) {
  X <- as.matrix(read.csv(file.path(d, "features.csv"), header = FALSE)); S <- readLines(file.path(d, "species.txt"))
  k <- as.integer(readLines(file.path(d, "pca_k.txt")))
  pcd <- function(Z) { p <- prcomp(Z, center = TRUE, scale. = FALSE); D <- as.matrix(dist(p$x[, 1:k, drop = FALSE])); dimnames(D) <- list(S, S); D }
  D0 <- pcd(X); write.csv(D0, file.path(d, "R_pca_distance.csv"))
  idx <- as.matrix(read.csv(file.path(d, "idx_pca.csv"), header = FALSE))
  ref <- phangorn::upgma(as.dist(D0))
  trees <- mclapply(seq_len(nrow(idx)), function(b) phangorn::upgma(as.dist(pcd(X[, idx[b, ]]))), mc.cores = cores)
  class(trees) <- "multiPhylo"
  desc_tips <- function(tr, nd) { g <- getDescendants(tr, nd); paste(sort(tr$tip.label[g[g <= Ntip(tr)]]), collapse = ",") }
  nodes <- Ntip(ref) + seq_len(ref$Nnode)
  write.csv(data.frame(clade = sapply(nodes, function(nd) desc_tips(ref, nd)), count = prop.clades(ref, trees, rooted = TRUE)),
            file.path(d, "R_support_pca.csv"), row.names = FALSE)
}
# 6./7. ouch::hansen and surface::runSurface
tipsof <- function(ot, nd) { lab <- as(ot, "data.frame")$labels; dd <- ouchDescendants(nd, ot)
  if (all(is.na(dd))) dd <- integer(0); sort(lab[intersect(c(nd, dd), ot@term)]) }
pack <- function(ot, sh) lapply(names(sh), function(n) list(tips = I(tipsof(ot, as.numeric(n))), lab = unname(sh[n])))
cases <- list()
tsim <- pbtree(n = 36, scale = 20); tsim$tip.label <- paste0("s", 1:36); tsim <- read.tree(text = write.tree(tsim, digits = 17))
write.tree(tsim, file.path(d, "sim_tree.nwk"), digits = 17)
ot <- convertTreeData(nameNodes(tsim), data.frame(z = rnorm(36), row.names = tsim$tip.label))[[1]]
# two non-nested clades of 3-8 tips for a convergent regime, one more for a third regime
ints <- (ot@nterm + 1):ot@nnodes; ints <- seq(2, ot@nnodes)[!seq(2, ot@nnodes) %in% ot@term]
sz <- sapply(ints, function(nd) length(tipsof(ot, nd)))
cand <- ints[sz >= 3 & sz <= 8]
pick <- c(); for (nd in sample(cand)) { tt <- tipsof(ot, nd)
  if (all(sapply(pick, function(p) length(intersect(tt, tipsof(ot, p))) == 0))) pick <- c(pick, nd); if (length(pick) == 3) break }
planted <- setNames(c("a", "b", "b", "c"), c("1", pick))
sp <- surfaceSimulate(ot, type = "hansen-paint", n_traits = 2, shifts = planted, sigma_squared = 0.15, alpha = 0.6,
                      optima = matrix(c(0, 4, -3, 0, -3, 4), 3))
Xs <- sp$dat[ot@term, , drop = FALSE]; rownames(Xs) <- as(ot, "data.frame")$labels[ot@term]
write.csv(Xs, file.path(d, "surface_planted.csv"))
Xb <- data.frame(V1 = fastBM(tsim, sig2 = 0.3), V2 = fastBM(tsim, sig2 = 0.3))[tsim$tip.label, ]
write.csv(Xb, file.path(d, "surface_bm.csv"))
sdata <- list(planted = list(tree = tsim, X = Xs), bm = list(tree = tsim, X = Xb))
if (file.exists(file.path(d, "real_pcs.csv"))) {
  tr <- read.tree(file.path(d, "real_tree.nwk")); Xr <- read.csv(file.path(d, "real_pcs.csv"), row.names = 1)
  sdata$real <- list(tree = tr, X = Xr[tr$tip.label, , drop = FALSE])
}
out$planted <- pack(ot, planted)
# hansen: several paintings on the simulated tree and on the real tree
hcases <- list()
for (nm in names(sdata)) {
  tr <- nameNodes(sdata[[nm]]$tree); ol <- convertTreeData(tr, sdata[[nm]]$X); o <- ol[[1]]; od <- ol[[2]]
  nn <- seq(2, o@nnodes); big <- nn[sapply(nn, function(x) length(tipsof(o, x))) >= 2]
  for (h in 1:4) {
    sh <- c("1" = "a")
    if (h >= 2) { s <- sample(big, h); sh <- c(sh, setNames(c("b", "c", "b", "d")[seq_len(h)], s)) }
    regs <- repaint(o, sh); rootage <- max(o@times)
    for (j in seq_len(ncol(od))) {
      sa <- sqrt(c(2, 0.5, 5, 1)[h] / rootage); sg <- sqrt(var(od[, j], na.rm = TRUE) / rootage) * c(1, 2, 0.5, 1)[h]
      f <- hansen(od[, j, drop = FALSE], o, regimes = regs, sqrt.alpha = sa, sigma = sg); s_ <- summary(f)
      hcases[[length(hcases) + 1]] <- list(data = nm, trait = j, shifts = pack(o, sh), sqrt_alpha0 = sa, sigma0 = sg,
        loglik = s_$loglik, alpha = s_$alpha[1, 1], sigma2 = s_$sigma.squared[1, 1], theta = as.list(s_$optima[[1]]))
    }
  }
}
out$hansen <- hcases
out$surface <- mclapply(names(sdata), function(nm) {
  tr <- nameNodes(sdata[[nm]]$tree); X <- sdata[[nm]]$X
  t0 <- Sys.time(); r <- runSurface(tr, X); el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ol <- convertTreeData(tr, X); o <- ol[[1]]
  sm <- surfaceSummary(r$fwd, r$bwd)
  list(name = nm, seconds = el, fwd_aic = unname(sapply(r$fwd, function(s) s$aic)),
       bwd_aic = unname(sapply(r$bwd, function(s) s$aic)), shifts = pack(o, sm$shifts),
       fwd_shifts = lapply(r$fwd, function(s) pack(o, s$savedshifts)),
       alpha = unname(sm$alpha), sigma2 = unname(sm$sigma_squared), theta = lapply(seq_len(ncol(sm$theta)), function(j) as.list(sm$theta[, j])),
       n_regimes = as.list(sm$n_regimes)) }, mc.cores = cores)
# 8. geiger on PC1 of the real data
if (file.exists(file.path(d, "real_pcs.csv"))) {
  tr <- read.tree(file.path(d, "real_tree.nwk")); tr <- multi2di(tr, random = FALSE); tr$node.label <- NULL
  Xr <- read.csv(file.path(d, "real_pcs.csv"), row.names = 1); x <- setNames(Xr[tr$tip.label, 1], tr$tip.label)
  out$geiger_pc1 <- lapply(c("BM", "OU", "EB", "lambda", "white"), function(m) {
    f <- suppressWarnings(fitContinuous(tr, x, model = m, control = list(niter = 50)))$opt
    list(model = m, lnL = f$lnL, aicc = f$aicc, param = switch(m, OU = f$alpha, EB = f$a, lambda = f$lambda, NA)) })
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), file.path(d, "R_results.json"))
'''


def _f32_ulps(ours, theirs):
    """|ours - theirs| in units of the float32 spacing at theirs."""
    sp = np.spacing(np.abs(np.asarray(theirs, np.float32))).astype(float)
    return np.abs(np.asarray(ours) - np.asarray(theirs)) / np.maximum(sp, 1e-30)


def _read_mat(path, header=True):
    rows = list(csv.reader(open(path)))
    if header:
        rows = rows[1:]
        return np.array([[float(v) for v in r[1:]] for r in rows])
    return np.array([[float(v) for v in r] for r in rows])


def _tipset_node(T, tips):
    tips = [tips] if isinstance(tips, str) else list(tips)
    s = frozenset(tips)
    if len(s) == len(T.names):
        return 0
    return next(i for i in range(T.N) if T.tipset[i] == s)


def _counts_vs_r(dp, D0, names, dist_fn, idx, rfile):
    ref = dp.build_colour_tree(D0, names, "upgma")
    cnt = dp.support_counts_fn(dist_fn, D0, names, "upgma", idx, ref); cnt.pop("__filled__")
    ours = {",".join(sorted(s)): cnt[id(n)] for n, s in dp.clade_sets(ref)}
    rc = {r["clade"]: int(r["count"]) for r in csv.DictReader(open(rfile)) if r["count"] not in ("NA", "")}
    shared = [k for k in ours if k in rc]
    diff = [abs(ours[k] - rc[k]) for k in shared]
    return {"replicates": len(idx), "nodes": len(ours), "nodes_compared": len(shared),
            "max_abs_count_diff": int(max(diff)) if diff else None,
            "support_range": [min(ours.values()) / len(idx), max(ours.values()) / len(idx)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", default="Rscript"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tree", default=None); ap.add_argument("--colour_table", default=None)
    ap.add_argument("--groups", default=None); ap.add_argument("--palettes", default=None)
    ap.add_argument("--surface_pcs", type=int, default=3, help="real-data PCs for SURFACE (the run's broken-stick choice)")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--reuse_r", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dp = load_dp()
    rng = np.random.default_rng(a.seed)
    summary = {}

    # ---------------------------------------------------------------- inputs
    rand_pairs = []
    with open(out / "emd_random.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["pair", "side", "w", "L", "a", "b"])
        for k in range(300):
            pr = []
            for side in (1, 2):
                m = int(rng.integers(1, 16))
                wt = rng.gamma(0.7, 1.0, m) + 1e-3
                lab = np.column_stack([rng.uniform(0, 100, m), rng.normal(0, 25, m), rng.normal(0, 25, m)])
                if k < 30 and side == 2:                         # near-identical palettes
                    lab = pr[0][1][:m] if len(pr[0][1]) >= m else lab
                    lab = lab + rng.normal(0, 0.5, lab.shape)
                pr.append((wt, lab))
                for wi, li in zip(wt, lab):
                    w.writerow([k + 1, side, repr(float(wi))] + [repr(float(v)) for v in li])
            rand_pairs.append(pr)
    ours_rand = np.array([dp.emd(p[0][0], p[0][1], p[1][0], p[1][1]) for p in rand_pairs])
    # exactness: one-dimensional palettes, where EMD = the 1-D Wasserstein distance
    from scipy.stats import wasserstein_distance
    d1 = []
    for k in range(300):
        m1, m2 = int(rng.integers(1, 16)), int(rng.integers(1, 16))
        x1, x2 = rng.normal(0, 30, m1), rng.normal(5, 30, m2)
        w1, w2 = rng.gamma(0.7, 1, m1) + 1e-3, rng.gamma(0.7, 1, m2) + 1e-3
        d1.append(abs(dp.emd(w1, x1[:, None], w2, x2[:, None]) - wasserstein_distance(x1, x2, w1, w2)))
    summary["emd_exact_1d_vs_scipy_wasserstein"] = {"pairs": 300, "max_abs_diff": float(max(d1))}

    real = None
    if a.tree and a.colour_table and a.groups:
        root, _ = dp.load_tree(a.tree)
        gmap = dp.load_group_map(a.groups)
        sp, cols, M, counts = dp.load_traits(a.colour_table, None, None, None, gmap, keep_missing=True)
        sp = [dp._norm_name(s) for s in sp]
        complete = ~np.isnan(M).any(axis=0)
        tipn = {t.name for t in dp.tips_of(root)}
        common = [s for s in sp if s in tipn]
        tree = dp.prune(dp._copy_tree(root), set(common)); tips = dp.tips_of(tree); names = [t.name for t in tips]
        (out / "real_tree.nwk").write_text(dp.to_newick(tree, fmt="{!r}") + ";\n")
        (out / "species.txt").write_text("\n".join(names) + "\n")
        Xs = dp.scale(M[[sp.index(s) for s in names]][:, complete], "standardize")
        np.savetxt(out / "features.csv", Xs, delimiter=",", fmt="%.17g")
        sc, ev = dp.pattern_pca(Xs)
        k = dp.n_axes(ev, "auto", 10, len(names))
        (out / "pca_k.txt").write_text(f"{k}\n")
        Dpca = dp._euclid(sc[:, :k])
        idx_pca = rng.integers(0, Xs.shape[1], size=(a.nboot, Xs.shape[1]))
        np.savetxt(out / "idx_pca.csv", idx_pca + 1, delimiter=",", fmt="%d")
        T = dp.OUTree(tree)
        kk = a.surface_pcs
        with open(out / "real_pcs.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["species"] + [f"PC{i + 1}" for i in range(kk)])
            for i, s in enumerate(names):
                w.writerow([s] + [repr(float(v)) for v in sc[i, :kk]])
        real = {"tree": tree, "names": names, "Xs": Xs, "k": k, "Dpca": Dpca, "idx_pca": idx_pca, "pcs": sc[:, :kk],
                "T": T}
        if a.palettes:
            pals = dp.load_palettes(a.palettes, gmap, names)
            with open(out / "palettes_real.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["image", "frac", "L", "a", "b"])
                for p in pals:
                    for wi, li in zip(p["w"], p["lab"]):
                        w.writerow([p["photo"], repr(float(wi))] + [repr(float(v)) for v in li])
            with open(out / "slots.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["photo", "species"])
                for p in pals:
                    w.writerow([p["photo"], p["species"]])
            Msp = dp.specimen_emd_matrix(pals)
            slot = np.array([names.index(p["species"]) for p in pals])
            Dsp = dp.species_mean_distance(Msp, slot, len(names))
            np.savetxt(out / "py_specimen_emd.csv", Msp, delimiter=",", fmt="%.17g")
            np.savetxt(out / "py_species_emd.csv", Dsp, delimiter=",", fmt="%.17g")
            idx_pal = dp.specimen_resamples(slot, a.nboot, rng)
            np.savetxt(out / "idx_palette.csv", idx_pal + 1, delimiter=",", fmt="%d")
            real.update(pals=pals, Msp=Msp, Dsp=Dsp, slot=slot, idx_pal=idx_pal)

    # ---------------------------------------------------------------- R
    (out / "r_side.R").write_text(R_CODE)
    if not (a.reuse_r and (out / "R_results.json").exists()):
        subprocess.run([a.rscript, str(out / "r_side.R"), str(out)], check=True)
    R = json.loads((out / "R_results.json").read_text())
    summary["R_versions"] = R["versions"]

    # 1. EMD
    rr = np.array(R["emd_random"], float)
    ul = _f32_ulps(ours_rand, rr)
    summary["emd_vs_emdist_random"] = {
        "pairs": len(rr), "max_abs_diff": float(np.abs(ours_rand - rr).max()),
        "max_rel_diff": float((np.abs(ours_rand - rr) / np.maximum(rr, 1e-300)).max()),
        "max_diff_in_float32_spacings": float(ul.max()),
        "emdist_values_are_float32": bool(np.all(rr.astype(np.float32).astype(float) == rr))}
    if real is not None and "pals" in real:
        Rsp = _read_mat(out / "R_specimen_emd.csv")
        iu = np.triu_indices(len(Rsp), 1)
        ulr = _f32_ulps(real["Msp"][iu], Rsp[iu])
        summary["emd_vs_emdist_real_palettes"] = {
            "specimens": len(Rsp), "pairs": len(iu[0]), "max_abs_diff": float(np.abs(real["Msp"] - Rsp).max()),
            "max_rel_diff": float((np.abs(real["Msp"][iu] - Rsp[iu]) / Rsp[iu]).max()),
            "max_diff_in_float32_spacings": float(ulr.max()),
            "emdist_values_are_float32": bool(np.all(Rsp[iu].astype(np.float32).astype(float) == Rsp[iu]))}
        R1 = _read_mat(out / "R_species_emd_R_specimens.csv"); R2 = _read_mat(out / "R_species_emd_py_specimens.csv")
        summary["species_emd_vs_trees_R"] = {
            "species": len(R1), "max_abs_diff_vs_R_with_emdist_specimens": float(np.abs(real["Dsp"] - R1).max()),
            "max_abs_diff_vs_R_species_mat_on_descriptron_specimens": float(np.abs(real["Dsp"] - R2).max())}
        # 3. PCoA
        X, ev = dp.pcoa(real["Dsp"])
        ecm = np.array([float(v) for v in (out / "R_cmdscale_eig.txt").read_text().split()])
        epc = np.array([float(v) for v in (out / "R_pcoa_eig.txt").read_text().split()])
        Pcm = np.array([[float(v) for v in r] for r in list(csv.reader(open(out / "R_cmdscale_points.csv")))[1:]])
        Ppc = np.array([[float(v) for v in r] for r in list(csv.reader(open(out / "R_pcoa_vectors.csv")))[1:]])
        kk = min(X.shape[1], Pcm.shape[1], Ppc.shape[1], 10)
        def sign_diff(A, B):
            return float(max(min(np.abs(A[:, j] - B[:, j]).max(), np.abs(A[:, j] + B[:, j]).max()) for j in range(kk)))
        npos = int((epc > 1e-10 * epc[0]).sum())
        summary["pcoa_vs_cmdscale_ape"] = {
            "axes_compared": kk, "max_abs_diff_eigenvalues_cmdscale": float(np.abs(ev[:len(ecm)] - ecm[:len(ev)]).max()),
            "max_abs_diff_eigenvalues_ape_pcoa": float(np.abs(ev[:len(epc)] - epc[:len(ev)]).max()),
            "max_abs_diff_coordinates_cmdscale_up_to_sign": sign_diff(X, Pcm),
            "max_abs_diff_coordinates_ape_pcoa_up_to_sign": sign_diff(X, Ppc),
            "positive_eigenvalues": npos, "negative_eigenvalue_share": float(-ev[ev < 0].sum() / np.abs(ev).sum())}
        # 5. palette bootstrap
        n = len(real["names"])
        summary["bootstrap_palette_emd_vs_ape_prop_clades"] = _counts_vs_r(
            dp, real["Dsp"], real["names"], lambda row: dp.species_mean_distance(real["Msp"], real["slot"], n, row),
            real["idx_pal"], out / "R_support_palette.csv")
        summary["bootstrap_palette_emd_vs_ape_prop_clades"]["species_with_one_specimen"] = int(
            sum(1 for s in range(n) if (real["slot"] == s).sum() == 1))
    if real is not None:
        Rp = _read_mat(out / "R_pca_distance.csv")
        summary["pca_distance_vs_prcomp_dist"] = {"k": real["k"], "max_abs_diff": float(np.abs(Rp - real["Dpca"]).max())}
        Xs, k = real["Xs"], real["k"]
        summary["bootstrap_pca_vs_ape_prop_clades"] = _counts_vs_r(
            dp, real["Dpca"], real["names"], lambda row: dp._euclid(dp._pca_scores(Xs[:, row], k)), real["idx_pca"],
            out / "R_support_pca.csv")

    # 6. hansen
    trees = {"planted": dp.parse_newick((out / "sim_tree.nwk").read_text()),
             "bm": dp.parse_newick((out / "sim_tree.nwk").read_text())}
    data = {}
    for nm in ("planted", "bm"):
        rows = list(csv.reader(open(out / f"surface_{nm}.csv")))
        data[nm] = {r[0]: [float(v) for v in r[1:]] for r in rows[1:]}
    if real is not None:
        trees["real"] = real["tree"]
        data["real"] = {s: list(real["pcs"][i]) for i, s in enumerate(real["names"])}
    OT = {nm: dp.OUTree(t) for nm, t in trees.items()}
    hrows = []
    for h in R["hansen"]:
        T = OT[h["data"]]
        y = np.array([data[h["data"]][s][h["trait"] - 1] for s in T.names])
        sh = [(_tipset_node(T, x["tips"]), x["lab"]) for x in h["shifts"]]
        f = dp.hansen_fit(T, y, T.paint(sh), h["sqrt_alpha0"], h["sigma0"])
        th = {k_: float(v) for k_, v in h["theta"].items()}
        hrows.append({"data": h["data"], "trait": h["trait"], "n_shifts": len(sh),
                      "abs_diff_loglik": abs(f["loglik"] - h["loglik"]),
                      "rel_diff_alpha": abs(f["alpha"] - h["alpha"]) / abs(h["alpha"]),
                      "rel_diff_sigma2": abs(f["sigma2"] - h["sigma2"]) / abs(h["sigma2"]),
                      "max_abs_diff_theta": max(abs(f["theta"][k_] - v) for k_, v in th.items()),
                      "loglik_ouch": h["loglik"], "alpha_ouch": h["alpha"]})
    summary["hansen_vs_ouch"] = {"fits": len(hrows),
                                 "max_abs_diff_loglik": max(r["abs_diff_loglik"] for r in hrows),
                                 "max_rel_diff_alpha": max(r["rel_diff_alpha"] for r in hrows),
                                 "max_rel_diff_sigma2": max(r["rel_diff_sigma2"] for r in hrows),
                                 "max_abs_diff_theta": max(r["max_abs_diff_theta"] for r in hrows), "per_fit": hrows}

    # 7. SURFACE
    srows = []
    planted = [(frozenset([x["tips"]] if isinstance(x["tips"], str) else x["tips"]), x["lab"]) for x in R["planted"]]
    for rs in R["surface"]:
        nm = rs["name"]; T = OT[nm]
        Y = np.array([data[nm][s] for s in T.names])
        import time
        t0 = time.time(); res = dp.run_surface(T, Y); el = time.time() - t0
        fa = [s["aic"] for s in res["fwd"]]; ba = [s["aic"] for s in res["bwd"]]
        rfa = [float(v) for v in np.atleast_1d(rs["fwd_aic"])]; rba = [float(v) for v in np.atleast_1d(rs["bwd_aic"])]

        def part(shifts):
            return sorted(sorted(sorted(ts) for ts, l in shifts if l == lab) for lab in {l for _, l in shifts})
        ours_sh = [(T.tipset[v], l) for v, l in res["final"]["shifts"]]
        r_sh = [(frozenset([x["tips"]] if isinstance(x["tips"], str) else x["tips"]), x["lab"]) for x in rs["shifts"]]
        same_steps = len(fa) == len(rfa) and len(ba) == len(rba)
        row = {"case": nm, "species": len(T.names), "traits": Y.shape[1], "forward_steps_descriptron": len(fa),
               "forward_steps_surface": len(rfa), "backward_steps_descriptron": len(ba), "backward_steps_surface": len(rba),
               "max_abs_diff_AICc_path": float(max(abs(x - y) for x, y in zip(fa + ba, rfa + rba))) if same_steps else None,
               "same_final_shift_set": {ts for ts, _ in ours_sh} == {ts for ts, _ in r_sh},
               "same_final_regime_partition": part(ours_sh) == part(r_sh),
               "final_AICc_descriptron": res["final"]["aic"], "final_AICc_surface": rba[-1],
               "n_regimes_descriptron": res["final"]["n_regimes"], "n_regimes_surface": rs["n_regimes"],
               "max_rel_diff_alpha": float(max(abs(f["alpha"] - ra) / ra for f, ra in zip(res["final"]["fit"], np.atleast_1d(rs["alpha"])))),
               "max_rel_diff_sigma2": float(max(abs(f["sigma2"] - ra) / ra for f, ra in zip(res["final"]["fit"], np.atleast_1d(rs["sigma2"])))),
               "seconds_descriptron": el, "seconds_surface": rs["seconds"]}
        th = []
        lab_map = {}
        for (ts, l), (ts2, l2) in zip(sorted(ours_sh, key=lambda x: sorted(x[0])), sorted(r_sh, key=lambda x: sorted(x[0]))):
            lab_map[l] = l2
        for j, f in enumerate(res["final"]["fit"]):
            rt = rs["theta"][j]
            th += [abs(v - float(rt[lab_map.get(k_, k_)])) for k_, v in f["theta"].items() if lab_map.get(k_, k_) in rt]
        row["max_abs_diff_theta"] = float(max(th)) if th else None
        if nm == "planted":
            row["planted_regime_partition"] = part(planted)
            row["recovered_planted_partition_exactly"] = part(ours_sh) == part(planted)
            row["planted_regimes_all_recovered"] = all(g in part(ours_sh) for g in part(planted))
        row["final_regimes_descriptron"] = part(ours_sh)
        srows.append(row)
    summary["surface_vs_runSurface"] = {
        "cases": len(srows),
        "all_same_final_shift_sets": all(r["same_final_shift_set"] for r in srows),
        "all_same_regime_partitions": all(r["same_final_regime_partition"] for r in srows),
        "max_abs_diff_AICc_path": max((r["max_abs_diff_AICc_path"] for r in srows if r["max_abs_diff_AICc_path"] is not None), default=None),
        "max_rel_diff_alpha": max(r["max_rel_diff_alpha"] for r in srows),
        "max_rel_diff_sigma2": max(r["max_rel_diff_sigma2"] for r in srows),
        "max_abs_diff_theta": max((r["max_abs_diff_theta"] for r in srows if r["max_abs_diff_theta"] is not None), default=None),
        "per_case": srows}

    # 8. geiger on PC1
    if real is not None and "geiger_pc1" in R:
        rtips = dp.tips_of(real["tree"])
        fits = {f["model"]: f for f in dp.fit_models(real["pcs"][:, 0], dp.vcv(rtips, rtips))}
        gr = []
        for g in R["geiger_pc1"]:
            f = fits[g["model"]]
            gr.append({"model": g["model"], "abs_diff_lnL": abs(f["logLik"] - g["lnL"]), "abs_diff_AICc": abs(f["AICc"] - g["aicc"]),
                       "param_descriptron": f["value"], "param_geiger": g["param"]})
        summary["models_on_PC1_vs_geiger"] = {"max_abs_diff_lnL": max(r["abs_diff_lnL"] for r in gr),
                                              "max_abs_diff_AICc": max(r["abs_diff_AICc"] for r in gr), "per_model": gr}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    brief = {k: ({kk: vv for kk, vv in v.items() if kk not in ("per_fit", "per_case", "per_model")} if isinstance(v, dict) else v)
             for k, v in summary.items()}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
