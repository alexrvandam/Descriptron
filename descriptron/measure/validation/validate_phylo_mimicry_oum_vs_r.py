#!/usr/bin/env python3
"""descriptron_phylo's reference-group scores, Mk models, ancestral states, stochastic maps, OUM and BMS against R.

Python writes the inputs (distance matrices and labels, trees, discrete states, colour-pattern PCs, regime
paintings); R (stats, cluster, ape, phytools, ouch) computes; Descriptron computes the same from the same inputs:

  1. reference scores, on the real colour distances of all species with data (soma: de2000, pca) and on random
     data: nearest-neighbour agreement (an R transcription: which(d == min(d)), ties sharing the vote), the UPGMA cut
     against cutree(hclust(as.dist(D), "average"), k), k-medoids against cluster::pam(diss = TRUE) (partition and
     objective), and the adjusted Rand index against a direct transcription of Hubert & Arabie's formula in R
     (mclust and aricode are not installed: R's value is computed on R's own partitions and on Descriptron's)
  2. Mk models (ER, SYM, ARD) of a discrete trait: the log-likelihood at the same Q against phytools::fitMk(fixedQ =
     Q) (the likelihood function itself), the maximised log-likelihood and rates against fitMk (pi = "equal") and
     ape::ace(type = "discrete") (whose log-likelihood is fitMk's + log(k): it sums the root's conditional
     likelihoods without the 1/k prior); on the real tree with the mimicry types and on a simulated tree
  3. marginal ancestral states at fitMk's Q against phytools::ancr(fitMk) and ape::ace(marginal = FALSE). (ape 5.8.1's
     lik.anc agree for ER and SYM, but under ARD they differ from ancr at the same Q, by up to 0.09 here; Descriptron
     agrees with ancr and with brute-force enumeration of all internal states, tests/test_descriptron_phylo.py.)
     fitMk and ace can stop at a local optimum for SYM / ARD on few species: the maximised log-likelihoods are
     compared as Descriptron minus R (>= 0 means Descriptron found at least as high a maximum)
  4. stochastic maps: mean number of transitions of every kind and mean time in each state over --nsim maps, against
     phytools::make.simmap(Q = the same matrix, pi = "equal") + describe.simmap, as z = difference / its Monte Carlo
     standard error
  5. OUM on a regime painting (the most probable marginal Mk states and several stochastic-map paintings) against
     ouch::hansen per colour-pattern PC (log-likelihood, alpha, sigma^2, every optimum) from the same start; BMS
     against phytools::brownie.lite on the same painting

  python validate_phylo_mimicry_oum_vs_r.py --rscript Rscript --out-dir val_mimicry_oum/ --tree tree.nwk \\
      --colour_table color_homology_features_soma_combined_hw1.csv --groups group_labels.csv \\
      --reference alex_lookalike_groups.csv --discrete_col mimicry_type
"""
import argparse, csv, importlib.util, json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def _sibling(name):
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)


def load_dp():
    spec = importlib.util.spec_from_file_location("descriptron_phylo", _sibling("descriptron_phylo.py"))
    dp = importlib.util.module_from_spec(spec); sys.modules["descriptron_phylo"] = dp; spec.loader.exec_module(dp)
    return dp


R_CODE = r'''
args <- commandArgs(TRUE); d <- args[1]; nsim <- as.integer(args[2])
suppressMessages({library(ape); library(phytools); library(cluster); library(ouch); library(surface); library(jsonlite)})
set.seed(20261107)
out <- list(versions = sapply(c("ape", "phytools", "cluster", "ouch", "stats"), function(p) as.character(packageVersion(p))))
ari <- function(a, b) {                          # Hubert & Arabie (1985), transcribed
  tab <- table(a, b); n <- length(a)
  sij <- sum(choose(tab, 2)); sa <- sum(choose(rowSums(tab), 2)); sb <- sum(choose(colSums(tab), 2))
  e <- sa * sb / choose(n, 2); (sij - e) / (0.5 * (sa + sb) - e) }
nn <- function(D, lab) {
  diag(D) <- Inf; cnt <- table(lab); ok <- cnt[lab] >= 2
  sh <- sapply(seq_len(nrow(D)), function(i) { j <- which(D[i, ] <= min(D[i, ]) * (1 + 1e-12)); mean(lab[j] == lab[i]) })
  c(score = mean(sh[ok]), counted = sum(ok)) }
# 1. scoring
sc <- list()
for (f in list.files(d, pattern = "^score_D_.*\\.csv$")) {
  nm <- sub("^score_D_(.*)\\.csv$", "\\1", f)
  D <- as.matrix(read.csv(file.path(d, f), header = FALSE)); lab <- readLines(file.path(d, paste0("score_lab_", nm, ".txt")))
  pyc <- read.csv(file.path(d, paste0("score_py_", nm, ".csv")))
  k <- length(unique(lab))
  cu <- cutree(hclust(as.dist(D), "average"), k)
  pm <- pam(as.dist(D), k, diss = TRUE)
  r <- nn(D, lab)
  sc[[nm]] <- list(n = nrow(D), k = k, nn = unname(r["score"]), nn_counted = unname(r["counted"]),
                   cut = cu, pam = unname(pm$clustering), pam_objective = unname(pm$objective["swap"]),
                   ari_cut_R = ari(cu, lab), ari_pam_R = ari(pm$clustering, lab),
                   ari_R_on_py_cut = ari(pyc$cut, lab), ari_R_on_py_pam = ari(pyc$pam, lab))
}
out$scoring <- sc
# 2.-4. Mk, ancestral states, stochastic maps
desc_tips <- function(tr, nd) { if (nd <= Ntip(tr)) return(tr$tip.label[nd])
  g <- getDescendants(tr, nd); paste(sort(tr$tip.label[g[g <= Ntip(tr)]]), collapse = ",") }
mk <- list()
cases <- list()
tr <- read.tree(file.path(d, "mk_real_tree.nwk")); st <- read.csv(file.path(d, "mk_real_states.csv"))
cases$real <- list(tree = tr, x = setNames(st$state, st$species)[tr$tip.label])
ts <- pbtree(n = 60, scale = 10); ts$tip.label <- paste0("t", 1:60); ts <- read.tree(text = write.tree(ts, digits = 17))
Qs <- matrix(c(-0.25, 0.2, 0.05, 0.1, -0.15, 0.05, 0.02, 0.1, -0.12), 3, 3, byrow = TRUE,
             dimnames = list(c("s1", "s2", "s3"), c("s1", "s2", "s3")))
repeat { xs <- sim.Mk(ts, Qs); if (length(unique(xs)) == 3) break }
write.tree(ts, file.path(d, "mk_sim_tree.nwk"), digits = 17)
write.csv(data.frame(species = names(xs), state = as.character(xs)), file.path(d, "mk_sim_states.csv"), row.names = FALSE)
cases$sim <- list(tree = ts, x = setNames(as.character(xs), names(xs))[ts$tip.label])
for (cn in names(cases)) {
  tr <- cases[[cn]]$tree; x <- factor(cases[[cn]]$x); nodes <- Ntip(tr) + seq_len(tr$Nnode)
  res <- list(states = levels(x), node_tips = sapply(nodes, function(nd) desc_tips(tr, nd)))
  for (m in c("ER", "SYM", "ARD")) {
    f <- fitMk(tr, x, model = m, pi = "equal")
    Qf <- unclass(as.Qmatrix(f))
    an <- ancr(f)
    ac <- tryCatch(ace(x, tr, type = "discrete", model = m), error = function(e) NULL)
    pyQ <- file.path(d, sprintf("py_Q_%s_%s.csv", cn, m))
    fix <- if (file.exists(pyQ)) { Qp <- as.matrix(read.csv(pyQ, header = FALSE)); dimnames(Qp) <- list(levels(x), levels(x))
      as.numeric(logLik(fitMk(tr, x, fixedQ = Qp, pi = "equal"))) } else NA
    acQ <- NULL; acanc <- NULL
    if (!is.null(ac)) { acQ <- matrix(0, length(levels(x)), length(levels(x)))
      acQ[] <- c(NA, ac$rates)[ac$index.matrix + 1]; diag(acQ) <- 0; acanc <- unname(ac$lik.anc) }
    res[[m]] <- list(logLik = as.numeric(logLik(f)), Q = unname(Qf), anc = unname(an$ace[as.character(nodes), , drop = FALSE]),
                     ace_loglik = if (is.null(ac)) NA else ac$loglik, ace_Q = acQ, ace_anc = acanc,
                     logLik_at_descriptron_Q = fix)
  }
  # stochastic maps with the ER (real) / ARD (simulated) fitMk Q
  mq <- if (cn == "real") "ER" else "ARD"
  Qm <- res[[mq]]$Q; dimnames(Qm) <- list(levels(x), levels(x))
  t0 <- Sys.time()
  sm <- make.simmap(tr, setNames(as.character(x), tr$tip.label), Q = Qm, nsim = nsim, pi = "equal", message = FALSE)
  ds <- describe.simmap(sm, message = FALSE)
  cnt <- ds$count; tms <- ds$times
  res$simmap <- list(model = mq, Q = unname(Qm), nsim = nsim, seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
                     count_cols = colnames(cnt), count = unname(as.matrix(cnt)), times_cols = colnames(tms),
                     times = unname(as.matrix(tms[seq_len(nsim), , drop = FALSE])))
  mk[[cn]] <- res
}
out$mk <- mk
# 5. OUM (ouch::hansen) and BMS (brownie.lite) on given paintings
if (file.exists(file.path(d, "oum_pcs.csv"))) {
  tr <- read.tree(file.path(d, "oum_tree.nwk")); X <- read.csv(file.path(d, "oum_pcs.csv"), row.names = 1)
  pt <- read.csv(file.path(d, "oum_paintings.csv"), stringsAsFactors = FALSE)       # painting, tips (sorted, comma), regime
  starts <- read.csv(file.path(d, "oum_starts.csv"))
  tn <- nameNodes(tr); ol <- convertTreeData(tn, X); o <- ol[[1]]; od <- ol[[2]]
  lab <- as(o, "data.frame")$labels
  tipsof <- function(nd) { dd <- ouchDescendants(nd, o); if (all(is.na(dd))) dd <- integer(0)
    paste(sort(lab[intersect(c(nd, dd), o@term)]), collapse = ",") }
  okeys <- sapply(seq_len(o@nnodes), tipsof)
  akeys <- sapply(seq_len(Ntip(tr) + tr$Nnode), function(nd) desc_tips(tr, nd))
  oum <- list()
  for (pn in unique(pt$painting)) {
    p <- pt[pt$painting == pn, ]; reg <- setNames(p$regime, p$tips)
    regs <- factor(unname(reg[okeys])); names(regs) <- o@nodes
    fits <- list()
    for (j in seq_len(ncol(od))) {
      h <- hansen(od[, j, drop = FALSE], o, regimes = regs, sqrt.alpha = starts$sqrt_alpha[j], sigma = starts$sigma[j])
      s <- summary(h)
      # BMS: brownie.lite on the same painting (each edge in its child's regime)
      sm <- tr; ch <- tr$edge[, 2]
      sm$maps <- lapply(seq_len(nrow(tr$edge)), function(e) setNames(tr$edge.length[e], reg[[akeys[ch[e]]]]))
      lv <- sort(unique(p$regime))
      sm$mapped.edge <- t(sapply(seq_len(nrow(tr$edge)), function(e) { v <- setNames(rep(0, length(lv)), lv)
        v[reg[[akeys[ch[e]]]]] <- tr$edge.length[e]; v }))
      rownames(sm$mapped.edge) <- apply(tr$edge, 1, paste, collapse = ",")
      class(sm) <- c("simmap", "phylo")
      xv <- setNames(X[tr$tip.label, j], tr$tip.label)
      b <- brownie.lite(sm, xv)
      fits[[j]] <- list(loglik = s$loglik, alpha = s$alpha[1, 1], sigma2 = s$sigma.squared[1, 1], theta = as.list(s$optima[[1]]),
                        bms_loglik = b$logL.multiple, bms_sig2 = as.list(b$sig2.multiple), bms_converged = b$convergence)
    }
    oum[[pn]] <- fits
  }
  out$oum <- oum
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), file.path(d, "R_results.json"))
'''


def _mat(x):
    return np.array(x, float)


def _rateQ(x):
    """an R rate matrix whose diagonal may be 0 (ape::ace's rates filled in): the diagonal set to -row sums."""
    Q = np.array(x, float)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(1))
    return Q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", default="Rscript"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tree", required=True); ap.add_argument("--colour_table", required=True)
    ap.add_argument("--groups", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--discrete_col", default="mimicry_type")
    ap.add_argument("--nsim", type=int, default=2000, help="stochastic maps on each side")
    ap.add_argument("--n_paintings", type=int, default=4, help="stochastic-map paintings checked against ouch")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--reuse_r", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dp = load_dp()
    rng = np.random.default_rng(a.seed)
    summary = {}

    # ------------------------------------------------------------ inputs: scoring
    gmap = dp.load_group_map(a.groups)
    sp, full_cols, full_M, counts = dp.load_traits(a.colour_table, None, None, None, gmap, keep_missing=True)
    sp = [dp._norm_name(s) for s in sp]
    complete = ~np.isnan(full_M).any(axis=0)
    S = {"species": sp, "cols": [c for c, k in zip(full_cols, complete) if k], "raw": full_M[:, complete],
         "counts": {dp._norm_name(k): v for k, v in counts.items()}, "full_cols": full_cols, "full_raw": full_M}
    ref = dp.load_reference_groups(a.reference, ["group", a.discrete_col])
    pool = [s for s in sp if s in ref["group"]]
    opts = SimpleNamespace(colour_cells=dp.COLOUR_CELLS_DEFAULT, conv_pcs="auto", conv_pcs_max=10, pca_pcs="auto",
                           palettes=[])
    score_cases = {}
    for dist in ("de2000", "pca"):
        eng, why = dp.colour_engine(dist, "soma", S, pool, opts, gmap)
        for col in ("group", a.discrete_col):
            score_cases[f"real_{dist}_{col}"] = (eng["D"], [ref[col][s] for s in pool])
    for r in range(6):
        n = int(rng.integers(15, 60)); X = rng.normal(size=(n, 3)) + np.repeat(rng.normal(0, 2, (4, 3)), -(-n // 4), 0)[:n]
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        if r == 5:
            D = np.round(D, 0)                                   # many ties
        score_cases[f"random{r}"] = (D, [f"L{int(v)}" for v in rng.integers(0, int(rng.integers(2, 8)), n)])
    py_scores = {}
    for nm, (D, lab) in score_cases.items():
        np.savetxt(out / f"score_D_{nm}.csv", D, delimiter=",", fmt="%.17g")
        (out / f"score_lab_{nm}.txt").write_text("\n".join(lab) + "\n")
        k = len(set(lab))
        cu = dp.cut_upgma(D, k); _, cp, obj = dp.pam(D, k)
        nnv, ncount, _ = dp.nn_agreement(D, lab)
        py_scores[nm] = {"cut": cu, "pam": cp, "obj": obj, "nn": nnv, "nn_counted": ncount,
                         "ari_cut": dp.adjusted_rand(cu, lab), "ari_pam": dp.adjusted_rand(cp, lab), "lab": lab}
        with open(out / f"score_py_{nm}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["cut", "pam"]); w.writerows(zip(cu, cp))

    # ------------------------------------------------------------ inputs: Mk (real tree, mimicry types)
    root, _ = dp.load_tree(a.tree)
    disc = dp.load_reference_groups(a.reference, [a.discrete_col])[a.discrete_col]
    tipn = {t.name for t in dp.tips_of(root)}
    common = [s for s in sp if s in tipn and s in disc]
    tree = dp.prune(dp._copy_tree(root), set(common))
    (out / "mk_real_tree.nwk").write_text(dp.to_newick(tree, fmt="{!r}") + ";\n")
    with open(out / "mk_real_states.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["species", "state"])
        for t in dp.tips_of(tree):
            w.writerow([t.name, disc[t.name]])

    def mk_inputs(nwk, states_csv):
        tr = dp.parse_newick(Path(nwk).read_text())
        stt = {r["species"]: r["state"] for r in csv.DictReader(open(states_csv))}
        T = dp.MkTree(tr)
        states = sorted(set(stt.values()))
        X = np.zeros((len(T.names), len(states)))
        for i, s in enumerate(T.names):
            X[i, states.index(stt[s])] = 1
        return T, X, states

    # our fits first, so R can evaluate fitMk at our Q (the R side simulates the second case: run R twice if needed)
    mkpy = {}
    T, X, states = mk_inputs(out / "mk_real_tree.nwk", out / "mk_real_states.csv")
    mkpy["real"] = (T, X, states, {m: dp.fit_mk(T, X, m) for m in ("ER", "SYM", "ARD")})
    for m, f in mkpy["real"][3].items():
        np.savetxt(out / f"py_Q_real_{m}.csv", f["Q"], delimiter=",", fmt="%.17g")

    # ------------------------------------------------------------ inputs: OUM paintings (real tree, soma PCs)
    names = [t.name for t in dp.tips_of(tree)]
    Xs = dp.scale(full_M[[sp.index(s) for s in names]][:, complete], "standardize")
    sc, ev = dp.pattern_pca(Xs)
    kpc = max(2, min(dp.broken_stick_count(ev, len(names)), 4))
    OT = dp.OUTree(tree)
    Yt = sc[:, :kpc][[names.index(s) for s in OT.names]]
    with open(out / "oum_pcs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["species"] + [f"PC{j + 1}" for j in range(kpc)])
        for i, s in enumerate(OT.names):
            w.writerow([s] + [repr(float(v)) for v in Yt[i]])
    (out / "oum_tree.nwk").write_text(dp.to_newick(tree, fmt="{!r}") + ";\n")
    fbest = min(mkpy["real"][3].values(), key=lambda f: f["AICc"])
    MT, MX, mstates = mkpy["real"][0], mkpy["real"][1], mkpy["real"][2]
    post = dp.mk_marginal(MT, MX, fbest["Q"])
    mkd = {"T": MT, "X": MX, "states": mstates}
    ps = post.argmax(1); ps[MT.tip_ix] = MX.argmax(1)
    paintings = {"MAP": dp.node_painting(OT, mkd, ps)}
    maps = dp.simmaps(MT, MX, fbest["Q"], a.n_paintings, rng)
    for i, m in enumerate(maps):
        paintings[f"map{i + 1}"] = dp.node_painting(OT, mkd, m["state"])
    with open(out / "oum_paintings.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["painting", "tips", "regime"])
        for pn, reg in paintings.items():
            for i in range(OT.N):
                w.writerow([pn, ",".join(sorted(OT.tipset[i])), reg[i]])
    starts = [(np.sqrt(2 / OT.depth), np.sqrt(np.var(Yt[:, j], ddof=1) / OT.depth)) for j in range(kpc)]
    with open(out / "oum_starts.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["sqrt_alpha", "sigma"]); w.writerows(starts)

    # ------------------------------------------------------------ R
    (out / "r_side.R").write_text(R_CODE)
    if not (a.reuse_r and (out / "R_results.json").exists()):
        subprocess.run([a.rscript, str(out / "r_side.R"), str(out), str(a.nsim)], check=True)
    R = json.loads((out / "R_results.json").read_text())
    T, X, states = mk_inputs(out / "mk_sim_tree.nwk", out / "mk_sim_states.csv")
    mkpy["sim"] = (T, X, states, {m: dp.fit_mk(T, X, m) for m in ("ER", "SYM", "ARD")})
    need_rerun = False
    for m, f in mkpy["sim"][3].items():
        fq = out / f"py_Q_sim_{m}.csv"
        if not fq.exists():
            need_rerun = True
        np.savetxt(fq, f["Q"], delimiter=",", fmt="%.17g")
    if need_rerun:                                              # the simulated case's fitMk(fixedQ = our Q)
        subprocess.run([a.rscript, str(out / "r_side.R"), str(out), str(a.nsim)], check=True)
        R = json.loads((out / "R_results.json").read_text())
        T, X, states = mk_inputs(out / "mk_sim_tree.nwk", out / "mk_sim_states.csv")
        mkpy["sim"] = (T, X, states, {m: dp.fit_mk(T, X, m) for m in ("ER", "SYM", "ARD")})
        for m, f in mkpy["sim"][3].items():
            np.savetxt(out / f"py_Q_sim_{m}.csv", f["Q"], delimiter=",", fmt="%.17g")
    summary["R_versions"] = R["versions"]

    # 1. scoring
    srows = []
    for nm, r in R["scoring"].items():
        p = py_scores[nm]
        srows.append({"case": nm, "n": r["n"], "k": r["k"],
                      "abs_diff_NN": abs(p["nn"] - r["nn"]), "same_NN_counted": p["nn_counted"] == r["nn_counted"],
                      "ARI_between_cuts_py_R": dp.adjusted_rand(p["cut"], r["cut"]),
                      "ARI_between_pam_py_R": dp.adjusted_rand(p["pam"], r["pam"]),
                      "abs_diff_pam_objective": abs(p["obj"] - r["pam_objective"]),
                      "abs_diff_ARI_formula_cut": abs(p["ari_cut"] - r["ari_R_on_py_cut"]),
                      "abs_diff_ARI_formula_pam": abs(p["ari_pam"] - r["ari_R_on_py_pam"]),
                      "abs_diff_ARI_cut_vs_R_pipeline": abs(p["ari_cut"] - r["ari_cut_R"]),
                      "abs_diff_ARI_pam_vs_R_pipeline": abs(p["ari_pam"] - r["ari_pam_R"]),
                      "NN": p["nn"], "ARI_cut": p["ari_cut"], "ARI_pam": p["ari_pam"]})
    summary["scoring_vs_R"] = {
        "cases": len(srows), "max_abs_diff_NN": max(r["abs_diff_NN"] for r in srows),
        "all_NN_counted_equal": all(r["same_NN_counted"] for r in srows),
        "min_ARI_between_UPGMA_cuts_py_R": min(r["ARI_between_cuts_py_R"] for r in srows),
        "min_ARI_between_PAM_partitions_py_R": min(r["ARI_between_pam_py_R"] for r in srows),
        "max_abs_diff_PAM_objective": max(r["abs_diff_pam_objective"] for r in srows),
        "max_abs_diff_ARI_formula": max(max(r["abs_diff_ARI_formula_cut"], r["abs_diff_ARI_formula_pam"]) for r in srows),
        "max_abs_diff_ARI_full_pipeline": max(max(r["abs_diff_ARI_cut_vs_R_pipeline"], r["abs_diff_ARI_pam_vs_R_pipeline"])
                                              for r in srows),
        "ARI_implementation_in_R": "direct transcription of Hubert & Arabie (1985); mclust/aricode not installed",
        "per_case": srows}

    # 2.-4. Mk
    mrows, arows, simrows = [], [], []
    for cn in ("real", "sim"):
        T, X, states, fits = mkpy[cn]
        rr = R["mk"][cn]
        assert rr["states"] == states, (rr["states"], states)
        key = {frozenset(t.name for t in dp.tips_of(T.nodes[v])): v for v in T.inner_ix}
        rnodes = [key[frozenset(s.split(","))] for s in rr["node_tips"]]
        k = len(states)
        for m in ("ER", "SYM", "ARD"):
            f = fits[m]; rm = rr[m]
            QR = _mat(rm["Q"])
            ll_at_R = dp.mk_loglik(T, X, QR, "equal")
            row = {"case": cn, "model": m, "n_tips": len(T.names), "states": k,
                   "logLik_descriptron": f["logLik"], "logLik_fitMk": rm["logLik"],
                   "abs_diff_logLik_at_same_Q_vs_fitMk_fixedQ": abs(f["logLik"] - rm["logLik_at_descriptron_Q"])
                   if rm["logLik_at_descriptron_Q"] is not None else None,
                   "abs_diff_our_lik_at_fitMk_Q": abs(ll_at_R - rm["logLik"]),
                   "maximised_diff_descriptron_minus_fitMk": f["logLik"] - rm["logLik"],
                   "ace_minus_fitMk_minus_logk": (rm["ace_loglik"] - rm["logLik"] - np.log(k))
                   if rm["ace_loglik"] is not None else None,
                   "maximised_diff_descriptron_minus_ace_minus_logk": (f["logLik"] - (rm["ace_loglik"] - np.log(k)))
                   if rm["ace_loglik"] is not None else None}
            off = ~np.eye(k, dtype=bool)
            # rates: absolute difference relative to the largest rate (a rate at 0 in R is at the lower bound here)
            row["max_diff_rates_over_max_rate_vs_fitMk"] = float(np.max(np.abs(f["Q"][off] - QR[off])) / QR[off].max())
            if rm.get("ace_Q") is not None:
                QA = _rateQ(rm["ace_Q"])
                row["max_diff_rates_over_max_rate_vs_ace"] = float(np.max(np.abs(f["Q"][off] - QA[off])) / QA[off].max())
            mrows.append(row)
            # 3. marginal ancestral states at fitMk's Q
            post = dp.mk_marginal(T, X, QR, "equal")
            ours = post[rnodes]
            d_ancr = float(np.abs(ours - _mat(rm["anc"])).max())
            d_ace = None
            if rm.get("ace_anc") is not None and rm.get("ace_Q") is not None:
                d_ace = float(np.abs(dp.mk_marginal(T, X, _rateQ(rm["ace_Q"]), "equal")[rnodes] - _mat(rm["ace_anc"])).max())
            arows.append({"case": cn, "model": m, "nodes": len(rnodes), "max_abs_diff_vs_ancr_at_fitMk_Q": d_ancr,
                          "max_abs_diff_vs_ace_marginalFALSE_at_ace_Q": d_ace})
        # 4. stochastic maps
        smr = rr["simmap"]
        Q = _mat(smr["Q"])
        nsim = smr["nsim"]
        import time
        t0 = time.time()
        maps = dp.simmaps(T, X, Q, nsim, rng)
        el = time.time() - t0
        C = np.array([dp.simmap_counts(T, mm, k)[0] for mm in maps]); TM = np.array([dp.simmap_counts(T, mm, k)[1] for mm in maps])
        RC = _mat(smr["count"]); cols = smr["count_cols"]
        zs = []
        for ci, cname in enumerate(cols):
            if cname == "N":
                ours_v = C.sum((1, 2))
            else:
                i, j = [states.index(x) for x in cname.split(",")]
                ours_v = C[:, i, j]
            rv = RC[:, ci]
            se = np.sqrt(ours_v.var(ddof=1) / len(ours_v) + rv.var(ddof=1) / len(rv))
            z = (ours_v.mean() - rv.mean()) / se if se > 0 else 0.0
            zs.append({"case": cn, "quantity": f"count {cname}", "descriptron_mean": float(ours_v.mean()),
                       "make_simmap_mean": float(rv.mean()), "z": float(z)})
        RT = _mat(smr["times"]); tcols = smr["times_cols"]
        for ci, cname in enumerate(tcols):
            if cname == "total":
                continue
            ours_v = TM[:, states.index(cname)]; rv = RT[:, ci]
            se = np.sqrt(ours_v.var(ddof=1) / len(ours_v) + rv.var(ddof=1) / len(rv))
            zs.append({"case": cn, "quantity": f"time in {cname}", "descriptron_mean": float(ours_v.mean()),
                       "make_simmap_mean": float(rv.mean()), "z": float((ours_v.mean() - rv.mean()) / se) if se > 0 else 0.0})
        simrows += zs
        summary.setdefault("simmap_timing_seconds", {})[cn] = {"descriptron": el, "make_simmap": smr["seconds"],
                                                               "maps": nsim, "Q_model": smr["model"]}
    summary["mk_vs_fitMk_ace"] = {
        "max_abs_diff_logLik_same_Q_vs_fitMk_fixedQ": max(r["abs_diff_logLik_at_same_Q_vs_fitMk_fixedQ"] for r in mrows
                                                          if r["abs_diff_logLik_at_same_Q_vs_fitMk_fixedQ"] is not None),
        "max_abs_diff_our_likelihood_at_fitMk_Q": max(r["abs_diff_our_lik_at_fitMk_Q"] for r in mrows),
        "min_maximised_diff_descriptron_minus_fitMk": min(r["maximised_diff_descriptron_minus_fitMk"] for r in mrows),
        "max_abs_ace_minus_fitMk_minus_logk": max(abs(r["ace_minus_fitMk_minus_logk"]) for r in mrows
                                                  if r["ace_minus_fitMk_minus_logk"] is not None),
        "max_abs_ace_minus_fitMk_minus_logk_where_ace_reached_fitMk_optimum": max(
            (abs(r["ace_minus_fitMk_minus_logk"]) for r in mrows if r["ace_minus_fitMk_minus_logk"] is not None
             and abs(r["ace_minus_fitMk_minus_logk"]) < 1e-3), default=None),
        "max_rate_diff_over_max_rate_vs_fitMk_ER_SYM": max(r["max_diff_rates_over_max_rate_vs_fitMk"] for r in mrows
                                                           if r["model"] != "ARD"),
        "per_fit": mrows}
    summary["ancestral_states_vs_ancr_ace"] = {
        "max_abs_diff_vs_ancr": max(r["max_abs_diff_vs_ancr_at_fitMk_Q"] for r in arows),
        "max_abs_diff_vs_ace": max((r["max_abs_diff_vs_ace_marginalFALSE_at_ace_Q"] for r in arows
                                    if r["max_abs_diff_vs_ace_marginalFALSE_at_ace_Q"] is not None), default=None),
        "per_fit": arows}
    zz = np.array([r["z"] for r in simrows])
    summary["simmap_vs_make_simmap"] = {"quantities": len(zz), "max_abs_z": float(np.abs(zz).max()),
                                        "n_abs_z_over_2": int((np.abs(zz) > 2).sum()),
                                        "n_abs_z_over_3": int((np.abs(zz) > 3).sum()), "per_quantity": simrows}

    # 5. OUM / BMS
    if "oum" in R:
        orows = []
        for pn, reg in paintings.items():
            labs = sorted(set(reg)); codes = np.array([labs.index(x) for x in reg])
            for j in range(kpc):
                f = dp.hansen_fit(OT, Yt[:, j], reg, starts[j][0], starts[j][1])
                b = dp.bms_fit(OT, Yt[:, j], codes, len(labs))
                rf = R["oum"][pn][j]
                th = {k_: float(v) for k_, v in rf["theta"].items()}
                rs2 = {k_: float(v) for k_, v in rf["bms_sig2"].items()}
                orows.append({"painting": pn, "PC": j + 1, "regimes": len(labs),
                              "abs_diff_loglik": abs(f["loglik"] - rf["loglik"]),
                              "rel_diff_alpha": abs(f["alpha"] - rf["alpha"]) / abs(rf["alpha"]),
                              "rel_diff_sigma2": abs(f["sigma2"] - rf["sigma2"]) / abs(rf["sigma2"]),
                              "max_abs_diff_theta": max(abs(f["theta"][k_] - v) for k_, v in th.items()),
                              "loglik_ouch": rf["loglik"], "alpha_ouch": rf["alpha"],
                              "BMS_loglik_descriptron": b["loglik"], "BMS_loglik_brownie_lite": rf["bms_loglik"],
                              "BMS_diff_descriptron_minus_brownie": b["loglik"] - rf["bms_loglik"],
                              "BMS_max_rel_diff_sigma2": max(abs(b["sigma2"][labs.index(k_)] - v) / v for k_, v in rs2.items()),
                              "brownie_converged": rf["bms_converged"]})
        summary["oum_vs_ouch_hansen"] = {
            "fits": len(orows), "paintings": list(paintings), "PCs": kpc,
            "max_abs_diff_loglik": max(r["abs_diff_loglik"] for r in orows),
            "max_rel_diff_alpha": max(r["rel_diff_alpha"] for r in orows),
            "max_rel_diff_sigma2": max(r["rel_diff_sigma2"] for r in orows),
            "max_abs_diff_theta": max(r["max_abs_diff_theta"] for r in orows),
            "BMS_max_abs_diff_loglik_vs_brownie_lite": max(abs(r["BMS_diff_descriptron_minus_brownie"]) for r in orows),
            "BMS_min_diff_descriptron_minus_brownie": min(r["BMS_diff_descriptron_minus_brownie"] for r in orows),
            "BMS_max_rel_diff_sigma2": max(r["BMS_max_rel_diff_sigma2"] for r in orows),
            "per_fit": orows}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    brief = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("per_")} if isinstance(v, dict) else v)
             for k, v in summary.items()}
    print(json.dumps(brief, indent=1, default=float))


if __name__ == "__main__":
    main()
