#!/usr/bin/env python3
"""descriptron_phylo's evolutionary models and variable rates against geiger and phytools.

R simulates the data and fits it; Descriptron fits the same trees and data:
  1. models: BM, OU, EB, lambda and white noise by geiger::fitContinuous, on an ultrametric tree (phytools::pbtree,
     60 tips) and the same tree made non-ultrametric (terminal branches stretched by 0.5-1.5x, where geiger switches
     OU to its covariance method), for data simulated under BM, OU, EB, lambda = 0.5 and white noise; and on a
     real tree given with --tree (for example a phylogram from IQ-TREE) for BM and OU data
     compared: log-likelihood, AICc and the model parameter
     Two differences are geiger's, not Descriptron's, and are reported apart from the comparison:
     - geiger's pruning algorithm assumes every node has two children: on a tree with a basal trifurcation (an
       unrooted IQ-TREE tree) its likelihoods are wrong, even for white noise, which does not use the tree (matrix()
       warns "data length is not a sub-multiple"). The real tree is therefore resolved with ape::multi2di (a
       zero-length branch, the same covariance) before geiger sees it.
     - OU with alpha -> 0 on a non-ultrametric tree: geiger computes 1 - exp(-2 alpha s) directly, which loses its
       digits for alpha ~ 1e-12, and can report a log-likelihood above the Brownian-motion maximum (impossible: OU
       tends to BM as alpha -> 0). Descriptron uses expm1. Such rows are flagged geiger_above_BM_limit.
  2. rates: phytools::multirateBM (lambda = 1) on a 40-tip tree whose one clade evolves 10x faster
     compared: the log rate at every node (matched by its descendant tips), the log-likelihood and the penalised
     log-likelihood (Descriptron's should be at least as high: both maximise the same function)

  python validate_phylo_models_vs_r.py --rscript Rscript --out-dir val_models/ [--tree real.nwk]
"""
import argparse, csv, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np


def _sibling(name):
    """measure/<name> in the source tree; beside this file once packaged (tools/)."""
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)


R_CODE = r'''
args <- commandArgs(TRUE); d <- args[1]; real <- if (length(args) > 1) args[2] else ""
suppressMessages({library(geiger); library(phytools)})
set.seed(20261104)
fitall <- function(phy, x, tag, rows) {
  for (m in c("BM", "OU", "EB", "lambda", "white")) {
    f <- suppressWarnings(fitContinuous(phy, x, model = m, control = list(niter = 50)))$opt
    par <- switch(m, OU = f$alpha, EB = f$a, lambda = f$lambda, NA)
    rows[[length(rows) + 1]] <- data.frame(data = tag, model = m, lnL = f$lnL, aicc = f$aicc, sigsq = f$sigsq,
                                           z0 = f$z0, param = par)
  }
  rows
}
tu <- pbtree(n = 60, scale = 1); tu$tip.label <- paste0("s", 1:60)
tn <- tu; te <- tn$edge[, 2] <= Ntip(tn); tn$edge.length[te] <- tn$edge.length[te] * runif(sum(te), 0.5, 1.5)
write.tree(tu, file.path(d, "tree_ultra.nwk")); write.tree(tn, file.path(d, "tree_nonultra.nwk"))
rows <- list(); dat <- list()
for (tt in c("ultra", "nonultra")) {
  phy <- if (tt == "ultra") tu else tn
  sims <- list(BM = fastBM(phy, sig2 = 1),
               OU = fastBM(phy, sig2 = 1, alpha = 3, theta = 0, a = 0),
               EB = fastBM(geiger:::rescale.phylo(phy, "EB", -4), sig2 = 1),
               lambda = fastBM(geiger:::rescale.phylo(phy, "lambda", 0.5), sig2 = 1),
               white = setNames(rnorm(Ntip(phy)), phy$tip.label))
  for (s in names(sims)) {
    tag <- paste(tt, s, sep = ":"); x <- sims[[s]][phy$tip.label]
    dat[[tag]] <- x; rows <- fitall(phy, x, tag, rows)
  }
}
if (nchar(real)) {
  phy <- multi2di(read.tree(real), random = FALSE); phy$node.label <- NULL
  write.tree(phy, file.path(d, "tree_real.nwk"))
  for (s in c("BM", "OU")) {
    x <- if (s == "BM") fastBM(phy, sig2 = 1) else fastBM(phy, sig2 = 1, alpha = 10, theta = 0, a = 0)
    tag <- paste("real", s, sep = ":"); dat[[tag]] <- x[phy$tip.label]; rows <- fitall(phy, x[phy$tip.label], tag, rows)
  }
}
write.csv(do.call(rbind, rows), file.path(d, "R_fitContinuous.csv"), row.names = FALSE)
for (tag in names(dat)) write.csv(data.frame(species = names(dat[[tag]]), y = dat[[tag]]),
                                  file.path(d, paste0("data_", gsub(":", "_", tag), ".csv")), row.names = FALSE)
# variable-rate Brownian motion: one clade 10x faster
tr <- pbtree(n = 40, scale = 1); tr$tip.label <- paste0("r", 1:40)
cl <- getDescendants(tr, Ntip(tr) + 2); fast <- tr; ee <- fast$edge[, 2] %in% cl
fast$edge.length[ee] <- fast$edge.length[ee] * 10
x <- fastBM(fast, sig2 = 1)[tr$tip.label]
write.tree(tr, file.path(d, "tree_rates.nwk"))
write.csv(data.frame(species = names(x), y = x), file.path(d, "data_rates.csv"), row.names = FALSE)
fit <- multirateBM(tr, x, lambda = 1, quiet = TRUE)
pen <- -phytools:::log_lik(log(fit$sig2), tree = tr, x = x, lambda = 1)
desc <- sapply(1:(Ntip(tr) + tr$Nnode), function(nd)
  if (nd <= Ntip(tr)) tr$tip.label[nd] else paste(sort(tr$tip.label[getDescendants(tr, nd)[getDescendants(tr, nd) <= Ntip(tr)]]), collapse = ","))
write.csv(data.frame(node = desc, sig2 = fit$sig2), file.path(d, "R_multirate_sig2.csv"), row.names = FALSE)
writeLines(jsonlite::toJSON(list(logLik = as.numeric(fit$logLik), penalised = pen, convergence = fit$convergence),
                            auto_unbox = TRUE, digits = NA), file.path(d, "R_multirate.json"))
'''


def load_dp():
    spec = importlib.util.spec_from_file_location("descriptron_phylo", _sibling("descriptron_phylo.py"))
    dp = importlib.util.module_from_spec(spec); sys.modules["descriptron_phylo"] = dp; spec.loader.exec_module(dp)
    return dp


def read_y(path, tips):
    d = {r["species"]: float(r["y"]) for r in csv.DictReader(open(path))}
    return np.array([d[t.name] for t in tips])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rscript", default="Rscript"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tree", default=None, help="a real tree to fit BM and OU data on as well")
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "r_side.R").write_text(R_CODE)
    cmd = [a.rscript, str(out / "r_side.R"), str(out)] + ([str(Path(a.tree).resolve())] if a.tree else [])
    subprocess.run(cmd, check=True)
    dp = load_dp()

    rows, worst = [], {"lnL": 0.0, "aicc": 0.0, "param_rel": 0.0}
    rfits = list(csv.DictReader(open(out / "R_fitContinuous.csv")))
    bm_ll = {r["data"]: float(r["lnL"]) for r in rfits if r["model"] == "BM"}
    for r in rfits:
        tt, sim = r["data"].split(":")
        root = dp.parse_newick((out / f"tree_{tt}.nwk").read_text()); tips = dp.tips_of(root)
        y = read_y(out / f"data_{tt}_{sim}.csv", tips)
        C = dp.vcv(tips, tips)
        f = dp.fit_model(y, C, r["model"], dp.default_bounds(C))
        k = dp.MODEL_K[r["model"]]; n = len(y)
        aicc = -2 * f["logLik"] + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        d_ll, d_aicc = f["logLik"] - float(r["lnL"]), aicc - float(r["aicc"])
        rp = float(r["param"]) if r["param"] not in ("NA", "") else np.nan
        rel = abs(f["value"] - rp) / max(abs(rp), 1e-12) if np.isfinite(rp) else np.nan
        flat = f["at_bound"] or (r["model"] == "OU" and rp < 1e-6) or (r["model"] == "EB" and rp > -1e-5)
        above = r["model"] == "OU" and rp < 1e-6 and float(r["lnL"]) > bm_ll[r["data"]] + 1e-6
        rows.append({"data": r["data"], "model": r["model"], "R_lnL": float(r["lnL"]), "D_lnL": f["logLik"],
                     "d_lnL": d_ll, "R_aicc": float(r["aicc"]), "D_aicc": aicc, "R_param": rp, "D_param": f["value"],
                     "param_rel_diff": rel, "param_at_bound_or_flat": bool(flat), "geiger_above_BM_limit": above})
        if not above:
            worst["lnL"] = max(worst["lnL"], abs(d_ll)); worst["aicc"] = max(worst["aicc"], abs(d_aicc))
        if np.isfinite(rel) and not flat:
            worst["param_rel"] = max(worst["param_rel"], rel)
    with open(out / "models_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    root = dp.parse_newick((out / "tree_rates.nwk").read_text()); tips = dp.tips_of(root)
    y = read_y(out / "data_rates.csv", tips)
    R = dp.multirate_bm(root, tips, y, 1.0)
    rs = {r["node"]: float(r["sig2"]) for r in csv.DictReader(open(out / "R_multirate_sig2.csv"))}
    key = lambda n: n.name if not n.kids else ",".join(sorted(t.name for t in dp.tips_of(n)))
    ours = np.log([R["node_rate"][i] for i, n in enumerate(R["nodes"])])
    theirs = np.log([rs[key(n)] for n in R["nodes"]])
    rj = json.loads((out / "R_multirate.json").read_text())
    rates = {"n_nodes": len(ours), "corr_log_sig2": float(np.corrcoef(ours, theirs)[0, 1]),
             "max_abs_diff_log_sig2": float(np.abs(ours - theirs).max()),
             "D_logLik": R["logLik"], "R_logLik": rj["logLik"],
             "D_penalised": R["penalised_logLik"], "R_penalised": rj["penalised"],
             "R_convergence": rj["convergence"], "D_converged": R["converged"]}
    summary = {"models_compared": len(rows), "models_max_abs_d_lnL": worst["lnL"], "models_max_abs_d_aicc": worst["aicc"],
               "models_max_param_rel_diff_off_bounds": worst["param_rel"],
               "geiger_above_BM_limit": [f"{r['data']} {r['model']} (R {r['R_lnL']:.6f}, BM max {bm_ll[r['data']]:.6f}, "
                                         f"Descriptron {r['D_lnL']:.6f})" for r in rows if r["geiger_above_BM_limit"]],
               "rates": rates}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    bad = [r for r in rows if abs(r["d_lnL"]) > 1e-3 and not r["geiger_above_BM_limit"]]
    for r in bad:
        print("lnL differs:", r["data"], r["model"], f"R {r['R_lnL']:.5f}  Descriptron {r['D_lnL']:.5f}")


if __name__ == "__main__":
    main()
