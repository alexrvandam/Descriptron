#!/usr/bin/env python3
"""descriptron_trait_stats against R, on real Descriptron trait tables.

Given a trait table and group labels (and optionally a size column and a categorical table), the same principal
component scores are analysed by descriptron_trait_stats and by R:
  PERMANOVA R2 and F (species; after removing a covariate; one species pair)  vs vegan::adonis2 (Euclidean)
  allometry R2 (trait set ~ log size)                                          vs vegan::adonis2
  leave-one-out nearest-neighbour assignments                                  vs class::knn.cv(k = 1)
  Wilson 95% interval                                                          vs prop.test(correct = FALSE)
  exact McNemar P                                                              vs binom.test on the discordant pairs
  chi-square behind Cramer's V                                                 vs chisq.test(correct = FALSE)
Deterministic quantities are compared by relative difference; permutation P-values within sampling error.

  python validate_trait_stats.py --traits colour.csv --groups groups.csv [--size table.csv:col] \
      [--categorical states.csv] --rscript /path/Rscript --rlib /path/Rlib --out-dir check/
"""
import argparse, importlib.util, json, subprocess
from pathlib import Path
import numpy as np

R_CODE = r'''
args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths())); d <- args[2]
suppressMessages({library(vegan); library(class)})
P <- as.matrix(read.csv(file.path(d, "pcs.csv"), row.names = 1)); y <- factor(read.csv(file.path(d, "groups.csv"))$g)
out <- list()
set.seed(1); a <- adonis2(dist(P) ~ y, permutations = 999, method = "euclidean")
out$R2 <- a$R2[1]; out$F <- a$F[1]; out$P <- a$"Pr(>F)"[1]
if (file.exists(file.path(d, "cov.csv"))) {
  C <- as.matrix(read.csv(file.path(d, "cov.csv"))); Rz <- residuals(lm(P ~ C))
  set.seed(1); b <- adonis2(dist(Rz) ~ y, permutations = 999); out$R2_beyond <- b$R2[1]; out$F_beyond <- b$F[1]
}
pr <- read.csv(file.path(d, "pair.csv"))$g; m <- y %in% pr
set.seed(1); pw <- adonis2(dist(P[m, ]) ~ factor(as.character(y[m])), permutations = 999)
out$pair_R2 <- pw$R2[1]; out$pair_F <- pw$F[1]
if (file.exists(file.path(d, "size.csv"))) {
  s <- log(read.csv(file.path(d, "size.csv"))$s); set.seed(1); al <- adonis2(dist(P) ~ s, permutations = 999)
  out$allometry_R2 <- al$R2[1]
}
k5 <- read.csv(file.path(d, "loo_idx.csv"))$i
nn <- as.character(knn.cv(P[, 1:5, drop = FALSE], y, k = 1))
write.csv(data.frame(assigned = nn), file.path(d, "R_knn.csv"), row.names = FALSE)
w <- read.csv(file.path(d, "wilson.csv")); pt <- prop.test(w$k, w$n, correct = FALSE)$conf.int
out$wilson_lo <- pt[1]; out$wilson_hi <- pt[2]
mc <- read.csv(file.path(d, "mcnemar.csv")); out$mcnemar_P <- binom.test(mc$a, mc$a + mc$b, 0.5)$p.value
if (file.exists(file.path(d, "cat.csv"))) {
  ct <- read.csv(file.path(d, "cat.csv"), colClasses = "character")
  out$chisq <- unname(suppressWarnings(chisq.test(table(ct$c, ct$g), correct = FALSE))$statistic)
}
writeLines(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA), file.path(d, "R_results.json"))
'''


def _sibling(name):
    """measure/<name> in the source tree; beside this file once packaged (tools/)."""
    here = Path(__file__).resolve()
    return next((c for c in (here.parents[1] / name, here.parent / name) if c.exists()), here.parents[1] / name)


def main(argv=None):
    import pandas as pd
    from sklearn.decomposition import PCA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traits", required=True); ap.add_argument("--groups", required=True)
    ap.add_argument("--size"); ap.add_argument("--categorical")
    ap.add_argument("--n_pcs", type=int, default=10)
    ap.add_argument("--rscript", required=True); ap.add_argument("--rlib", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("ts", _sibling("descriptron_trait_stats.py"))
    ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    gmap = ts.load_groups(a.groups)
    ids, D = ts.load_table(a.traits)
    y = np.array([ts.group_of(i, gmap) for i in ids], dtype=object); keep = np.array([g is not None for g in y])
    ids = [i for i, k in zip(ids, keep) if k]; D = D.loc[keep].reset_index(drop=True); y = y[keep].astype(str)
    cnt = pd.Series(y).value_counts(); k2 = np.isin(y, cnt[cnt >= 2].index)
    ids = [i for i, k in zip(ids, k2) if k]; D = D.loc[k2].reset_index(drop=True); y = y[k2]
    num = ts.numeric_block(D); X = ((num - num.mean()) / num.std()).values
    P = PCA().fit_transform(X)[:, :a.n_pcs]
    pd.DataFrame(P, index=ids, columns=[f"PC{i + 1}" for i in range(P.shape[1])]).to_csv(out / "pcs.csv")
    pd.DataFrame({"g": y}).to_csv(out / "groups.csv", index=False)
    res = {}
    res["R2"], res["F"], res["P"] = ts.permanova(P, y, 999, 1)
    raw = D.apply(pd.to_numeric, errors="coerce")
    parts = [raw[[c for c in raw.columns if c.endswith(s)]] for s in ("_L_mean_abs", "_a_mean_abs", "_b_mean_abs")]
    if all(p.shape[1] for p in parts):
        C = np.column_stack([p.mean(1) for p in parts]); C = (C - C.mean(0)) / C.std(0)
        pd.DataFrame(C, columns=["L", "a", "b"]).to_csv(out / "cov.csv", index=False)
        res["R2_beyond"], res["F_beyond"], _ = ts.permanova(ts.residualise(P, C), y, 999, 1)
    labs = sorted(set(y), key=ts._natural); pair = labs[:2]; m = np.isin(y, pair)
    pd.DataFrame({"g": pair}).to_csv(out / "pair.csv", index=False)
    res["pair_R2"], res["pair_F"], _ = ts.permanova(P[m], y[m], 999, 1)
    if a.size:
        tp, col = a.size.rsplit(":", 1); sids, SD = ts.load_table(tp)
        smap = {}
        for sid, v in zip(sids, pd.to_numeric(SD[col], errors="coerce")):
            for kf in ts._key_forms(sid):
                smap.setdefault(kf, v)
        sv = np.array([next((smap[kf] for kf in ts._key_forms(i) if kf in smap), np.nan) for i in ids], float)
        if np.isfinite(sv).all():
            pd.DataFrame({"s": sv}).to_csv(out / "size.csv", index=False)
            res["allometry_R2"], _ = ts.regression_perm(P, np.log(sv), 99, 1)
    hits = ts.loo_1nn(P[:, :5], y)
    pd.DataFrame({"i": range(len(y))}).to_csv(out / "loo_idx.csv", index=False)
    k = sum(h[1] for h in hits if h[0]); n = sum(h[0] for h in hits)
    pd.DataFrame({"k": [k], "n": [n]}).to_csv(out / "wilson.csv", index=False)
    res["wilson_lo"], res["wilson_hi"] = ts.wilson(k, n)
    hits_all = ts.loo_1nn(P, y)
    pm, a_only, b_only = ts.mcnemar_exact([h[1] for h in hits], [h[1] for h in hits_all])
    pd.DataFrame({"a": [a_only], "b": [b_only]}).to_csv(out / "mcnemar.csv", index=False)
    res["mcnemar_P"] = pm
    if a.categorical:
        cids, CD = ts.load_table(a.categorical); col = CD.columns[0]
        cy = [ts.group_of(i, gmap) for i in cids]; ok = [g is not None and str(v) not in ("", "nan") for g, v in zip(cy, CD[col])]
        cc = CD[col][ok].astype(str).reset_index(drop=True); gg = np.array([g for g, o in zip(cy, ok) if o])
        pd.DataFrame({"c": cc, "g": gg}).to_csv(out / "cat.csv", index=False)
        tab = pd.crosstab(cc.values, gg).values.astype(float); e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        res["chisq"] = float(((tab - e) ** 2 / e).sum())
    (out / "compare.R").write_text(R_CODE)
    subprocess.run([a.rscript, str(out / "compare.R"), a.rlib, str(out)], check=True)
    R = json.load(open(out / "R_results.json"))
    rk = pd.read_csv(out / "R_knn.csv")["assigned"].astype(str).values
    rows = []
    for key in ("R2", "F", "R2_beyond", "F_beyond", "pair_R2", "pair_F", "allometry_R2", "wilson_lo", "wilson_hi", "mcnemar_P", "chisq"):
        if key in res and key in R:
            rows.append({"quantity": key, "descriptron": res[key], "R": R[key], "rel_diff": abs(res[key] - R[key]) / abs(R[key])})
    rows.append({"quantity": "P (permutation)", "descriptron": res["P"], "R": R["P"], "rel_diff": float("nan")})
    agree = sum(h[2] == r for h, r in zip(hits, rk)); rows.append({"quantity": "LOO 1-NN assignments identical",
                                                                    "descriptron": agree, "R": len(rk), "rel_diff": (len(rk) - agree) / len(rk)})
    pd.DataFrame(rows).to_csv(out / "trait_stats_vs_R.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
