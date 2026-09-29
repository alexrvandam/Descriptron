#!/usr/bin/env python3
"""Do the colour-pattern features separate species, and do they still once overall wing colour is removed?
PERMANOVA (Euclidean, = MANOVA sums of squares on the full PC scores) of species on each method's features, before
and after regressing out each wing's mean L*a*b* (3 covariates). Same 48 wings / 9 species as the colour benchmark.

  python colour_species_structure_v1.py --colour_dir <colour_benchmark> --descriptron_features <color_homology_features_whole_wing_*.csv> \
      --out_csv colour_species_structure.csv
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA


def permanova(X, y, n_perm=9999, seed=20260929):
    X = X - X.mean(0); sst = (X ** 2).sum(); labs = np.unique(y); g, n = len(labs), len(y)
    def ssb(lbl):
        return sum((lbl == l).sum() * (X[lbl == l].mean(0) ** 2).sum() for l in labs)
    b = ssb(y); f = (b / (g - 1)) / ((sst - b) / (n - g))
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y); bp = ssb(yp)
        if (bp / (g - 1)) / ((sst - bp) / (n - g)) >= f:
            cnt += 1
    return b / sst, f, (cnt + 1) / (n_perm + 1)


def loo_1nn(S, y, per_wing=False):
    """Leave one wing out, give it the species of its nearest remaining wing; returns (correct, scored)."""
    ok = [i for i in range(len(y)) if (y == y[i]).sum() >= 2]; hits = []
    for i in ok:
        d = np.linalg.norm(S - S[i], axis=1); d[i] = np.inf; hits.append(int(y[d.argmin()] == y[i]))
    return (np.array(hits) if per_wing else (sum(hits), len(ok)))


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test on paired right/wrong calls for the same wings."""
    from scipy.stats import binomtest
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    return (1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10


def residualise(S, C):
    C1 = np.column_stack([np.ones(len(C)), C]); beta, *_ = np.linalg.lstsq(C1, S, rcond=None)
    return S - C1 @ beta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colour_dir", required=True); ap.add_argument("--descriptron_features", required=True)
    ap.add_argument("--out_csv", required=True); ap.add_argument("--n_perm", type=int, default=9999)
    ap.add_argument("--n_pcs", type=int, default=10,
                    help="principal components per method (patternize and Colormesh scores were saved for PC1-10)")
    a = ap.parse_args()
    cdir = Path(a.colour_dir); res = cdir / "results"
    sp = pd.read_csv(cdir / "inputs" / "specimens.csv").sort_values("order")
    F = pd.read_csv(a.descriptron_features)
    X = sp[["filename", "species"]].merge(F, on="filename", how="left"); y = X.species.values
    feats = [c for c in F.columns if c != "filename"]
    M = X[feats].astype(float); M = M.loc[:, M.notna().mean() > 0.9]; M = M.fillna(M.mean()); M = M.loc[:, M.std() > 0]
    P_std = PCA().fit_transform(((M - M.mean()) / M.std()).values)
    others = {}
    for m in ("patternize", "Colormesh"):
        t = pd.read_csv(res / f"pc_scores_{m}.csv"); t = t.set_index("id").loc[[Path(f).stem for f in X.filename.str.split(".tif").str[0] + ".tif"]]
        assert (t.species.values == y).all(); others[m] = t[[c for c in t.columns if c.startswith("PC")]].values
    mean_col = np.column_stack([X[[c for c in feats if c.endswith(f"_{k}_mean_abs")]].mean(1) for k in ("L", "a", "b")])
    mean_col = (mean_col - mean_col.mean(0)) / mean_col.std(0)
    # all-PC accuracy for the R packages comes from the benchmark's own summary (only their PC1-10 scores were saved)
    summ = pd.read_csv(res / "summary_per_method.csv").set_index("method")
    rows = []; hits_desc = loo_1nn(P_std[:, :5], y, per_wing=True)
    for name, S in (("Descriptron", P_std), ("patternize", others["patternize"]), ("Colormesh", others["Colormesh"]),
                    ("mean wing colour only", mean_col)):
        c5, nsc = loo_1nn(S[:, :5], y)
        if name == "Descriptron":
            call, _ = loo_1nn(S, y); acc_all = call / nsc
        elif name in summ.index:
            acc_all = float(summ.loc[name, "loo1nn_acc_allPCs"]); call = round(acc_all * nsc)
        else:
            acc_all, call = np.nan, np.nan
        S = S[:, :a.n_pcs]
        r2, f, p = permanova(S, y, a.n_perm)
        lo5, hi5 = wilson(c5, nsc)
        pm, only_desc, only_other = mcnemar_exact(hits_desc, loo_1nn(S[:, :5], y, per_wing=True))
        lo_a, hi_a = (wilson(call, nsc) if not (isinstance(call, float) and np.isnan(call)) else (np.nan, np.nan))
        row = dict(method=name, n_wings=len(y), n_species=len(set(y)), n_dims=S.shape[1],
                   loo_correct_PC1to5=c5, loo_scored=nsc, loo_acc_PC1to5=c5 / nsc, loo_ci95_PC1to5=f"{lo5:.2f}-{hi5:.2f}",
                   mcnemar_p_vs_Descriptron_PC1to5=pm, wings_only_Descriptron_right=only_desc,
                   wings_only_this_right=only_other,
                   loo_ci95_allPCs=(f"{lo_a:.2f}-{hi_a:.2f}" if not np.isnan(lo_a) else ""),
                   loo_correct_allPCs=call, loo_acc_allPCs=acc_all,
                   species_R2=r2, pseudo_F=f, p=p)
        if name != "mean wing colour only":
            Sr = residualise(S, mean_col); r2r, fr, pr = permanova(Sr, y, a.n_perm)
            row.update(species_R2_beyond_mean_colour=r2r, pseudo_F_beyond=fr, p_beyond=pr)
        rows.append(row); print(row)
    pd.DataFrame(rows).to_csv(a.out_csv, index=False); print(a.out_csv)


if __name__ == "__main__":
    main()
