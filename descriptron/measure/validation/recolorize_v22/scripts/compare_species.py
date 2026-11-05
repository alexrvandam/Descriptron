#!/usr/bin/env python3
"""Per-species colour summaries from each method's palettes (palettes_<NAME>.csv from compare_palettes.py) and
their agreement between methods.
Per specimen: k (number of colours), pixel-weighted mean colour of the palette (L*, a*, b*, C*ab), dominant colour
(largest cluster) and its area fraction. Per species: mean over specimens. Agreement per method pair, over species:
Pearson and Spearman r of each summary, and CIEDE2000 between the two methods' species mean / dominant colours.
Extra references: 'raw' = mean CIELAB of all specimen pixels (raw_mean_lab.csv); 'colhom' = Descriptron colour
homology specimen block of the moth run (L_mean/2.55, a_mean, b_mean; L_mean is stored in OpenCV 0-255 units).
Usage: compare_species.py --compare_dir <dir> --methods A,B,... --pairs A:B,C:D [--colhom_csv <csv>] --out_dir <dir>"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from skimage.color import deltaE_ciede2000

ap = argparse.ArgumentParser()
ap.add_argument("--compare_dir", required=True); ap.add_argument("--methods", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--colhom_csv"); ap.add_argument("--out_dir", required=True)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
spec = {}
for n in a.methods.split(","):
    p = pd.read_csv(os.path.join(a.compare_dir, f"palettes_{n}.csv"))
    g = p.groupby("image")
    rows = []
    for im, d in g:
        w = d.frac / d.frac.sum(); mu = (d[["L", "a", "b"]].values * w.values[:, None]).sum(0); dom = d.loc[d.frac.idxmax()]
        rows.append(dict(image=im, species=d.species.iloc[0], k=len(d), L=mu[0], a=mu[1], b=mu[2], C=np.hypot(mu[1], mu[2]),
                         dom_L=dom.L, dom_a=dom.a, dom_b=dom.b, dom_frac=dom.frac))
    spec[n] = pd.DataFrame(rows)
raw = pd.read_csv(os.path.join(a.compare_dir, "raw_mean_lab.csv")); raw["C"] = np.hypot(raw.a, raw.b); raw["image"] = raw.image
spec["raw"] = raw
if a.colhom_csv:
    c = pd.read_csv(a.colhom_csv)
    c = pd.DataFrame(dict(image=c.file, species=c.species, L=c.L_mean / 2.55, a=c.a_mean, b=c.b_mean))
    c["C"] = np.hypot(c.a, c.b); spec["colhom"] = c
sp = {n: d.groupby("species").mean(numeric_only=True) for n, d in spec.items()}
for n, d in sp.items():
    d.to_csv(os.path.join(a.out_dir, f"species_summary_{n}.csv"))
rows = []
for pr in a.pairs.split(","):
    A, B = pr.split(":")
    if A not in sp or B not in sp:
        continue
    S = sp[A].index.intersection(sp[B].index); X, Y = sp[A].loc[S], sp[B].loc[S]
    r = dict(pair=f"{A} vs {B}", n_species=len(S))
    for v in ["k", "L", "a", "b", "C", "dom_frac"]:
        if v in X and v in Y:
            r[f"pearson_{v}"] = pearsonr(X[v], Y[v])[0]; r[f"spearman_{v}"] = spearmanr(X[v], Y[v])[0]
            r[f"median_absdiff_{v}"] = float(np.median(np.abs(X[v] - Y[v])))
    de = deltaE_ciede2000(X[["L", "a", "b"]].values, Y[["L", "a", "b"]].values)
    r.update(mean_colour_dE00_median=float(np.median(de)), mean_colour_dE00_q25=float(np.percentile(de, 25)),
             mean_colour_dE00_q75=float(np.percentile(de, 75)), mean_colour_dE00_max=float(de.max()))
    if "dom_L" in X and "dom_L" in Y:
        dd = deltaE_ciede2000(X[["dom_L", "dom_a", "dom_b"]].values, Y[["dom_L", "dom_a", "dom_b"]].values)
        r.update(dominant_dE00_median=float(np.median(dd)), dominant_dE00_q75=float(np.percentile(dd, 75)), dominant_dE00_max=float(dd.max()))
    rows.append(r)
out = pd.DataFrame(rows); out.to_csv(os.path.join(a.out_dir, "species_summary_agreement.csv"), index=False)
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(out[[c for c in out.columns if c.startswith(("pair", "pearson_k", "spearman_k", "pearson_L", "pearson_a", "pearson_b", "pearson_C", "mean_colour_dE00_median", "dominant_dE00_median"))]].round(3).to_string(index=False))
