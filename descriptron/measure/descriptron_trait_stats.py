#!/usr/bin/env python3
"""descriptron_trait_stats.py - do the species differ, in which traits, and which species does each trait set separate?

For every trait set given (colour pattern, texture, outline or landmark shape, measurements, descriptive categorical
characters, or any other table with one row per specimen), against the species (or any group) labels:

  PCA            principal components of the standardised features; PC1 x PC2 plot with each species' convex hull
  PERMANOVA      species effect on PC1-k (Euclidean; = MANOVA sums of squares), R2, pseudo-F, permutation P;
                 optionally again after removing covariates - each specimen's mean colour (automatic for
                 colour tables: "does the pattern differ beyond overall colour?") and/or log size
  pairwise       PERMANOVA for every pair of species, Benjamini-Hochberg corrected: which species each trait set
                 separates, and which it does not
  identification leave-one-out nearest-neighbour species assignment (PC1-5 and all PCs), overall and per species,
                 with Wilson 95% intervals; exact McNemar tests between trait sets on the same specimens
  allometry      trait set ~ log size (permutation P), when a size column is given
  categorical    for descriptive characters: association of each character with species (Cramer's V, permutation
                 P); the characters also enter PERMANOVA and identification as one-hot codes

The statistics are the ones checked against R: PERMANOVA against vegan::adonis2 (validation/validate_trait_stats.py),
and the benchmark's colour results (Supplementary Text S13.5) are reproduced exactly.

Trait tables: CSV with a specimen id column (--id_col, default: filename / file / specimen / id) and numeric columns.
Descriptron's own tables are read as they are: colour/texture homology features, measurements (all_metrics.csv),
and aligned outlines (aligned_coco.json from the semilandmark step). The id may carry Descriptron's "_<n>" annotation
suffix ("wing01.png_3"); it is matched to the group labels with and without it, and with and without extension.

  python descriptron_trait_stats.py --groups group_labels.csv --out_dir trait_stats/ \
      --traits colour=color_homology/forewing/color_homology_features_forewing_combined_hw1.csv \
      --traits texture=texture_homology/forewing/texture_homology_features_forewing.csv \
      --traits shape=semilandmarks/forewing/aligned_coco.json \
      --categorical states=descriptive_states.csv --size meas=measurements/all_metrics.csv:area_mm2
"""
import argparse, csv, itertools, json, math, re, sys
from pathlib import Path
import numpy as np

COLOUR_MEAN_RX = {"L": r"_L_mean_abs$", "a": r"_a_mean_abs$", "b": r"_b_mean_abs$"}


# ============================================================================ statistics
def _ssb(X, y, labs):
    """between-group sum of squares of column-centred X for labels y (vectorised)."""
    s = 0.0
    for l in labs:
        m = y == l
        if m.any():
            s += m.sum() * float((X[m].mean(0) ** 2).sum())
    return s


def permanova(X, y, n_perm=999, seed=1, exact_max=None):
    """Euclidean PERMANOVA (one factor): R2, pseudo-F, P. Exact enumeration when the number of distinct label
    arrangements is at most n_perm (two groups only), otherwise random permutations."""
    X = np.asarray(X, float); y = np.asarray(y)
    X = X - X.mean(0); sst = float((X ** 2).sum()); labs = np.unique(y); g, n = len(labs), len(y)
    if g < 2 or n <= g or sst == 0:
        return float("nan"), float("nan"), float("nan")
    b = _ssb(X, y, labs); f = (b / (g - 1)) / max((sst - b) / (n - g), 1e-300)
    fstat = lambda bb: (bb / (g - 1)) / max((sst - bb) / (n - g), 1e-300)
    if g == 2:
        n1 = int((y == labs[0]).sum())
        total = math.comb(n, n1)
        if total <= (exact_max or n_perm):
            cnt = 0
            for idx in itertools.combinations(range(n), n1):
                yp = np.full(n, labs[1], dtype=y.dtype); yp[list(idx)] = labs[0]
                cnt += fstat(_ssb(X, yp, labs)) >= f - 1e-12 * abs(f)
            return b / sst, f, cnt / total
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        cnt += fstat(_ssb(X, rng.permutation(y), labs)) >= f - 1e-12 * abs(f)
    return b / sst, f, (cnt + 1) / (n_perm + 1)


def residualise(S, C):
    C1 = np.column_stack([np.ones(len(C)), C]); beta, *_ = np.linalg.lstsq(C1, S, rcond=None)
    return S - C1 @ beta


def loo_1nn(S, y):
    """leave one specimen out, give it the species of its nearest remaining specimen. Returns per-specimen
    (scored, correct, assigned) for specimens whose species has >= 2 specimens."""
    out = []
    for i in range(len(y)):
        if (y == y[i]).sum() < 2:
            out.append((False, False, None)); continue
        d = np.linalg.norm(S - S[i], axis=1); d[i] = np.inf; j = int(d.argmin())
        out.append((True, bool(y[j] == y[i]), y[j]))
    return out


def fit_ratio(S, y):
    """per specimen: distance to the nearest OTHER specimen of its own species / distance to the nearest specimen of
    any other species (same space as loo_1nn). Above 1 = closer to another species than to its own; NaN when the
    species has a single specimen."""
    out = []
    for i in range(len(y)):
        d = np.linalg.norm(S - S[i], axis=1); d[i] = np.inf
        own, oth = d[(y == y[i])], d[(y != y[i])]
        own = own[np.isfinite(own)]
        out.append(float(own.min() / oth.min()) if len(own) and len(oth) and oth.min() > 0 else float("nan"))
    return out


def wilson(k, n, z=1.959963984540054):          # qnorm(0.975), as R
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(a, b):
    """exact two-sided McNemar test on paired right/wrong calls; returns P, n(a right only), n(b right only)."""
    from scipy.stats import binomtest
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return (1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10


def bh(p):
    p = np.asarray(p, float); n = len(p); q = np.full(n, np.nan)
    ok = ~np.isnan(p); pv = p[ok]
    if not len(pv):
        return q
    o = np.argsort(pv); r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; qq = np.empty_like(r); qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def regression_perm(S, x, n_perm=999, seed=1):
    """multivariate regression of S on one predictor: R2 and permutation P (predictor permuted)."""
    S = S - S.mean(0); x = np.asarray(x, float) - np.mean(x); sst = float((S ** 2).sum())
    def r2(xx):
        beta = (xx @ S) / (xx @ xx); return float(((np.outer(xx, beta)) ** 2).sum()) / sst
    obs = r2(x); rng = np.random.default_rng(seed)
    cnt = sum(r2(rng.permutation(x)) >= obs - 1e-12 for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


def cramers_v_perm(col, y, n_perm=999, seed=1):
    """association of one categorical character with the groups: Cramer's V and permutation P (chi-square)."""
    import pandas as pd
    ok = pd.notna(col) & (col.astype(str) != "")
    c, yy = col[ok].astype(str).values, np.asarray(y)[ok.values]
    if len(set(c)) < 2 or len(set(yy)) < 2:
        return float("nan"), float("nan"), int(ok.sum())
    def chi(yv):
        t = pd.crosstab(c, yv).values.astype(float); e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        return float(((t - e) ** 2 / np.where(e > 0, e, 1)).sum()), t.shape
    x2, (r, k) = chi(yy); v = math.sqrt(x2 / (len(c) * max(1, min(r, k) - 1)))
    rng = np.random.default_rng(seed)
    cnt = sum(chi(rng.permutation(yy))[0] >= x2 - 1e-9 for _ in range(n_perm))
    return v, (cnt + 1) / (n_perm + 1), int(ok.sum())


# ============================================================================ input
def _key_forms(s):
    s = str(s).strip(); base = re.sub(r"\.(png|jpe?g|tiff?|bmp)_\d+$", r".\1", s, flags=re.I)
    base = re.sub(r"_\d+$", "", base) if base == s and re.search(r"\.[A-Za-z]{2,4}_\d+$", s) else base
    stem = re.sub(r"\.(png|jpe?g|tiff?|bmp)$", "", base, flags=re.I)
    return [s, base, stem, Path(stem).name, Path(base).name]


def load_groups(path, col=None):
    import pandas as pd
    d = pd.read_csv(path, sep="\t" if str(path).endswith(".tsv") else ",")
    lc = {c.strip().lower(): c for c in d.columns}
    fcol = lc.get("filename") or lc.get("file") or lc.get("specimen") or d.columns[0]
    gcol = col or lc.get("group_label") or lc.get("species") or lc.get("group") or d.columns[1]
    m = {}
    for f, g in zip(d[fcol].astype(str), d[gcol].astype(str)):
        for k in _key_forms(f):
            m.setdefault(k, g)
    return m


def group_of(sid, gmap):
    for k in _key_forms(sid):
        if k in gmap:
            return gmap[k]
    return None


def load_table(path, id_col=None):
    """returns (ids, columns, matrix as DataFrame of raw values)."""
    import pandas as pd
    p = Path(path)
    if p.suffix.lower() == ".json":                  # aligned outline / landmark COCO from the semilandmark step
        d = json.load(open(p)); names = {im["id"]: im["file_name"] for im in d["images"]}
        rows, ids = [], []
        for a in d["annotations"]:
            pts = a.get("keypoints") or (a.get("segmentation") or [[]])[0]
            if a.get("keypoints"):
                xy = np.array(pts, float).reshape(-1, 3)[:, :2]
            else:
                xy = np.array(pts, float).reshape(-1, 2)
                if len(xy) > 2 and np.allclose(xy[0], xy[-1]):
                    xy = xy[:-1]
            rows.append(xy.flatten()); ids.append(names[a["image_id"]])
        k = min(len(r) for r in rows)
        cols = [f"{c}{i // 2 + 1}" for i, c in zip(range(k), itertools.cycle("xy"))]
        return ids, pd.DataFrame([r[:k] for r in rows], columns=cols)
    d = pd.read_csv(p, sep="\t" if p.suffix.lower() in (".tsv", ".txt") else ",", low_memory=False)
    icol = id_col if id_col in (d.columns if id_col else []) else next(
        (c for c in ("filename", "file", "image_filename", "specimen", "specimen_id", "image", "id") if c in d.columns), d.columns[0])
    ids = d[icol].astype(str).tolist()
    return ids, d.drop(columns=[icol])


def numeric_block(D, min_fill=0.9):
    import pandas as pd
    num = D.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().mean() >= min_fill]
    num = num.fillna(num.mean())
    return num.loc[:, num.std() > 0]


# ============================================================================ plots
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
           "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354",
           "#756bb1", "#636363"]


def plot_pca(P, ev, y, title, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    labs = sorted(set(y), key=_natural)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, l in enumerate(labs):
        c = PALETTE[i % len(PALETTE)]; m = y == l; pts = P[m, :2]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=c, label=l, zorder=3, edgecolor="white", lw=0.3)
        if len(pts) >= 3:
            try:
                h = ConvexHull(pts); ax.fill(pts[h.vertices, 0], pts[h.vertices, 1], color=c, alpha=0.15, lw=0.8, ec=c)
            except Exception:
                pass
        elif len(pts) == 2:
            ax.plot(pts[:, 0], pts[:, 1], color=c, lw=0.8)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)" if len(ev) > 1 else "PC2")
    ax.set_title(title, fontsize=8.5)
    ax.legend(fontsize=6, ncol=2 if len(labs) > 12 else 1, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_overview(rows, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(4, 1.3 * len(rows) + 1.5), 3.6))
    x = np.arange(len(rows)); acc = [r["loo_acc_PC1to5"] for r in rows]
    lo = [r["loo_acc_PC1to5"] - r["loo_ci_low_PC1to5"] for r in rows]; hi = [r["loo_ci_high_PC1to5"] - r["loo_acc_PC1to5"] for r in rows]
    ax.bar(x, acc, 0.6, color="#0072B2", yerr=[lo, hi], capsize=3, zorder=3)
    for i, r in enumerate(rows):
        ax.text(i, acc[i] + hi[i] + 0.02, f"{acc[i]:.2f}", ha="center", fontsize=7)
    ch = rows[0]["chance"] if rows else 0
    ax.axhline(ch, ls="--", color="grey", lw=0.8); ax.text(len(rows) - 0.5, ch + 0.01, "chance", fontsize=6, color="grey", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r["trait_set"] for r in rows], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("leave-one-out species accuracy (PC1-5)", fontsize=7.5)
    ax.set_title("Species identified by each trait set (95% CI)", fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_separation(sp, sets, sepmat, path):
    """species x species: how many trait sets separate each pair (pairwise PERMANOVA, q < alpha)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(sp); fig, ax = plt.subplots(figsize=(0.32 * n + 2.5, 0.32 * n + 1.8))
    im = ax.imshow(sepmat, cmap="Blues", vmin=0, vmax=max(1, len(sets)))
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(sp, rotation=90, fontsize=6); ax.set_yticklabels(sp, fontsize=6)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, int(sepmat[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if sepmat[i, j] > len(sets) / 2 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label(f"trait sets separating the pair (of {len(sets)})", fontsize=7)
    ax.set_title("Which species pairs are separated, and by how many trait sets", fontsize=8.5)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


# ============================================================================ main
def analyse_set(name, S_raw, y, ids, a, out, covs):
    """one trait set -> summary row, per-specimen hits, per-species rows, pairwise rows."""
    from sklearn.decomposition import PCA
    Z = S_raw
    P = PCA().fit_transform(Z); ev = np.var(P, axis=0, ddof=1); ev = ev / ev.sum()
    k = min(a.n_pcs, P.shape[1]); Pk = P[:, :k]
    r2, f, p = permanova(Pk, y, a.permutations, a.seed)
    row = dict(trait_set=name, n_specimens=len(y), n_species=len(set(y)), n_features=Z.shape[1], n_pcs_tested=k,
               PC1_var=float(ev[0]), PC2_var=float(ev[1]) if len(ev) > 1 else float("nan"),
               permanova_R2=r2, permanova_F=f, permanova_P=p)
    beyond_txt = []
    for cname, C in covs.items():
        rr, ff, pp = permanova(residualise(Pk, C), y, a.permutations, a.seed)
        row[f"permanova_R2_beyond_{cname}"] = rr; row[f"permanova_P_beyond_{cname}"] = pp
        beyond_txt.append(f"beyond {cname.replace('_', ' ')} R2 {rr:.2f}, P = {pp:.3g}")
    hits5 = loo_1nn(Pk[:, :min(5, k)], y); hitsA = loo_1nn(P, y)
    # per-specimen fit to its own species (4th element; everything else reads [0]-[2])
    hits5 = [h + (r,) for h, r in zip(hits5, fit_ratio(Pk[:, :min(5, k)], y))]
    sc = [h[0] for h in hits5]; c5 = sum(h[1] for h in hits5 if h[0]); cA = sum(h[1] for h in hitsA if h[0]); ns = sum(sc)
    lo5, hi5 = wilson(c5, ns); loA, hiA = wilson(cA, ns)
    scored_species = sorted({yy for yy, s in zip(y, sc) if s})
    row.update(loo_scored=ns, loo_correct_PC1to5=c5, loo_acc_PC1to5=c5 / ns if ns else float("nan"),
               loo_ci_low_PC1to5=lo5, loo_ci_high_PC1to5=hi5, loo_correct_allPCs=cA,
               loo_acc_allPCs=cA / ns if ns else float("nan"), loo_ci_low_allPCs=loA, loo_ci_high_allPCs=hiA,
               chance=1 / len(scored_species) if scored_species else float("nan"))
    if a.size_values is not None:
        ok = ~np.isnan(a.size_values)
        if ok.sum() > 3:
            ar2, ap = regression_perm(Pk[ok], np.log(a.size_values[ok]), a.permutations, a.seed)
            row.update(allometry_R2=ar2, allometry_P=ap)
    title = (f"{name}: PC1 vs PC2\nPERMANOVA (PC1-{k}): species R2 {r2:.2f}, P = {p:.3g}"
             + ("".join("\n" + t for t in beyond_txt)))
    plot_pca(P, ev, y, title, out / f"pca_{_safe(name)}.png")
    # per species
    sp_rows = []
    means = {s: Pk[y == s].mean(0) for s in sorted(set(y), key=_natural)}
    for s in means:
        idx = [i for i in range(len(y)) if y[i] == s and hits5[i][0]]
        kk = sum(hits5[i][1] for i in idx); lo, hi = wilson(kk, len(idx))
        conf = sorted({hits5[i][2] for i in idx if not hits5[i][1]}, key=_natural)
        d = sorted(((float(np.linalg.norm(means[s] - m)), o) for o, m in means.items() if o != s))
        sp_rows.append(dict(trait_set=name, species=s, n=int((y == s).sum()), loo_scored=len(idx), loo_correct=kk,
                            loo_ci_low=lo, loo_ci_high=hi, misassigned_to="; ".join(conf),
                            nearest_species="; ".join(o for _, o in d[:2])))
    # pairwise
    pw = []
    if a.pairwise:
        labs = sorted(set(y), key=_natural)
        for s1, s2 in itertools.combinations(labs, 2):
            m = (y == s1) | (y == s2)
            if (y == s1).sum() < 2 or (y == s2).sum() < 2:
                continue
            rr, ff, pp = permanova(Pk[m], y[m], a.permutations, a.seed)
            pw.append(dict(trait_set=name, species_a=s1, species_b=s2, n_a=int((y == s1).sum()), n_b=int((y == s2).sum()),
                           R2=rr, F=ff, P=pp))
        q = bh([r["P"] for r in pw])
        for r, qq in zip(pw, q):
            r["q_BH"] = qq; r["separated"] = bool(qq < a.alpha)
            r["min_attainable_P"] = 1 / math.comb(r["n_a"] + r["n_b"], r["n_a"])
    return row, hits5, sp_rows, pw


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traits", action="append", default=[], metavar="NAME=TABLE", help="continuous trait table (repeatable)")
    ap.add_argument("--categorical", action="append", default=[], metavar="NAME=TABLE",
                    help="descriptive categorical characters, one row per specimen (repeatable)")
    ap.add_argument("--groups", required=True, help="group labels: CSV with filename and group_label (or species) columns")
    ap.add_argument("--group_col", default=None, help="column of --groups holding the label (default group_label/species)")
    ap.add_argument("--id_col", default=None, help="specimen id column of the trait tables (default: filename/file/...)")
    ap.add_argument("--scale", action="append", default=[], metavar="NAME=standardize|none",
                    help="default standardize (z-scores); use none for Procrustes shape coordinates")
    ap.add_argument("--mean_colour", default="auto", choices=["auto", "off"],
                    help="for colour tables with *_L/_a/_b_mean_abs columns, also test the pattern beyond each "
                         "specimen's mean colour (default auto)")
    ap.add_argument("--size", default=None, metavar="TABLE:COLUMN[:CATEGORY]",
                    help="size per specimen (e.g. measurements/all_metrics.csv:area_mm2:forewing); adds allometry "
                         "and the test beyond log size")
    ap.add_argument("--n_pcs", type=int, default=10, help="principal components entering PERMANOVA (default 10)")
    ap.add_argument("--permutations", type=int, default=999)
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR level for pairwise separation (default 0.05)")
    ap.add_argument("--no_pairwise", dest="pairwise", action="store_false")
    ap.add_argument("--min_per_group", type=int, default=2, help="species with fewer specimens are left out (default 2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    if not a.traits and not a.categorical:
        ap.error("give at least one --traits or --categorical table")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    gmap = load_groups(a.groups, a.group_col)
    scales = dict(s.split("=", 1) for s in a.scale)
    size_map = None
    if a.size:
        parts = a.size.split(":")
        tp, col, cat = (parts + [None])[:3] if len(parts) >= 2 else (a.size, None, None)
        sids, SD = load_table(tp, a.id_col)
        if cat:                                    # measurements tables hold every structure: keep one
            ccol = next((c for c in ("category_name", "category") if c in SD.columns), None)
            keepc = (SD[ccol].astype(str) == cat).values if ccol else np.ones(len(SD), bool)
            sids = [i for i, k in zip(sids, keepc) if k]; SD = SD.loc[keepc]
        vals = pd.to_numeric(SD[col], errors="coerce")
        size_map = {}
        for sid, v in zip(sids, vals):
            for kf in _key_forms(sid):
                size_map.setdefault(kf, v)
    rows, sp_all, pw_all, hits_by_set, cat_rows, notes = [], [], [], {}, [], []
    sets = [(t, False) for t in a.traits] + [(t, True) for t in a.categorical]
    for spec, is_cat in sets:
        name, path = spec.split("=", 1)
        ids, D = load_table(path, a.id_col)
        y = np.array([group_of(i, gmap) for i in ids], dtype=object)
        keep = np.array([g is not None for g in y])
        if (~keep).any():
            notes.append(f"{name}: {int((~keep).sum())} of {len(ids)} specimens have no group label and were left out")
        ids = [i for i, k in zip(ids, keep) if k]; D = D.loc[keep].reset_index(drop=True); y = y[keep].astype(str)
        cnt = pd.Series(y).value_counts(); small = cnt[cnt < a.min_per_group].index
        if len(small):
            notes.append(f"{name}: left out (fewer than {a.min_per_group} specimens): {', '.join(sorted(small, key=_natural))}")
            k2 = ~np.isin(y, small); ids = [i for i, k in zip(ids, k2) if k]; D = D.loc[k2].reset_index(drop=True); y = y[k2]
        if len(set(y)) < 2:
            notes.append(f"{name}: fewer than two groups with data - skipped"); continue
        if is_cat:
            Dc = D.astype(str).replace({"nan": np.nan, "": np.nan})
            Dc = Dc.loc[:, Dc.notna().mean() >= 0.5]
            for c in Dc.columns:
                v, p, n = cramers_v_perm(Dc[c], y, a.permutations, a.seed)
                cat_rows.append(dict(trait_set=name, character=c, n_specimens=n, cramers_V=v, P=p))
            q = bh([r["P"] for r in cat_rows if r["trait_set"] == name])
            for r, qq in zip([r for r in cat_rows if r["trait_set"] == name], q):
                r["q_BH"] = qq
            X = pd.get_dummies(Dc.fillna("NA")).astype(float).values
        else:
            num = numeric_block(D)
            covs = {}
            if a.mean_colour == "auto":
                raw = D.apply(pd.to_numeric, errors="coerce")      # every cell's mean colour, gaps skipped
                parts = [raw[[c for c in raw.columns if re.search(rx, c)]] for rx in COLOUR_MEAN_RX.values()]
                if all(p_.shape[1] for p_ in parts):
                    C = np.column_stack([p_.mean(1) for p_ in parts])
                    C = C[:, np.isfinite(C).all(0) & (np.nanstd(C, 0) > 0)]      # constant or missing channels dropped
                    if C.shape[1]:
                        covs["mean_colour"] = (C - C.mean(0)) / C.std(0)
            X = num.values
            if scales.get(name, "standardize") == "standardize":
                X = (X - X.mean(0)) / X.std(0, ddof=1)
        a.size_values = None
        if size_map is not None and not is_cat:
            sv = np.array([next((size_map[kf] for kf in _key_forms(i) if kf in size_map), np.nan) for i in ids], float)
            if np.isfinite(sv).sum() > 3 and (sv[np.isfinite(sv)] > 0).all():
                a.size_values = sv
                if np.isfinite(sv).all() and not is_cat:
                    covs["log_size"] = np.log(sv)[:, None]
        if X.shape[1] == 0 or not np.isfinite(X).all():
            notes.append(f"{name}: no feature varies among the specimens (for example empty homology cells) - skipped"); continue
        row, hits, sp_rows, pw = analyse_set(name, X, y, ids, a, out, covs if not is_cat else {})
        rows.append(row); sp_all += sp_rows; pw_all += pw; hits_by_set[name] = (ids, hits)
        print(f"{name}: {row['n_specimens']} specimens, {row['n_species']} species; PERMANOVA R2 {row['permanova_R2']:.3f} "
              f"P {row['permanova_P']:.3g}; LOO {row['loo_correct_PC1to5']}/{row['loo_scored']}", flush=True)
    if not rows:
        sys.exit("no trait set could be analysed: " + "; ".join(notes))
    # McNemar between trait sets, on the specimens scored in both
    mc = []
    for (n1, (i1, h1)), (n2, (i2, h2)) in itertools.combinations(hits_by_set.items(), 2):
        canon = lambda i: _key_forms(i)[2]                 # same specimen whatever suffix/extension the table uses
        d1 = {canon(i): h for i, h in zip(i1, h1) if h[0]}; d2 = {canon(i): h for i, h in zip(i2, h2) if h[0]}
        common = sorted(set(d1) & set(d2))
        if len(common) < 5:
            continue
        pm, a_only, b_only = mcnemar_exact([d1[i][1] for i in common], [d2[i][1] for i in common])
        mc.append(dict(set_a=n1, set_b=n2, n_common=len(common), acc_a=np.mean([d1[i][1] for i in common]),
                       acc_b=np.mean([d2[i][1] for i in common]), only_a_right=a_only, only_b_right=b_only, mcnemar_P=pm))
    write = lambda fn, rr: pd.DataFrame(rr).to_csv(out / fn, index=False) if rr else None
    write("summary.csv", rows); write("per_species.csv", sp_all); write("pairwise_permanova.csv", pw_all)
    write("mcnemar_between_trait_sets.csv", mc); write("categorical_characters.csv", cat_rows)
    # every specimen against the species hypothesis it was given (group labels: morphology, DNA, field notes...):
    # what each trait set calls it under leave-one-out, and whether it should be looked at again
    sf = {}
    for nm, (ids_, hits_) in hits_by_set.items():
        for i_, h_ in zip(ids_, hits_):
            r_ = sf.setdefault(_key_forms(i_)[2], {"specimen_id": i_, "assigned_species": group_of(i_, gmap)})
            if h_[0]:
                r_[f"named_{nm}"] = h_[2]
                r_[f"fit_ratio_{nm}"] = round(h_[3], 3) if len(h_) > 3 and h_[3] == h_[3] else None
    spec_rows = []
    for r_ in sf.values():
        named = [v for k_, v in r_.items() if k_.startswith("named_")]
        back = sum(1 for v in named if v == r_["assigned_species"])
        elsewhere = [v for v in named if v != r_["assigned_species"]]
        r_.update(sets_scored=len(named), named_back=back, named_elsewhere=len(elsewhere),
                  named_elsewhere_as="; ".join(sorted(set(elsewhere), key=_natural)),
                  status=("not scored (single specimen of its species)" if not named else
                          "fits" if back == len(named) else
                          "re-examine" if len(elsewhere) > len(named) / 2 else "mixed"))
        spec_rows.append(r_)
    if spec_rows:
        order = {"re-examine": 0, "mixed": 1, "fits": 2}
        spec_rows.sort(key=lambda r_: (order.get(r_["status"], 3), -r_["named_elsewhere"], _natural(str(r_["specimen_id"]))))
        lead = ["specimen_id", "assigned_species", "status", "sets_scored", "named_back", "named_elsewhere",
                "named_elsewhere_as"]
        sfd = pd.DataFrame(spec_rows)
        sfd = sfd[lead + [c for c in sfd.columns if c not in lead]]
        sfd.to_csv(out / "specimen_flags.csv", index=False)
    plot_overview(rows, out / "identification_by_trait_set.png")
    sep = None
    if pw_all:
        sp = sorted({r["species_a"] for r in pw_all} | {r["species_b"] for r in pw_all}, key=_natural)
        ix = {s: i for i, s in enumerate(sp)}; sep = np.zeros((len(sp), len(sp)))
        names = [r["trait_set"] for r in rows]
        for r in pw_all:
            if r["separated"]:
                sep[ix[r["species_a"]], ix[r["species_b"]]] += 1; sep[ix[r["species_b"]], ix[r["species_a"]]] += 1
        plot_separation(sp, names, sep, out / "species_separation.png")
        sm = []
        for s in sp:
            by = {}
            for r in pw_all:
                if s in (r["species_a"], r["species_b"]):
                    o = r["species_b"] if r["species_a"] == s else r["species_a"]
                    by.setdefault(o, []).append(r["trait_set"] if r["separated"] else None)
            never = sorted([o for o, v in by.items() if not any(v)], key=_natural)
            sm.append(dict(species=s, species_compared=len(by),
                           separated_by_at_least_one_set=sum(1 for v in by.values() if any(v)),
                           not_separated_by_any_set="; ".join(never)))
        write("species_validation.csv", sm)
        # which trait set separates which pairs, and the pairs only one set separates
        per_pair = {}
        for r in pw_all:
            per_pair.setdefault((r["species_a"], r["species_b"]), set())
            if r["separated"]:
                per_pair[(r["species_a"], r["species_b"])].add(r["trait_set"])
        by_set = []
        for nm in names:
            only = [f"{x} / {y2}" for (x, y2), v in per_pair.items() if v == {nm}]
            by_set.append(dict(trait_set=nm, pairs_tested=sum(1 for r in pw_all if r["trait_set"] == nm),
                               pairs_separated=sum(1 for r in pw_all if r["trait_set"] == nm and r["separated"]),
                               pairs_separated_only_by_this_set=len(only), those_pairs="; ".join(only)))
        write("separation_by_trait_set.csv", by_set)
    # report
    L = ["# Morphological statistics by trait set", "",
         f"Groups from `{Path(a.groups).name}`; PERMANOVA on PC1-{a.n_pcs} with {a.permutations} permutations; pairwise tests "
         f"Benjamini-Hochberg corrected at q < {a.alpha}; identification = leave-one-out nearest neighbour.", ""]
    L += ["| Trait set | Specimens | Species | PERMANOVA R2 | P | Beyond covariate | LOO accuracy PC1-5 (95% CI) | all PCs |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bey = "; ".join(f"{k.replace('permanova_R2_beyond_', '').replace('_', ' ')}: R2 {v:.2f}, P = {r[k.replace('R2', 'P')]:.3g}"
                        for k, v in r.items() if k.startswith("permanova_R2_beyond_")) or "-"
        L.append(f"| {r['trait_set']} | {r['n_specimens']} | {r['n_species']} | {r['permanova_R2']:.2f} | {r['permanova_P']:.3g} | {bey} | "
                 f"{r['loo_acc_PC1to5']:.2f} ({r['loo_ci_low_PC1to5']:.2f}-{r['loo_ci_high_PC1to5']:.2f}) | {r['loo_acc_allPCs']:.2f} |")
    if any("allometry_R2" in r for r in rows):
        L += ["", "Allometry (trait set ~ log size): " + "; ".join(
            f"{r['trait_set']} R2 {r['allometry_R2']:.2f}, P = {r['allometry_P']:.3g}" for r in rows if "allometry_R2" in r)]
    if mc:
        L += ["", "## Do trait sets identify species differently? (exact McNemar, same specimens)", "",
              "| Trait set A | Trait set B | Specimens | Accuracy A | Accuracy B | Only A right | Only B right | P |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {m['set_a']} | {m['set_b']} | {m['n_common']} | {m['acc_a']:.2f} | {m['acc_b']:.2f} | {m['only_a_right']} | "
              f"{m['only_b_right']} | {m['mcnemar_P']:.3g} |" for m in mc]
    if pw_all:
        L += ["", "## Species validation", "",
              "`species_validation.csv`: for each species, how many others it is separated from by at least one trait set "
              "(pairwise PERMANOVA, q < alpha) and which it is not separated from by any. `pairwise_permanova.csv` "
              "gives every pair and trait set. With two or three specimens per species the smallest attainable P "
              "(column min_attainable_P) can exceed alpha, so a pair can be unseparable for lack of specimens, not for "
              "lack of difference."]
    if pw_all:
        L += ["", "| Trait set | Pairs tested | Pairs separated | Separated by this set only |", "|---|---|---|---|"]
        L += [f"| {b['trait_set']} | {b['pairs_tested']} | {b['pairs_separated']} | {b['pairs_separated_only_by_this_set']} |"
              for b in by_set]
        L += ["", "The pairs each set alone separates are listed in `separation_by_trait_set.csv`; a species pair "
              "that no set separates is listed for each species in `species_validation.csv`."]
    if cat_rows:
        L += ["", f"## Descriptive categorical characters", "", "`categorical_characters.csv`: Cramer's V and permutation "
              "P (BH-corrected q) for each character against the groups."]
    if notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    L += ["", "Figures: `pca_<trait set>.png` (species hulls; PERMANOVA in the title), `identification_by_trait_set.png`, "
          "`species_separation.png`."]
    (out / "trait_stats_report.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "trait_stats_report.md")


if __name__ == "__main__":
    main()
