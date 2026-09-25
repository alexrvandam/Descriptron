#!/usr/bin/env python3
"""
descriptron_shape_stats.py — metadata-driven shape analyses (in addition to the existing GPA analyses)
======================================================================================================

Landmark shapes (COCO keypoints, or a .tps) are aligned by generalised Procrustes analysis and analysed
against specimen metadata from descriptron_metadata.py (a table plus column roles). Each analysis runs
only when the metadata supports it; nothing here replaces the existing semilandmark/landmark GPA outputs.

  anova        Procrustes ANOVA / regression with residual randomisation (RRPP), several factors and
               covariates, interactions (e.g. species x locality); sequential (type I) sums of squares
  allometry    is the shape ~ size relationship the same in every group? (log centroid size x group)
  trajectory   phenotypic trajectory analysis: do groups change shape in the same direction and by the
               same amount across the levels of a second factor (e.g. species across localities)?
  disparity    morphological disparity (Procrustes variance) per group, pairwise permutation tests
  pls          two-block PLS: shape vs the continuous covariates (environment)
  mantel       shape distance vs geographic distance (needs latitude/longitude)
  asymmetry    directional asymmetry and per-specimen asymmetry for paired (left/right) landmarks
  modularity   covariance ratio (CR) between landmark modules, and integration by two-block PLS
  assign       unlabelled specimens: most likely group with typicality probabilities (can be "none");
               leave-one-out accuracy on the labelled ones; clustering proposes groups
  phylosignal  multivariate phylogenetic signal Kmult on group means (needs a Newick tree)

    python descriptron_shape_stats.py wings.json --metadata meta/specimen_metadata.csv \\
        --schema meta/metadata_schema.json --out-dir stats/
    python descriptron_shape_stats.py wings.json --metadata m.csv --schema s.json --formula "species*locality + elevation" \\
        --analyses anova trajectory disparity --out-dir stats/
    python descriptron_shape_stats.py heads.json --metadata m.csv --schema s.json --pairs 1-5,2-6 --modules 1,2,3|4,5,6 \\
        --tree species.nwk --out-dir stats/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descriptron_metadata as dm  # noqa: E402

ALL = ["anova", "allometry", "trajectory", "disparity", "pls", "mantel", "asymmetry", "modularity", "assign", "phylosignal"]


# ================================================================== shapes and GPA
def load_shapes(path: str, category: Optional[str]) -> Tuple[List[str], np.ndarray, List[str]]:
    """-> image file names, raw coordinates (n, p, 2), landmark names. Specimens with missing landmarks are dropped."""
    if path.lower().endswith(".tps"):
        import descriptron_convert as dcv
        specs = dcv.parse_tps(path)
        p = max(len(s["lm"]) for s in specs)
        names, X = [], []
        for k, s in enumerate(specs):
            if len(s["lm"]) == p and all(x >= 0 and y >= 0 for x, y in s["lm"]):
                names.append(s["image"] or s["id"] or f"specimen_{k + 1}"); X.append(s["lm"])
        return names, np.array(X, float), [str(i + 1) for i in range(p)]
    coco = json.load(open(path))
    cats = {c["id"]: c for c in coco["categories"]}
    ims = {i["id"]: i for i in coco["images"]}
    rows = []
    for a in coco["annotations"]:
        if "keypoints" not in a or a.get("semilandmarks"):
            continue
        c = cats.get(a["category_id"], {})
        if category and c.get("name") != category:
            continue
        kp = np.array(a["keypoints"], float).reshape(-1, 3)
        order = a.get("point_order") or list(range(1, len(kp) + 1))
        rows.append((ims[a["image_id"]]["file_name"], dict(zip(map(int, order), kp)), c))
    if not rows:
        sys.exit("no landmark sets found" + (f" in category {category}" if category else ""))
    p = max(max(d) for _, d, _ in rows)
    lm_names = next((c.get("keypoints") for _, _, c in rows if c.get("keypoints")), None) or [str(i) for i in range(1, p + 1)]
    names, X, dropped = [], [], 0
    for fn, d, _ in rows:
        if all(k in d and d[k][2] > 0 for k in range(1, p + 1)):
            names.append(fn); X.append([d[k][:2] for k in range(1, p + 1)])
        else:
            dropped += 1
    if dropped:
        print(f"note: {dropped} specimens with missing landmarks left out")
    return names, np.array(X, float), list(lm_names)[:p]


def gpa(X: np.ndarray, iters: int = 50, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Generalised Procrustes (no reflection): -> aligned shapes (unit centroid size), centroid sizes."""
    Xc = X - X.mean(1, keepdims=True)
    cs = np.sqrt((Xc ** 2).sum(axis=(1, 2)))
    Z = Xc / cs[:, None, None]
    M = Z[0].copy()
    for _ in range(iters):
        for i in range(len(Z)):
            u, s, vt = np.linalg.svd(Z[i].T @ M)
            d = np.sign(np.linalg.det(u @ vt))
            D = np.diag([1] * (u.shape[0] - 1) + [d])
            Z[i] = Z[i] @ (u @ D @ vt)
        M_new = Z.mean(0); M_new /= np.linalg.norm(M_new)
        if np.linalg.norm(M_new - M) < tol:
            M = M_new; break
        M = M_new
    # rotate the whole sample to the mean's principal axes, as geomorph's gpagen does. Distances, SS, R2, F,
    # disparity, PLS and Kmult do not depend on it; the covariance ratio (CR) does.
    _, _, vt = np.linalg.svd(M - M.mean(0))
    Z = Z @ vt.T
    return Z, cs


def mirrored_specimens(X: np.ndarray, iters: int = 20) -> np.ndarray:
    """Specimens that fit the consensus better when reflected (photographed from the other side, or the
    opposite body side): a GPA that allows reflection, then the sign of each specimen's best rotation."""
    Xc = X - X.mean(1, keepdims=True)
    Z = Xc / np.sqrt((Xc ** 2).sum(axis=(1, 2)))[:, None, None]
    M = Z[0].copy()
    flags = np.zeros(len(Z), bool)
    for _ in range(iters):
        W = np.empty_like(Z)
        for i in range(len(Z)):
            u, s, vt = np.linalg.svd(Z[i].T @ M)
            R = u @ vt
            flags[i] = np.linalg.det(R) < 0
            W[i] = Z[i] @ R
        M = W.mean(0); M /= np.linalg.norm(M)
    # the majority orientation is the reference; the minority are the mirror images
    return flags if flags.sum() <= len(flags) / 2 else ~flags


def flat(Z: np.ndarray) -> np.ndarray:
    return Z.reshape(len(Z), -1)


# ================================================================== model matrices
class Design:
    """Builds model matrices from a formula like 'species * locality + elevation' (R style: * expands)."""

    def __init__(self, formula: str, data: Dict[str, List], kinds: Dict[str, str]):
        self.data, self.kinds = data, kinds
        terms = []
        for part in [t.strip() for t in formula.split("+") if t.strip()]:
            vars_ = [v.strip() for v in re.split(r"[*:]", part)]
            if "*" in part:
                for r in range(1, len(vars_) + 1):
                    for combo in _combinations(vars_, r):
                        terms.append(tuple(combo))
            else:
                terms.append(tuple(vars_))
        seen, self.terms = set(), []
        for t in sorted(terms, key=len):
            if t not in seen:
                seen.add(t); self.terms.append(t)
        for t in self.terms:
            for v in t:
                if v not in data:
                    raise SystemExit(f"formula term {v!r} is not a metadata column ({list(data)})")

    def columns(self, var: str) -> np.ndarray:
        vals = self.data[var]
        if self.kinds[var] == "continuous":
            return np.array(vals, float)[:, None]
        levels = sorted(set(vals), key=str)
        return np.array([[1.0 if v == lv else 0.0 for lv in levels[1:]] for v in vals])   # treatment contrasts

    def term_matrix(self, term) -> np.ndarray:
        M = np.ones((len(next(iter(self.data.values()))), 1))
        for v in term:
            C = self.columns(v)
            M = np.einsum("ni,nj->nij", M, C).reshape(len(M), -1)
        return M

    def model(self, k: int) -> np.ndarray:
        n = len(next(iter(self.data.values())))
        return np.hstack([np.ones((n, 1))] + [self.term_matrix(t) for t in self.terms[:k]])


def _combinations(v, r):
    if r == 0:
        yield []
        return
    for i in range(len(v)):
        for rest in _combinations(v[i + 1:], r - 1):
            yield [v[i]] + rest


def _fit(X, Y):
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    F = X @ B
    return F, Y - F


def _rank(X):
    return int(np.linalg.matrix_rank(X))


def _box_cox_rrpp(y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """RRPP's box.cox.fast: lambda estimated on the random values only (y[1:]), bounded to [-5, 5]."""
    from scipy.optimize import minimize_scalar
    y = np.asarray(y, float)
    if (y <= 0).any():
        y = y - y.min() + 1e-4
    yr = y[1:]
    yy = yr / math.exp(np.log(yr).mean())
    logy = np.log(yy)
    n = len(yr)

    def negll(la):
        if abs(la) > eps:
            yt = (yy ** la - 1) / la
        else:
            yt = logy * (1 + (la * logy) / 2 * (1 + (la * logy) / 3 * (1 + (la * logy) / 4)))
        return n / 2 * math.log(max(((yt - yt.mean()) ** 2).sum(), 1e-300))
    lam = minimize_scalar(negll, bounds=(-5, 5), method="bounded").x
    if abs(lam) < eps:
        return np.log(y)
    return (y ** lam - 1) / lam


def effect_size(obs: float, rand: np.ndarray) -> float:
    """Z of the observed statistic against its permutation distribution, as RRPP's effect.size: Box-Cox
    (lambda from the random values), centred, divided by the population standard deviation."""
    x = np.concatenate([[obs], np.asarray(rand, float)])
    x = x[np.isfinite(x)]
    if len(np.unique(x)) == 1:
        return 0.0
    z = _box_cox_rrpp(x)
    z = z - z.mean()
    sd = math.sqrt((z ** 2).sum() / len(z))
    return float(z[0] / sd) if sd > 0 else float("nan")


def procrustes_anova(Y: np.ndarray, design: Design, iters: int, rng) -> List[Dict]:
    """Sequential (type I) SS, RRPP: residuals of each term's reduced model are permuted (Collyer & Adams)."""
    n = len(Y)
    Xfull = design.model(len(design.terms))
    _, Rfull = _fit(Xfull, Y)
    RSS_full, df_res = float((Rfull ** 2).sum()), n - _rank(Xfull)
    TSS = float(((Y - Y.mean(0)) ** 2).sum())
    perms = [rng.permutation(n) for _ in range(iters)]
    rows = []
    for k, term in enumerate(design.terms, 1):
        Xr, Xf = design.model(k - 1), design.model(k)
        df = _rank(Xf) - _rank(Xr)
        Fr, Rr = _fit(Xr, Y)
        SS = float((_fit(Xr, Y)[1] ** 2).sum() - (_fit(Xf, Y)[1] ** 2).sum())
        F = (SS / df) / (RSS_full / df_res) if df > 0 and df_res > 0 else float("nan")
        Fs = []
        for pm in perms:
            Ys = Fr + Rr[pm]
            ss = float((_fit(Xr, Ys)[1] ** 2).sum() - (_fit(Xf, Ys)[1] ** 2).sum())
            rss = float((_fit(Xfull, Ys)[1] ** 2).sum())
            Fs.append((ss / df) / (rss / df_res) if df > 0 else float("nan"))
        Fs = np.array(Fs)
        rows.append({"term": ":".join(term), "Df": df, "SS": SS, "MS": SS / df if df else float("nan"),
                     "Rsq": SS / TSS, "F": F, "Z": effect_size(F, Fs), "P": (1 + (Fs >= F - 1e-12).sum()) / (iters + 1)})
    rows.append({"term": "Residuals", "Df": df_res, "SS": RSS_full, "MS": RSS_full / df_res if df_res else float("nan"),
                 "Rsq": RSS_full / TSS, "F": "", "Z": "", "P": ""})
    rows.append({"term": "Total", "Df": n - 1, "SS": TSS, "MS": "", "Rsq": "", "F": "", "Z": "", "P": ""})
    return rows


# ================================================================== trajectory analysis
def _trajectory_stats(means: Dict[str, np.ndarray], levels: List[str]):
    out = {}
    for g, M in means.items():
        path = float(sum(np.linalg.norm(M[i + 1] - M[i]) for i in range(len(levels) - 1)))
        C = M - M.mean(0)
        v = np.linalg.svd(C, full_matrices=False)[2][0]
        v = v if (M[-1] - M[0]) @ v >= 0 else -v
        out[g] = (path, v, M)
    return out


def trajectory(Y, groups, levels_of, iters, rng):
    """Collyer & Adams (2013): group trajectories across levels; magnitude (path length), direction (angle of
    first principal axis) differences; RRPP with residuals of the model without the interaction."""
    G = sorted(set(groups), key=str); L = sorted(set(levels_of), key=str)
    if len(L) < 2 or len(G) < 2:
        return None
    cells = {(g, l): [i for i in range(len(Y)) if groups[i] == g and levels_of[i] == l] for g in G for l in L}
    if any(len(v) == 0 for v in cells.values()):
        G = [g for g in G if all(cells[(g, l)] for l in L)]
        if len(G) < 2:
            return None
    idx = [i for i in range(len(Y)) if groups[i] in G]
    Yk, gk, lk = Y[idx], [groups[i] for i in idx], [levels_of[i] for i in idx]
    data = {"g": gk, "l": lk}
    d = Design("g + l", data, {"g": "factor", "l": "factor"})
    Fr, Rr = _fit(d.model(2), Yk)

    def stats(Yv):
        means = {g: np.array([Yv[[i for i in range(len(Yv)) if gk[i] == g and lk[i] == l]].mean(0) for l in L]) for g in G}
        return _trajectory_stats(means, L)
    obs = stats(Yk)
    pairs = [(a, b) for i, a in enumerate(G) for b in G[i + 1:]]
    def comp(st):
        return {(a, b): (abs(st[a][0] - st[b][0]), math.degrees(math.acos(max(-1, min(1, float(st[a][1] @ st[b][1])))))) for a, b in pairs}
    o = comp(obs)
    rnd = defaultdict(list)
    for _ in range(iters):
        c = comp(stats(Fr + Rr[rng.permutation(len(Yk))]))
        for k, v in c.items():
            rnd[k].append(v)
    rows = []
    for a, b in pairs:
        r = np.array(rnd[(a, b)])
        rows.append({"group_1": a, "group_2": b, "path_length_1": obs[a][0], "path_length_2": obs[b][0],
                     "magnitude_difference": o[(a, b)][0], "P_magnitude": (1 + (r[:, 0] >= o[(a, b)][0] - 1e-12).sum()) / (iters + 1),
                     "angle_deg": o[(a, b)][1], "P_direction": (1 + (r[:, 1] >= o[(a, b)][1] - 1e-9).sum()) / (iters + 1)})
    return {"levels": L, "groups": G, "pairs": rows, "means": {g: obs[g][2] for g in G}}


# ================================================================== disparity
def disparity(Y, groups, iters, rng):
    """Procrustes variance per group (mean squared distance to the group mean, as geomorph); pairwise absolute
    differences tested by permuting residuals of the group-means model among groups."""
    G = sorted(set(groups), key=str)
    gi = {g: [i for i in range(len(Y)) if groups[i] == g] for g in G}
    def pv(R):
        return {g: float((R[ix] ** 2).sum() / len(ix)) for g, ix in gi.items()}
    means = {g: Y[ix].mean(0) for g, ix in gi.items()}
    R = np.vstack([Y[i] - means[groups[i]] for i in range(len(Y))])
    obs = pv(R)
    pairs = [(a, b) for i, a in enumerate(G) for b in G[i + 1:]]
    rnd = defaultdict(list)
    for _ in range(iters):
        v = pv(R[rng.permutation(len(Y))])
        for a, b in pairs:
            rnd[(a, b)].append(abs(v[a] - v[b]))
    rows = [{"group": g, "n": len(gi[g]), "procrustes_variance": obs[g]} for g in G]
    prs = [{"group_1": a, "group_2": b, "difference": abs(obs[a] - obs[b]),
            "P": (1 + (np.array(rnd[(a, b)]) >= abs(obs[a] - obs[b]) - 1e-15).sum()) / (iters + 1)} for a, b in pairs]
    return rows, prs


# ================================================================== PLS, Mantel
def two_block_pls(A, B, iters, rng):
    A = A - A.mean(0); B = B - B.mean(0)
    def rpls(Bm):
        u, s, vt = np.linalg.svd(A.T @ Bm, full_matrices=False)
        return float(np.corrcoef(A @ u[:, 0], Bm @ vt[0])[0, 1]), s
    r, s = rpls(B)
    rand = np.array([rpls(B[rng.permutation(len(B))])[0] for _ in range(iters)])
    return {"r_PLS": abs(r), "P": (1 + (np.abs(rand) >= abs(r) - 1e-12).sum()) / (iters + 1),
            "Z": effect_size(abs(r), np.abs(rand)), "first_axis_covariance_share": float(s[0] ** 2 / (s ** 2).sum())}


def haversine_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla, dlo = la[:, None] - la[None], lo[:, None] - lo[None]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin(dlo / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mantel(D1, D2, iters, rng):
    iu = np.triu_indices(len(D1), 1)
    r = float(np.corrcoef(D1[iu], D2[iu])[0, 1])
    rand = []
    for _ in range(iters):
        p = rng.permutation(len(D1))
        rand.append(np.corrcoef(D1[iu], D2[p][:, p][iu])[0, 1])
    rand = np.array(rand)
    return {"r": r, "P": (1 + (rand >= r - 1e-12).sum()) / (iters + 1)}


# ================================================================== asymmetry
def asymmetry(X, pairs, iters, rng):
    """Object symmetry (Klingenberg et al. 2002): each configuration and its relabelled mirror image are aligned
    together; the symmetric component is their mean, the asymmetric one half their difference. DA is the mean
    asymmetry (tested against zero by sign-flipping individuals); each specimen's asymmetry is reported."""
    p = X.shape[1]
    perm = list(range(p))
    for a, b in pairs:
        perm[a - 1], perm[b - 1] = b - 1, a - 1
    Xm = X.copy(); Xm[..., 0] *= -1; Xm = Xm[:, perm]
    Z, _ = gpa(np.concatenate([X, Xm]))
    n = len(X)
    A = flat(Z[:n]) - flat(Z[n:])                       # asymmetry vectors (twice the asymmetric component)
    ind = np.linalg.norm(A, axis=1) / 2
    da = float(np.linalg.norm(A.mean(0)) / 2)
    rand = [np.linalg.norm((A * rng.choice([-1, 1], n)[:, None]).mean(0)) / 2 for _ in range(iters)]
    return {"directional_asymmetry": da, "P_DA": (1 + (np.array(rand) >= da - 1e-15).sum()) / (iters + 1),
            "mean_individual_asymmetry": float(ind.mean()), "per_specimen": ind.tolist()}


# ================================================================== modularity (CR), integration
def cr_coefficient(Y2d: np.ndarray, modules: List[int]) -> float:
    """Covariance ratio (Adams 2016) for two or more modules; Y2d is n x (p*2), modules gives each landmark's module."""
    S = np.cov(Y2d, rowvar=False)
    lab = np.repeat(np.array(modules), 2)
    groups = sorted(set(modules))
    crs = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            ia, ib = lab == a, lab == b
            S11, S22, S12 = S[np.ix_(ia, ia)].copy(), S[np.ix_(ib, ib)].copy(), S[np.ix_(ia, ib)]
            np.fill_diagonal(S11, 0); np.fill_diagonal(S22, 0)
            crs.append(math.sqrt(np.trace(S12 @ S12.T) / math.sqrt(np.trace(S11 @ S11) * np.trace(S22 @ S22))))
    return float(np.mean(crs))


def modularity(Y2d, modules, iters, rng):
    obs = cr_coefficient(Y2d, modules)
    rand = np.array([cr_coefficient(Y2d, list(rng.permutation(modules))) for _ in range(iters)])
    lab = np.repeat(np.array(modules), 2)
    g = sorted(set(modules))
    pls = two_block_pls(Y2d[:, lab == g[0]], Y2d[:, lab == g[1]], iters, rng)
    return {"CR": obs, "P_CR (share of random partitions with CR <= observed)": (1 + (rand <= obs + 1e-15).sum()) / (iters + 1),
            "random_CR_mean": float(rand.mean()), "integration_r_PLS_modules_1_2": pls["r_PLS"], "P_integration": pls["P"]}


# ================================================================== assignment, clustering
def assign(P: np.ndarray, groups: List[Optional[str]], k: Optional[int] = None):
    """LDA-type assignment on PC scores (pooled covariance), with typicality probabilities (chi-square on the
    Mahalanobis distance): a specimen unlike every group is reported as 'none'."""
    from scipy.stats import chi2
    lab = [i for i, g in enumerate(groups) if g not in (None, "")]
    G = sorted(set(groups[i] for i in lab), key=str)
    if len(G) < 2:
        return None
    nmin = min(sum(1 for i in lab if groups[i] == g) for g in G)
    k = k or max(1, min(P.shape[1], len(lab) - len(G), 10))
    S = P[:, :k]

    def fit(ix):
        mu = {g: S[[i for i in ix if groups[i] == g]].mean(0) for g in G}
        R = np.vstack([S[i] - mu[groups[i]] for i in ix])
        C = np.cov(R, rowvar=False).reshape(k, k) + 1e-9 * np.eye(k)
        return mu, np.linalg.inv(C)

    def predict(x, mu, Ci):
        d2 = {g: float((x - m) @ Ci @ (x - m)) for g, m in mu.items()}
        w = np.array([math.exp(-0.5 * (d2[g] - min(d2.values()))) for g in G]); post = w / w.sum()
        best = G[int(np.argmax(post))]
        typ = {g: float(1 - chi2.cdf(d2[g], k)) for g in G}
        return best, post, typ
    correct = 0
    for i in lab:
        mu, Ci = fit([j for j in lab if j != i])
        correct += predict(S[i], mu, Ci)[0] == groups[i]
    mu, Ci = fit(lab)
    rows = []
    for i, g in enumerate(groups):
        if g not in (None, ""):
            continue
        best, post, typ = predict(S[i], mu, Ci)
        rows.append({"index": i, "most_likely": best if typ[best] >= 0.05 else "none (unlike all groups)",
                     "posterior": float(post.max()), "typicality": typ[best], **{f"typicality_{h}": typ[h] for h in G}})
    return {"k_PCs": k, "loo_accuracy": correct / len(lab), "n_labelled": len(lab), "unlabelled": rows}


def cluster(P: np.ndarray, max_k: int = 8, seed: int = 0):
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None
    k = min(P.shape[1], 5)
    best = None
    for c in range(1, min(max_k, len(P) // 3) + 1):
        gm = GaussianMixture(c, covariance_type="diag", random_state=seed, n_init=3).fit(P[:, :k])
        b = gm.bic(P[:, :k])
        if best is None or b < best[0]:
            best = (b, c, gm.predict(P[:, :k]))
    return {"n_clusters_BIC": best[1], "labels": best[2].tolist()} if best else None


# ================================================================== phylogenetic signal
def parse_newick(s: str):
    """Minimal Newick parser -> (tips, C) with C the phylogenetic covariance (shared path length from the root)."""
    s = s.strip().rstrip(";")
    pos = 0
    def node():
        nonlocal pos
        kids = []
        if s[pos] == "(":
            pos += 1
            kids.append(node())
            while s[pos] == ",":
                pos += 1; kids.append(node())
            pos += 1                                     # ')'
        m = re.match(r"([^:,();]*)(?::([0-9.eE+\-]+))?", s[pos:])
        pos += m.end()
        return {"name": m.group(1).strip().strip("'\""), "len": float(m.group(2) or 0), "kids": kids}
    root = node()
    tips, paths = [], {}
    def walk(n, anc, depth):
        here = anc + [(id(n), n["len"])]
        if not n["kids"]:
            tips.append(n["name"]); paths[n["name"]] = here
        for c in n["kids"]:
            walk(c, here, depth + n["len"])
    walk(root, [], 0)
    C = np.zeros((len(tips), len(tips)))
    for i, a in enumerate(tips):
        for j, b in enumerate(tips):
            shared = 0.0
            for (ia, la), (ib, lb) in zip(paths[a][1:], paths[b][1:]):     # skip the root's own branch
                if ia != ib:
                    break
                shared += la
            C[i, j] = shared
    return tips, C


def kmult(Y: np.ndarray, C: np.ndarray) -> float:
    """Adams (2014) multivariate K."""
    n = len(Y)
    Ci = np.linalg.inv(C)
    one = np.ones((n, 1))
    a = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)            # phylogenetic mean
    R = Y - one @ a
    obs = np.trace(R.T @ R) / np.trace(R.T @ Ci @ R)
    expct = (np.trace(C) - n / Ci.sum()) / (n - 1)
    return float(obs / expct)


def phylosignal(Y, C, iters, rng):
    K = kmult(Y, C)
    rand = np.array([kmult(Y[rng.permutation(len(Y))], C) for _ in range(iters)])
    return {"Kmult": K, "P": (1 + (rand >= K - 1e-12).sum()) / (iters + 1), "Z": effect_size(K, rand)}


# ================================================================== reporting
def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    cols = list(OrderedDict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def pca(Yf):
    C = Yf - Yf.mean(0)
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    return C @ vt.T, s ** 2 / (s ** 2).sum()


def plot_pca(P, ev, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for g in sorted(set(labels), key=str):
        ix = [i for i, l in enumerate(labels) if l == g]
        ax.scatter(P[ix, 0], P[ix, 1], s=22, label=str(g))
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)"); ax.set_title(title, fontsize=10)
    if len(set(labels)) <= 12:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("Landmark shapes")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shapes", help="COCO keypoints JSON or .tps")
    ap.add_argument("--metadata", required=True); ap.add_argument("--schema", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--category", help="landmark category in the COCO")
    ap.add_argument("--formula", help="terms for anova, e.g. 'species*locality + elevation' "
                                      "(default: the group, then factors, then continuous covariates, main effects)")
    ap.add_argument("--analyses", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--iterations", type=int, default=999)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trajectory-levels", help="factor whose levels trajectories run across (default: first factor)")
    ap.add_argument("--pairs", help="paired landmarks for asymmetry, e.g. 1-5,2-6")
    ap.add_argument("--modules", help="landmark modules, e.g. 1,2,3|4,5,6 (numbers as in point_order)")
    ap.add_argument("--tree", help="Newick tree whose tips are group names (for phylosignal)")
    ap.add_argument("--no-reflect-mirrored", dest="reflect", action="store_false",
                    help="do not reflect mirror-image specimens before GPA (default: detect and reflect them)")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    files, X, lm_names = load_shapes(a.shapes, a.category)
    cols, rows = dm.load_table(a.metadata)
    schema = json.load(open(a.schema))
    links, unmatched = dm.link_images(rows, schema, files)
    keep = [i for i, f in enumerate(files) if f in links]
    if unmatched:
        print(f"note: {len(unmatched)} specimens without metadata left out (e.g. {unmatched[0]})")
    if len(keep) < 4:
        sys.exit("fewer than 4 specimens with both landmarks and metadata")
    files, X = [files[i] for i in keep], X[keep]
    meta = [rows[links[f]] for f in files]
    refl = mirrored_specimens(X) if a.reflect else np.zeros(len(X), bool)
    if refl.any():
        X = X.copy(); X[refl, :, 0] *= -1
        print(f"note: {int(refl.sum())} mirror-image specimens reflected before GPA (listed in specimens.csv; "
              f"--no-reflect-mirrored to keep them as they are)")
    Z, cs = gpa(X)
    Y = flat(Z)
    P, ev = pca(Y)
    role = lambda r: [c for c, v in schema.items() if v == r]
    group_col = (role("group") or [None])[0]
    factors = [c for c in role("factor") if len(set(m.get(c, "") for m in meta)) > 1]
    conts = [c for c in role("continuous") if all(dm._isnum(m.get(c, "")) for m in meta)]
    kinds = {c: "factor" for c in ([group_col] if group_col else []) + factors}
    kinds.update({c: "continuous" for c in conts})
    kinds["logCS"] = "continuous"
    data = {c: [m.get(c, "") for m in meta] for c in kinds if c != "logCS"}
    for c in conts:
        data[c] = [float(str(v).replace(",", ".")) for v in data[c]]
    data["logCS"] = list(np.log(cs))
    report = [f"# Shape statistics\n", f"{len(files)} specimens, {X.shape[1]} landmarks, GPA on {a.shapes}",
              f"metadata: group = {group_col}, factors = {factors}, covariates = {conts}", f"{a.iterations} permutations, seed {a.seed}\n"]
    report.append(f"{int(refl.sum())} mirror-image specimens were reflected before alignment (column 'reflected' in specimens.csv).\n")
    write_rows(out / "specimens.csv", [{"image": f, "reflected": bool(refl[i]), "centroid_size": float(c), **{f"PC{j + 1}": float(P[i, j]) for j in range(min(5, P.shape[1]))},
                                        **{k: m.get(k, "") for k in kinds if k != "logCS"}} for i, (f, c, m) in enumerate(zip(files, cs, meta))])
    if group_col:
        plot_pca(P, ev, data[group_col], out / "pca_by_group.png", f"PCA of Procrustes shape, by {group_col}")

    def safe_name(c):
        return re.sub(r"\W+", "_", c)

    # specimens with a value for every variable of a model: a blank group (an unidentified specimen) or a blank
    # factor must not become a level of its own. The GPA above still uses every specimen; unidentified ones are
    # what the assignment is for.
    def complete(cols):
        return [i for i in range(len(meta))
                if all(str(data[c][i]).strip().lower() not in ("", "na", "nan", "none") for c in cols if c in data)]

    def sub(idx):
        return {c: [v[i] for i in idx] for c, v in data.items()}

    def left_out(idx):
        n = len(meta) - len(idx)
        return [f"{n} specimen(s) without a value for a model variable left out of this analysis."] if n else []
    if "anova" in a.analyses:
        default = " + ".join([c for c in [group_col] + factors + conts if c])
        f = a.formula or default
        if f:
            used = [c for c in kinds if c != "logCS" and re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", f)]
            idx = complete(used)
            design = Design(f, sub(idx), kinds)
            tab = procrustes_anova(Y[idx], design, a.iterations, rng)
            write_rows(out / "procrustes_anova.csv", tab)
            report += ["## Procrustes ANOVA (RRPP, type I SS)", f"shape ~ {f}", *left_out(idx), _md(tab), ""]
    if "allometry" in a.analyses and group_col:
        idx = complete([group_col])
        design = Design(f"logCS*{group_col}", sub(idx), kinds)
        tab = procrustes_anova(Y[idx], design, a.iterations, rng)
        write_rows(out / "allometry_homogeneity_of_slopes.csv", tab)
        inter = next(r for r in tab if r["term"] == f"logCS:{group_col}")
        report += ["## Allometry: homogeneity of slopes", f"shape ~ logCS * {group_col}; the interaction tests whether groups "
                   f"differ in allometric slope (P = {inter['P']:.3g}).", *left_out(idx), _md(tab), ""]
    if "trajectory" in a.analyses and group_col and (a.trajectory_levels or factors):
        lev = a.trajectory_levels or factors[0]
        idx = [i for i in complete([group_col]) if str(meta[i].get(lev, "")).strip()]
        tr = trajectory(Y[idx], [data[group_col][i] for i in idx], [meta[i].get(lev, "") for i in idx], a.iterations, rng)
        if tr:
            write_rows(out / "trajectory_pairwise.csv", tr["pairs"])
            report += ["## Phenotypic trajectories", f"{group_col} across levels of {lev}: {tr['levels']}", _md(tr["pairs"]), ""]
        else:
            report += ["## Phenotypic trajectories", f"skipped: needs >= 2 groups each observed at every level of {lev}", ""]
    if "disparity" in a.analyses and group_col:
        idx = complete([group_col])
        pv, prs = disparity(Y[idx], [data[group_col][i] for i in idx], a.iterations, rng)
        write_rows(out / "disparity_by_group.csv", pv); write_rows(out / "disparity_pairwise.csv", prs)
        report += ["## Morphological disparity (Procrustes variance)", *left_out(idx), _md(pv), _md(prs), ""]
    if "pls" in a.analyses and conts:
        E = np.column_stack([np.array(data[c], float) for c in conts]); E = (E - E.mean(0)) / np.where(E.std(0) > 0, E.std(0), 1)
        r = two_block_pls(Y, E, a.iterations, rng)
        write_rows(out / "pls_shape_environment.csv", [{"covariates": " + ".join(conts), **r}])
        report += ["## Two-block PLS: shape vs covariates", _md([{"covariates": " + ".join(conts), **r}]), ""]
    lat, lon = (role("latitude") or [None])[0], (role("longitude") or [None])[0]
    if "mantel" in a.analyses and lat and lon and all(dm._isnum(m.get(lat, "")) and dm._isnum(m.get(lon, "")) for m in meta):
        Dg = haversine_km(np.array([float(m[lat]) for m in meta]), np.array([float(m[lon]) for m in meta]))
        Ds = np.linalg.norm(Y[:, None] - Y[None], axis=2)
        r = mantel(Ds, Dg, a.iterations, rng)
        write_rows(out / "mantel_shape_geography.csv", [r])
        report += ["## Mantel: Procrustes distance vs geographic distance (km)", _md([r]), ""]
    if "asymmetry" in a.analyses and a.pairs:
        pairs = [tuple(int(x) for x in p.split("-")) for p in a.pairs.split(",")]
        r = asymmetry(X, pairs, a.iterations, rng)
        write_rows(out / "asymmetry_per_specimen.csv", [{"image": f, "asymmetry": v} for f, v in zip(files, r["per_specimen"])])
        rs = {k: v for k, v in r.items() if k != "per_specimen"}
        write_rows(out / "asymmetry_summary.csv", [rs])
        report += ["## Asymmetry (object symmetry)", f"paired landmarks {pairs}", _md([rs]), ""]
    if "modularity" in a.analyses and a.modules:
        mods = [0] * X.shape[1]
        for gi, part in enumerate(a.modules.split("|"), 1):
            for x in part.split(","):
                mods[int(x) - 1] = gi
        if 0 in mods:
            sys.exit("--modules must assign every landmark to a module")
        r = modularity(Y, mods, a.iterations, rng)
        write_rows(out / "modularity.csv", [r])
        report += ["## Modularity (CR) and integration (two-block PLS)", _md([r]), ""]
    if "assign" in a.analyses and group_col and any(m.get(group_col, "") == "" for m in meta):
        r = assign(P, [m.get(group_col) or None for m in meta])
        if r:
            for u in r["unlabelled"]:
                u["image"] = files[u.pop("index")]
            write_rows(out / "assignment_unlabelled.csv", r["unlabelled"])
            report += ["## Assignment of unlabelled specimens", f"{r['n_labelled']} labelled specimens, {r['k_PCs']} PCs; "
                       f"leave-one-out accuracy {100 * r['loo_accuracy']:.1f}%", _md(r["unlabelled"]), ""]
    if "assign" in a.analyses:
        c = cluster(P)
        if c:
            write_rows(out / "clusters.csv", [{"image": f, "cluster": l} for f, l in zip(files, c["labels"])])
            report += ["## Clustering (Gaussian mixture, BIC)", f"{c['n_clusters_BIC']} clusters proposed (clusters.csv)", ""]
    if "phylosignal" in a.analyses and a.tree and group_col:
        tips, C = parse_newick(Path(a.tree).read_text())
        G = [t for t in tips if t in set(data[group_col])]
        if len(G) >= 4:
            Ym = np.array([Y[[i for i, g in enumerate(data[group_col]) if g == t]].mean(0) for t in G])
            ix = [tips.index(t) for t in G]
            r = phylosignal(Ym, C[np.ix_(ix, ix)], a.iterations, rng)
            write_rows(out / "phylogenetic_signal.csv", [{"n_tips": len(G), **r}])
            report += ["## Phylogenetic signal (Kmult)", _md([{"n_tips": len(G), **r}]), ""]
        else:
            report += ["## Phylogenetic signal (Kmult)", f"skipped: only {len(G)} tree tips match {group_col} values", ""]
    (out / "shape_stats_report.md").write_text("\n".join(report))
    print(f"{len(files)} specimens analysed; results and shape_stats_report.md in {out}")


def _md(rows):
    rows = list(rows)
    if not rows:
        return ""
    cols = list(OrderedDict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows])


if __name__ == "__main__":
    main()
