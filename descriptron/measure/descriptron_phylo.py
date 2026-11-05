#!/usr/bin/env python3
"""
descriptron_phylo.py - phylogenetic comparative analyses of any continuous Descriptron traits
=============================================================================================

Given a Newick tree and one or more trait tables (landmark shape, linear measurements, colour pattern, texture, or
any other continuous features), computes on species means:

  signal     multivariate phylogenetic signal Kmult (Adams 2014), permutation P and effect size Z,
             as geomorph::physignal
  pgls       phylogenetic regression of a trait set on a predictor (for example log centroid size: phylogenetic
             allometry) by generalised least squares under Brownian motion, SS / R2 / F with residual
             randomisation, as geomorph::procD.pgls
  morphospace principal components of the species means with ancestral states estimated under Brownian motion
             (generalised least squares) projected into the same space and the tree drawn through them, as
             geomorph::gm.prcomp(phy = )
  models     for every variable on its own: Brownian motion (BM), Ornstein-Uhlenbeck with one optimum (OU), early
             burst (EB), Pagel's lambda and white noise (no phylogeny) fitted by maximum likelihood and compared by
             AICc and Akaike weights, as geiger::fitContinuous (same covariance forms and default bounds); lambda
             is also tested against 0 (no signal) and 1 (BM) by likelihood ratio
  rates      for every variable on its own: a separate evolutionary rate on every branch (variable-rate Brownian
             motion by penalised likelihood, Revell 2021), as phytools::multirateBM; the branches with the fastest
             change are listed and the tree is drawn with its branches coloured by rate. Each branch's change
             between the Brownian-motion ancestral states is also given, divided by its expected standard deviation
             (sqrt(sigma^2 x branch length)), so single large jumps stand out from a generally fast clade

Models and rates are univariate: fit them to the original variables (for colour, L*, a*, b* or chroma of each part,
never a raw hue angle, which is circular), not to principal components, which favour OU and EB spuriously
(Uyeda et al. 2015, Syst. Biol. 64: 677). OU and EB parameters are in the tree's units: on a tree in substitutions
per site (not time) alpha and a are per substitution, and geiger's default bounds (alpha <= e, a >= ln(1e-5)/depth)
depend on those units; estimates at a bound are flagged and the bounds can be changed with --model_bounds.

Trait tables: CSV with one row per specimen (or per species) and numeric columns. The species of each row comes
from a column (--species_col) or from a regular expression applied to an id column (--species_regex). Rows of the
same species are averaged. Species absent from the tree, or tree tips without data, are dropped (the tree is pruned).

  python descriptron_phylo.py --tree tree.nwk --out_dir phylo/ \
      --traits shape=landmark_gpa/forewing/forewing_procrustes_coords.csv \
      --traits colour=color_homology/whole_wing/color_homology_features_whole_wing.csv \
      --species_regex "morph_([A-Za-z]+\\d*?)_?\\d+_" --scale colour=standardize \
      --pgls shape~logCS --analyses signal pgls morphospace

Scaling (--scale NAME=none|standardize|log): Kmult and the PGLS statistics weight every column by its variance, so a
set that mixes units (millimetres with ratios, colour channels with entropies) should be standardised; Procrustes
shape coordinates are analysed as they are (the default for a set whose columns look like x/y coordinates).

Tree tips holding more than the species (voucher codes such as "QH001-S1_Abemus chloropterus"): --tip_regex with one
group extracts the species; a species appearing on several tips keeps the first. Spaces and underscores count as the
same character when tips are matched to species.

Validated against geomorph 4.1.1 and ape on geomorph's plethspecies data (validation/validate_phylo_real_tree.py);
models and rates against geiger 2.0.11 and phytools 2.4.4 (validation/validate_phylo_models_vs_r.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================ tree
class Node:
    __slots__ = ("name", "length", "kids", "parent", "depth", "idx")

    def __init__(self, name="", length=0.0):
        self.name, self.length, self.kids, self.parent, self.depth, self.idx = name, length, [], None, 0.0, -1


def parse_newick(s: str) -> Node:
    s = s.strip().rstrip(";")
    pos = 0

    def node() -> Node:
        nonlocal pos
        n = Node()
        if s[pos] == "(":
            pos += 1
            n.kids.append(node())
            while s[pos] == ",":
                pos += 1
                n.kids.append(node())
            pos += 1                                                  # ')'
        m = re.match(r"([^:,();\[]*)(?:\[[^\]]*\])?(?::([0-9.eE+\-]+))?", s[pos:])
        pos += m.end()
        n.name = m.group(1).strip().strip("'\"")
        n.length = float(m.group(2) or 0.0)
        for k in n.kids:
            k.parent = n
        return n

    root = node()
    root.length = 0.0                                                 # the root's own branch does not count
    _set_depths(root)
    return root


def _set_depths(n: Node, d: float = 0.0):
    n.depth = d
    for k in n.kids:
        _set_depths(k, d + k.length)


def tips_of(n: Node) -> List[Node]:
    return [n] if not n.kids else [t for k in n.kids for t in tips_of(k)]


def internal_nodes(n: Node) -> List[Node]:
    return ([n] if n.kids else []) + [x for k in n.kids for x in internal_nodes(k)]


def prune(root: Node, keep: set) -> Node:
    """Drop tips not in `keep`; collapse nodes left with one child (their branch lengths add up)."""
    def rec(n: Node) -> Optional[Node]:
        if not n.kids:
            return n if n.name in keep else None
        kids = [c for c in (rec(k) for k in n.kids) if c is not None]
        if not kids:
            return None
        if len(kids) == 1:
            kids[0].length += n.length
            return kids[0]
        n.kids = kids
        for k in kids:
            k.parent = n
        return n

    r = rec(root)
    r.length, r.parent = 0.0, None
    _set_depths(r)
    return r


def mrca_depth(a: Node, b: Node) -> float:
    anc = set()
    x = a
    while x is not None:
        anc.add(id(x)); x = x.parent
    y = b
    while id(y) not in anc:
        y = y.parent
    return y.depth


def vcv(nodes_a: List[Node], nodes_b: List[Node]) -> np.ndarray:
    """Brownian-motion covariance: shared path length from the root (tips and internal nodes alike)."""
    return np.array([[mrca_depth(a, b) for b in nodes_b] for a in nodes_a])


# ============================================================================ statistics
def _whiten(C: np.ndarray) -> np.ndarray:
    """P with P'P = C^-1 (any such P gives the same GLS sums of squares)."""
    return np.linalg.inv(np.linalg.cholesky(C))


def kmult(Y: np.ndarray, C: np.ndarray) -> float:
    """Adams (2014) multivariate K, as geomorph::physignal."""
    n = len(Y)
    Ci = np.linalg.inv(C)
    one = np.ones((n, 1))
    a = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)
    R = Y - one @ a
    obs = np.trace(R.T @ R) / np.trace(R.T @ Ci @ R)
    expct = (np.trace(C) - n / Ci.sum()) / (n - 1)
    return float(obs / expct)


def effect_size(obs: float, rand: np.ndarray) -> float:
    r = np.r_[obs, rand]
    sd = r.std(ddof=1)
    return float((obs - r.mean()) / sd) if sd > 0 else float("nan")


def phylo_signal(Y, C, iters, rng) -> Dict:
    K = kmult(Y, C)
    rand = np.array([kmult(Y[rng.permutation(len(Y))], C) for _ in range(iters)])
    return {"Kmult": K, "P": float((1 + (rand >= K - 1e-12).sum()) / (iters + 1)), "Z": effect_size(K, rand)}


def pgls(Y: np.ndarray, x: np.ndarray, C: np.ndarray, iters: int, rng) -> Dict:
    """Y ~ x under Brownian motion by GLS; SS from the whitened data, residual randomisation of the null model
    (intercept only), as RRPP / geomorph::procD.pgls with a single predictor."""
    n = len(Y)
    P = _whiten(C)
    Yt = P @ Y
    X0 = P @ np.ones((n, 1))
    X1 = P @ np.column_stack([np.ones(n), x])

    def rss(X, Yy):
        B, *_ = np.linalg.lstsq(X, Yy, rcond=None)
        R = Yy - X @ B
        return float((R ** 2).sum())

    B0, *_ = np.linalg.lstsq(X0, Yt, rcond=None)
    fit0, res0 = X0 @ B0, Yt - X0 @ B0
    ss_tot, ss_res = rss(X0, Yt), rss(X1, Yt)
    ss = ss_tot - ss_res
    df, dfr = 1, n - 2
    F = (ss / df) / (ss_res / dfr)
    rand = []
    for _ in range(iters):
        Yp = fit0 + res0[rng.permutation(n)]
        s_t, s_r = rss(X0, Yp), rss(X1, Yp)
        rand.append(((s_t - s_r) / df) / (s_r / dfr))
    rand = np.array(rand)
    return {"Df": df, "SS": ss, "MS": ss / df, "Rsq": ss / ss_tot, "F": F, "Z": effect_size(F, rand),
            "P": float((1 + (rand >= F - 1e-12).sum()) / (iters + 1)), "SS_residual": ss_res, "SS_total": ss_tot}


def ancestral_states(Y: np.ndarray, tips: List[Node], nodes: List[Node]) -> np.ndarray:
    """Ancestral states under Brownian motion by GLS: a = mu + C_nt C_tt^-1 (Y - 1 mu), with mu the GLS root state."""
    Ctt = vcv(tips, tips)
    Cnt = vcv(nodes, tips)
    Ci = np.linalg.inv(Ctt)
    one = np.ones((len(tips), 1))
    mu = np.linalg.solve(one.T @ Ci @ one, one.T @ Ci @ Y)
    return mu + Cnt @ Ci @ (Y - one @ mu)


# ============================================================================ evolutionary models
MODELS = ("BM", "OU", "EB", "lambda", "white")
MODEL_PARAM = {"OU": "alpha", "EB": "a", "lambda": "lambda"}
MODEL_K = {"BM": 2, "white": 2, "OU": 3, "EB": 3, "lambda": 3}      # with sigma^2 and the root state, as geiger counts


def model_vcv(C: np.ndarray, model: str, p: float = 0.0) -> np.ndarray:
    """Tip covariance per unit sigma^2 under each model, geiger::fitContinuous's forms:
    BM      C, the shared path length from the root
    OU      one optimum at the root state: exp(-alpha (t_i + t_j - 2 s_ij)) (1 - exp(-2 alpha s_ij)) / (2 alpha),
            geiger's covariance form (its branch-length transform gives the same on ultrametric trees)
    EB      rate sigma^2 exp(a t) at time t from the root: (exp(a C) - 1) / a
    lambda  shared paths times lambda, root-to-tip lengths unchanged
    white   the identity: no phylogeny"""
    if model == "BM":
        return C
    if model == "white":
        return np.eye(len(C))
    if model == "lambda":
        V = C * p
        np.fill_diagonal(V, np.diag(C))
        return V
    if model == "EB":
        return C if abs(p) < 1e-12 else np.expm1(p * C) / p
    if model == "OU":
        d = np.diag(C)
        return np.exp(-p * (d[:, None] + d[None, :] - 2 * C)) * -np.expm1(-2 * p * C) / (2 * p)
    raise ValueError(model)


def profile_loglik(y: np.ndarray, V: np.ndarray) -> Tuple[float, float, float]:
    """log-likelihood of y ~ N(z0, sigma^2 V) with z0 (the GLS mean) and sigma^2 = Q / n at their maximum."""
    from scipy.linalg import solve_triangular
    n = len(y)
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return -np.inf, np.nan, np.nan
    ty = solve_triangular(L, y, lower=True)
    t1 = solve_triangular(L, np.ones(n), lower=True)
    z0 = float(t1 @ ty / (t1 @ t1))
    s2 = float(((ty - z0 * t1) ** 2).sum()) / n
    if not s2 > 0:
        return -np.inf, s2, z0
    return float(-0.5 * n * (np.log(2 * np.pi) + np.log(s2) + 1) - np.log(np.diag(L)).sum()), s2, z0


def default_bounds(C: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """geiger::fitContinuous's defaults: alpha in (exp(-500), e], a in [ln(1e-5) / depth, -1e-6], lambda in (0, 1]."""
    return {"alpha": (np.exp(-500), np.e), "a": (np.log(1e-5) / float(np.diag(C).max()), -1e-6),
            "lambda": (np.exp(-500), 1.0)}


def fit_model(y: np.ndarray, C: np.ndarray, model: str, bounds: Dict[str, Tuple[float, float]]) -> Dict:
    """Maximum-likelihood fit of one model to one variable: a grid over the model's parameter, then Brent's method
    between the grid points either side of the best (alpha on a log scale, where its likelihood surface is smooth)."""
    from scipy.optimize import minimize_scalar
    n = len(y)
    par, at_bound = np.nan, ""
    if model in MODEL_PARAM:
        name = MODEL_PARAM[model]
        lo, hi = bounds[name]
        logscale = name == "alpha"
        if logscale:            # below alpha * depth = 1e-8 OU is BM to machine precision: no need to search lower
            lo_s = np.log(max(lo, 1e-8 / float(np.diag(C).max())))
            to_p = np.exp
            grid = np.linspace(lo_s, np.log(hi), 61)
        else:
            to_p = float
            grid = np.linspace(lo, hi, 61)
        f = lambda u: -profile_loglik(y, model_vcv(C, model, to_p(u)))[0]
        vals = np.array([f(u) for u in grid])
        i = int(np.argmin(vals))
        best_u, best_v = grid[i], vals[i]
        a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        if b_ > a_:
            r = minimize_scalar(f, bounds=(a_, b_), method="bounded", options={"xatol": 1e-10})
            if r.fun < best_v:
                best_u, best_v = float(r.x), float(r.fun)
        par = float(to_p(best_u))
        tol = 1e-6 * (grid[-1] - grid[0])
        at_bound = "lower" if best_u - grid[0] < tol else "upper" if grid[-1] - best_u < tol else ""
        V = model_vcv(C, model, par)
    else:
        V = model_vcv(C, model)
    ll, s2, z0 = profile_loglik(y, V)
    k = MODEL_K[model]
    aicc = -2 * ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    return {"model": model, "logLik": ll, "k": k, "AICc": aicc, "sigsq": s2, "z0": z0,
            "param": MODEL_PARAM.get(model, ""), "value": par, "at_bound": at_bound}


def fit_models(y: np.ndarray, C: np.ndarray, models=MODELS, bounds=None) -> List[Dict]:
    """every model for one variable, with delta-AICc, Akaike weights, OU half-life and the lambda tests."""
    from scipy.stats import chi2
    bounds = {**default_bounds(C), **(bounds or {})}
    fits = [fit_model(y, C, m, bounds) for m in models]
    best = min(f["AICc"] for f in fits)
    w = np.array([np.exp(-0.5 * (f["AICc"] - best)) for f in fits])
    depth = float(np.diag(C).max())
    for f, wi in zip(fits, w / w.sum()):
        f["dAICc"], f["weight"] = f["AICc"] - best, float(wi)
        f["half_life"] = float(np.log(2) / f["value"]) if f["model"] == "OU" else np.nan
        f["half_life_over_depth"] = f["half_life"] / depth if f["model"] == "OU" else np.nan
        if f["model"] == "lambda":
            ll0 = profile_loglik(y, model_vcv(C, "lambda", 0.0))[0]
            ll1 = profile_loglik(y, C)[0]
            f["P_lambda_vs_0"] = float(chi2.sf(max(0.0, 2 * (f["logLik"] - ll0)), 1))
            f["P_lambda_vs_1"] = float(chi2.sf(max(0.0, 2 * (f["logLik"] - ll1)), 1))
    return fits


# ============================================================================ variable-rate Brownian motion
def tree_arrays(root: Node):
    """nodes in preorder (root first), each node's parent index (-1 for the root) and branch length, and the
    ancestry matrix B (B[i, j] = 1 when the branch above node j lies on the path from the root to node i)."""
    nodes: List[Node] = []

    def rec(n):
        nodes.append(n)
        for k in n.kids:
            rec(k)

    rec(root)
    ix = {id(n): i for i, n in enumerate(nodes)}
    parent = np.array([ix[id(n.parent)] if n.parent is not None else -1 for n in nodes])
    length = np.array([0.0 if n.parent is None else n.length for n in nodes])
    B = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        j = i
        while parent[j] >= 0:
            B[i, j] = 1.0
            j = parent[j]
    return nodes, parent, length, B


def _logmean(u: np.ndarray, v: np.ndarray):
    """mean rate along a branch whose log rate runs linearly from u (parent) to v (child), and its derivatives:
    (e^u - e^v) / (u - v), phytools' ln.mean."""
    d = u - v
    ev = np.exp(v)
    small = np.abs(d) < 1e-6
    ds = np.where(small, 1.0, d)
    phi = np.where(small, 1 + d / 2 + d * d / 6, np.expm1(ds) / ds)                     # (e^d - 1) / d
    dphi = np.where(small, 0.5 + d / 3, (np.exp(ds) * ds - np.expm1(ds)) / (ds * ds))   # its derivative
    return ev * phi, ev * dphi, ev * (phi - dphi)                                        # L, dL/du, dL/dv


def multirate_bm(root: Node, tips: List[Node], y: np.ndarray, lam: float = 1.0, maxiter: int = 20000) -> Dict:
    """Variable-rate Brownian motion (Revell 2021, phytools::multirateBM): a log rate at every node, each branch
    evolving at the mean of the rates at its two ends, the tip data normal given those rates, and the log rates
    themselves penalised as Brownian motion along the tree with weight lam. Maximised with L-BFGS-B within
    phytools' bounds (log(var(y) / tree height) +/- 10) from its start (the Brownian-motion ML rate)."""
    from scipy.optimize import minimize
    nodes, parent, length, B = tree_arrays(root)
    N = len(nodes)
    tip_ix = {id(n): i for i, n in enumerate(nodes)}
    ti = np.array([tip_ix[id(t)] for t in tips])
    A = B[ti]                                                    # tips x branches
    nr = np.arange(1, N)                                         # every node but the root (index 0)
    pen_len = np.maximum(length, 1e-12 * length.max())           # zero-length branches would make Vn singular
    Vn = (B[nr] * pen_len) @ B[nr].T                             # shared depth among the non-root nodes
    Vn_i = np.linalg.inv(Vn)
    _, Vn_logdet = np.linalg.slogdet(Vn)
    n = len(y)

    def obj(th):
        L_, dLu, dLv = _logmean(th[parent[nr]], th[nr])
        w = np.zeros(N)
        w[nr] = length[nr] * L_
        C = (A * w) @ A.T
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e300, np.zeros(N)
        Ci = np.linalg.inv(C)
        one = np.ones(n)
        a = (one @ Ci @ y) / (one @ Ci @ one)
        r = y - a
        Cr = Ci @ r
        ll1 = -0.5 * (n * np.log(2 * np.pi) + 2 * np.log(np.diag(Lc)).sum() + r @ Cr)
        z = th[nr] - th[0]
        s = Vn_i @ z
        ll2 = -0.5 * ((N - 1) * np.log(2 * np.pi) + Vn_logdet + z @ s)
        # gradients: d ll1 / d w_j = 1/2 A_j' (C^-1 r r' C^-1 - C^-1) A_j (the GLS mean drops out at its optimum)
        M = np.outer(Cr, Cr) - Ci
        gw = 0.5 * np.einsum("ij,ik,kj->j", A, M, A)
        g = np.zeros(N)
        np.add.at(g, parent[nr], gw[nr] * length[nr] * dLu)
        g[nr] += gw[nr] * length[nr] * dLv
        g[nr] += -lam * s
        g[0] += lam * s.sum()
        return -(ll1 + lam * ll2), -g

    height = float((B * length).sum(axis=1).max())
    vv = np.log(np.var(y, ddof=1) / height)
    C0 = (A * length) @ A.T
    s2_bm = profile_loglik(y, C0)[1]
    th0 = np.full(N, np.clip(np.log(s2_bm), vv - 10, vv + 10))
    res = minimize(obj, th0, jac=True, method="L-BFGS-B", bounds=[(vv - 10, vv + 10)] * N,
                   options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-13, "gtol": 1e-8})
    th = res.x
    edge_rate = np.full(N, np.nan)
    edge_rate[nr] = _logmean(th[parent[nr]], th[nr])[0]
    w = np.zeros(N)
    w[nr] = length[nr] * edge_rate[nr]
    C = (A * w) @ A.T
    one = np.ones(n); Ci = np.linalg.inv(C); a = (one @ Ci @ y) / (one @ Ci @ one); r = y - a
    ll = float(-0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + r @ Ci @ r))    # unpenalised, as phytools
    return {"nodes": nodes, "parent": parent, "length": length, "node_rate": np.exp(th), "edge_rate": edge_rate,
            "penalised_logLik": float(-res.fun), "logLik": ll, "k": N + 1, "converged": bool(res.success),
            "message": str(res.message), "lambda": lam}


def branch_changes(root: Node, tips: List[Node], y: np.ndarray, nodes, parent, length) -> Tuple[np.ndarray, np.ndarray]:
    """each branch's change between the Brownian-motion ancestral states (tips observed) and that change divided by
    its standard deviation under the fitted Brownian motion, sqrt(sigma^2 x branch length)."""
    inner = [n for n in nodes if n.kids]
    anc = ancestral_states(y[:, None], tips, inner)[:, 0]
    val = {id(n): float(v) for n, v in zip(inner, anc)}
    val.update({id(t): float(v) for t, v in zip(tips, y)})
    s2 = profile_loglik(y, vcv(tips, tips))[1]
    state = np.array([val[id(n)] for n in nodes])
    delta = np.full(len(nodes), np.nan)
    delta[1:] = state[1:] - state[parent[1:]]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = delta / np.sqrt(s2 * length)
    return delta, z


def clade_label(n: Node) -> str:
    t = tips_of(n)
    return t[0].name if len(t) == 1 else f"{t[0].name} .. {t[-1].name} ({len(t)} tips)"


# ============================================================================ data
def _is_number(v: str) -> bool:
    try:
        float(v); return True
    except ValueError:
        return False


def _key_forms(s: str) -> List[str]:
    """an id with and without Descriptron's annotation suffix ("x.png_3") and extension."""
    s = str(s).strip(); base = re.sub(r"\.(png|jpe?g|tiff?|bmp)_\d+$", r".\1", s, flags=re.I)
    stem = re.sub(r"\.(png|jpe?g|tiff?|bmp)$", "", base, flags=re.I)
    forms = [s, base, stem, Path(stem).name]
    # some steps write file names with spaces replaced by underscores ("Scan 001" -> "Scan_001")
    return forms + [f.replace(" ", "_") for f in forms if " " in f]


def load_group_map(path: str, group_col: Optional[str] = None) -> Dict[str, str]:
    """group labels CSV (filename + group_label/species): id -> species."""
    rows = list(csv.DictReader(open(path, newline=""), delimiter="\t" if str(path).endswith(".tsv") else ","))
    lc = {c.strip().lower(): c for c in rows[0]}
    fcol = lc.get("filename") or lc.get("file") or lc.get("specimen") or list(rows[0])[0]
    gcol = group_col or lc.get("group_label") or lc.get("species") or list(rows[0])[1]
    m: Dict[str, str] = {}
    for r in rows:
        for k in _key_forms(r[fcol]):
            m.setdefault(k, r[gcol])
    return m


def load_traits(path: str, species_col: Optional[str], species_regex: Optional[str], id_col: Optional[str],
                groups: Optional[Dict[str, str]] = None):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        sys.exit(f"{path}: no rows")
    cols = list(rows[0])
    if groups is not None and not (species_col and species_col in cols):     # species from a group-labels file
        idc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        sp = [next((groups[k] for k in _key_forms(r[idc]) if k in groups), None) for r in rows]
        species_col = idc                                                # the id column is not a trait
    elif species_col and species_col in cols:
        sp = [r[species_col] for r in rows]
    else:
        idc = id_col or next((c for c in ("filename", "specimen_id", "specimen", "image", "id", "file") if c in cols), cols[0])
        if not species_regex:
            sys.exit(f"{path}: give --species_col or --species_regex (no species column found)")
        rx = re.compile(species_regex)
        sp = []
        for r in rows:
            m = rx.search(r[idc])
            sp.append(m.group(1) if m else None)
    num = [c for c in cols if c != species_col and all(_is_number(r[c]) or r[c] == "" for r in rows)
           and any(r[c] != "" for r in rows)]
    M = np.array([[float(r[c]) if r[c] != "" else np.nan for c in num] for r in rows])
    by: Dict[str, List[int]] = {}
    for i, s in enumerate(sp):
        if s:
            by.setdefault(s, []).append(i)
    species = sorted(by)
    if not species:
        sys.exit(f"{path}: no row could be given a species (check the id column against the group labels / regex)")
    means = np.array([np.nanmean(M[by[s]], axis=0) for s in species])
    keep = ~np.isnan(means).any(axis=0)                  # drop columns missing for any species
    return species, [c for c, k in zip(num, keep) if k], means[:, keep], {s: len(v) for s, v in by.items()}


def scale(M: np.ndarray, how: str) -> np.ndarray:
    if how == "standardize":
        sd = M.std(axis=0, ddof=1)
        ok = sd > 0
        return (M[:, ok] - M[:, ok].mean(0)) / sd[ok]
    if how == "log":
        if (M <= 0).any():
            sys.exit("--scale log needs positive values")
        return np.log(M)
    return M


# ============================================================================ plot
def plot_morphospace(pc_tips, pc_nodes, tips, nodes, ev, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = {id(t): pc_tips[i] for i, t in enumerate(tips)}
    pos.update({id(n): pc_nodes[i] for i, n in enumerate(nodes)})
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in nodes:
        for k in n.kids:
            a, b = pos[id(n)], pos[id(k)]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#6b7780", lw=0.9, zorder=1)
    ax.scatter(pc_nodes[:, 0], pc_nodes[:, 1], s=14, color="white", edgecolor="#6b7780", zorder=2, label="ancestral states")
    ax.scatter(pc_tips[:, 0], pc_tips[:, 1], s=30, color="#0072B2", zorder=3, label="species means")
    for i, t in enumerate(tips):
        ax.annotate(t.name, pc_tips[i, :2], xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_rate_tree(nodes, parent, length, edge_rate, path, title):
    """rectangular tree, each branch coloured by its rate relative to the median branch (log scale: slow blue,
    median grey, fast red)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    N = len(nodes)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = x[parent[i]] + length[i]
    tips = [i for i in range(N) if not nodes[i].kids]
    yv = np.zeros(N)
    yv[tips] = np.arange(len(tips))[::-1]
    for i in range(N - 1, -1, -1):                      # preorder reversed: children before parents
        if nodes[i].kids:
            kids = [j for j in range(N) if parent[j] == i]
            yv[i] = np.mean(yv[kids])
    rel = np.log10(edge_rate / np.nanmedian(edge_rate[1:]))
    lim = max(0.5, float(np.nanmax(np.abs(rel[1:]))))
    cmap = LinearSegmentedColormap.from_list("rate", ["#2166ac", "#b8bec4", "#b2182b"])
    norm = Normalize(-lim, lim)
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.13 * len(tips) + 1.2)))
    for i in range(1, N):
        p = parent[i]
        c = cmap(norm(rel[i]))
        ax.plot([x[p], x[i]], [yv[i], yv[i]], color=c, lw=1.6, solid_capstyle="butt")
        ax.plot([x[p], x[p]], [yv[p], yv[i]], color="#8a9399", lw=0.6)
    for i in tips:
        ax.text(x[i] + 0.01 * x.max(), yv[i], nodes[i].name, va="center", fontsize=5.5)
    ax.set_xlim(0, x.max() * 1.45); ax.set_yticks([]); ax.set_xlabel("branch length")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("log10 rate / median rate", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:80]


def _norm_name(s: str) -> str:
    """spaces and underscores count as the same character when tips are matched to species."""
    return re.sub(r"[\s_]+", "_", str(s).strip())


def load_tree(path: str, tip_regex: Optional[str] = None) -> Tuple[Node, List[str]]:
    """the tree with tips renamed to species (--tip_regex) and duplicate species pruned to their first tip."""
    root = parse_newick(Path(path).read_text().strip().split(";")[0] + ";")
    rx = re.compile(tip_regex) if tip_regex else None
    notes, seen, keep = [], set(), set()
    for t in tips_of(root):
        if rx:
            m = rx.search(t.name)
            if not m:
                notes.append(f"tip '{t.name}' does not match --tip_regex - kept as it is")
            else:
                t.name = m.group(1)
        t.name = _norm_name(t.name)
        if t.name in seen:
            notes.append(f"species '{t.name}' is on several tips - the first is kept")
            t.name = f"__dup__{id(t)}"
        else:
            seen.add(t.name)
        keep.add(t.name)
    if any(n.startswith("__dup__") for n in keep):
        root = prune(root, {n for n in keep if not n.startswith("__dup__")})
    return root, notes


def _copy_tree(n: Node, parent: Optional[Node] = None) -> Node:
    c = Node(n.name, n.length)
    c.parent = parent
    c.kids = [_copy_tree(k, c) for k in n.kids]
    return c


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True, help="Newick tree whose tips are species names")
    ap.add_argument("--traits", action="append", required=True, metavar="NAME=CSV",
                    help="a trait table; repeat for several sets (shape, measurements, colour, texture, ...)")
    ap.add_argument("--species_col", default=None, help="column holding the species (default: use --species_regex)")
    ap.add_argument("--species_regex", default=None, help="regex with one group extracting the species from the id column")
    ap.add_argument("--groups", default=None, help="group labels CSV (filename + group_label/species) giving each "
                    "specimen's species, as for the rest of the pipeline (instead of --species_col/--species_regex)")
    ap.add_argument("--id_col", default=None, help="id column for --species_regex (default: filename/specimen_id/...)")
    ap.add_argument("--columns", action="append", default=[], metavar="NAME=REGEX",
                    help="keep only the columns of trait set NAME that match REGEX")
    ap.add_argument("--scale", action="append", default=[], metavar="NAME=none|standardize|log")
    ap.add_argument("--size_col", default=r"centroid|csize",
                    help="regex for a size column kept out of the trait set and used as the logCS predictor")
    ap.add_argument("--pgls", action="append", default=[], metavar="SET~PREDICTOR",
                    help="PREDICTOR is logCS (from a set's centroid-size column) or SET2:COLUMN")
    ap.add_argument("--analyses", nargs="+", default=["signal", "morphospace"],
                    choices=["signal", "pgls", "morphospace", "models", "rates"])
    ap.add_argument("--tip_regex", default=None,
                    help='regex with one group extracting the species from each tree tip (e.g. "_([A-Z][a-z]+ .+)$" '
                         'for tips such as "QH001-S1_Abemus chloropterus")')
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                    help="models compared by 'models' (default all)")
    ap.add_argument("--model_bounds", action="append", default=[], metavar="PARAM=LO,HI",
                    help="bounds for alpha (OU), a (EB) or lambda; default geiger::fitContinuous's")
    ap.add_argument("--model_columns", action="append", default=[], metavar="NAME=REGEX",
                    help="variables of trait set NAME analysed by 'models' and 'rates' (default all its variables)")
    ap.add_argument("--rate_lambda", type=float, default=1.0,
                    help="smoothing of the variable rates (phytools::multirateBM lambda; larger = smoother)")
    ap.add_argument("--rate_top", type=int, default=10, help="fastest branches listed in the report per variable")
    ap.add_argument("--iterations", type=int, default=999); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    root, tree_notes = load_tree(a.tree, a.tip_regex)
    colsel = dict(x.split("=", 1) for x in a.columns)
    modsel = dict(x.split("=", 1) for x in a.model_columns)
    mbounds = {k: tuple(float(v) for v in b.split(",")) for k, b in (x.split("=", 1) for x in a.model_bounds)}
    gmap = load_group_map(a.groups) if a.groups else None
    scales = dict(x.split("=", 1) for x in a.scale)

    sets = {}
    for spec in a.traits:
        name, path = spec.split("=", 1)
        sp, cols, M, counts = load_traits(path, a.species_col, a.species_regex, a.id_col, gmap)
        sp = [_norm_name(s) for s in sp]
        counts = {_norm_name(k): v for k, v in counts.items()}
        if name in colsel:
            rx = re.compile(colsel[name]); k = [i for i, c in enumerate(cols) if rx.search(c)]
            cols, M = [cols[i] for i in k], M[:, k]
        # a size column (centroid size) is kept aside: it is a predictor (logCS), not part of the trait set
        szi = [i for i, c in enumerate(cols) if re.search(a.size_col, c, re.I)]
        size = M[:, szi[0]] if szi else None
        keep = [i for i in range(len(cols)) if i not in szi]
        sets[name] = {"species": sp, "cols": [cols[i] for i in keep], "raw": M[:, keep], "size": size,
                      "counts": counts}

    report = ["# Phylogenetic comparative analyses", "", f"Tree: {a.tree}", ""]
    if tree_notes:
        report += [f"- {x}" for x in tree_notes[:20]] + ([f"- ... {len(tree_notes) - 20} more"] if len(tree_notes) > 20 else []) + [""]
    signal_rows, pgls_rows, model_rows, rate_summary = [], [], [], []
    for name, S in sets.items():
        tipnames = {t.name for t in tips_of(root)}
        common = [s for s in S["species"] if s in tipnames]
        if len(common) < 4:
            report.append(f"{name}: only {len(common)} species match the tree - skipped"); continue
        tree = prune(_copy_tree(root), set(common))
        tips = tips_of(tree)
        idx = [S["species"].index(t.name) for t in tips]
        how = scales.get(name, "none")
        Y = scale(S["raw"][idx], how)
        if Y.shape[1] == 0 or not np.isfinite(Y).all() or float(Y.var(axis=0).sum()) == 0:
            report += [f"## {name}", "no variable differs among the species (for example empty homology cells) - skipped", ""]
            continue
        C = vcv(tips, tips)
        S.update(tree=tree, tips=tips, Y=Y, C=C, idx=idx)
        dropped = sorted(set(S["species"]) - set(common))
        report += [f"## {name}", f"{len(tips)} species, {Y.shape[1]} variables (scale: {how})"
                   + (f"; not in the tree: {', '.join(dropped)}" if dropped else ""), ""]
        if "signal" in a.analyses:
            r = phylo_signal(Y, C, a.iterations, rng)
            signal_rows.append({"trait_set": name, "n_species": len(tips), "n_variables": Y.shape[1], "scale": how, **r})
            report.append(f"Phylogenetic signal: Kmult = {r['Kmult']:.4f}, P = {r['P']:.4f}, Z = {r['Z']:.2f}")
        if "morphospace" in a.analyses and Y.shape[1] < 2:
            report.append("Phylomorphospace: fewer than two variables - not drawn")
        elif "morphospace" in a.analyses:
            nodes = internal_nodes(tree)
            anc = ancestral_states(Y, tips, nodes)
            mu = Y.mean(0)
            U, sv, Vt = np.linalg.svd(Y - mu, full_matrices=False)
            ev = sv ** 2 / (sv ** 2).sum()
            pc_t, pc_n = (Y - mu) @ Vt.T, (anc - mu) @ Vt.T
            np.savetxt(out / f"{name}_morphospace_tips.csv", pc_t[:, :min(5, pc_t.shape[1])], delimiter=",",
                       header=",".join(f"PC{i + 1}" for i in range(min(5, pc_t.shape[1]))), comments="")
            plot_morphospace(pc_t, pc_n, tips, nodes, ev, out / f"{name}_phylomorphospace.png",
                             f"{name}: phylomorphospace ({len(tips)} species)")
            report.append(f"Phylomorphospace: {name}_phylomorphospace.png (PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%)")
        if "models" in a.analyses or "rates" in a.analyses:
            cols = [c for c in S["cols"] if name not in modsel or re.search(modsel[name], c)]
            Yraw = np.log(S["raw"][idx]) if how == "log" else S["raw"][idx]       # models are scale-free otherwise
            depth = np.diag(C)
            if depth.max() - depth.min() > 1e-6 * depth.max():
                report.append(f"The tree is not ultrametric (root-to-tip {depth.min():.4g}-{depth.max():.4g}): rates, "
                              "alpha and a are per unit of branch length, not per unit of time.")
        if "models" in a.analyses:
            best = {}
            for c in cols:
                y = Yraw[:, S["cols"].index(c)]
                if np.ptp(y) == 0:
                    continue
                fits = fit_models(y, C, a.models, mbounds)
                for f in fits:
                    model_rows.append({"trait_set": name, "variable": c, "n_species": len(tips), **f})
                b = min(fits, key=lambda f: f["AICc"])
                best[b["model"]] = best.get(b["model"], 0) + 1
            report.append(f"Evolutionary models ({len(cols)} variables, lowest AICc): "
                          + ", ".join(f"{m} {best.get(m, 0)}" for m in a.models) + f"  (see {name}_model_fits.csv)")
            nb = sum(1 for r in model_rows if r["trait_set"] == name and r["at_bound"])
            if nb:
                report.append(f"  {nb} fits have a parameter at its bound (column at_bound): widen --model_bounds or "
                              "read those estimates as limits")
        if "rates" in a.analyses:
            rows = []
            for c in cols:
                y = Yraw[:, S["cols"].index(c)]
                if np.ptp(y) == 0:
                    continue
                R = multirate_bm(tree, tips, y, a.rate_lambda)
                delta, z = branch_changes(tree, tips, y, R["nodes"], R["parent"], R["length"])
                med = float(np.nanmedian(R["edge_rate"][1:]))
                for i in range(1, len(R["nodes"])):
                    rows.append({"trait_set": name, "variable": c, "branch_to": clade_label(R["nodes"][i]),
                                 "is_tip": not R["nodes"][i].kids, "length": R["length"][i],
                                 "rate": R["edge_rate"][i], "rate_over_median": R["edge_rate"][i] / med,
                                 "change": delta[i], "change_over_sd": z[i]})
                tag = _safe(f"{name}__{c}")
                plot_rate_tree(R["nodes"], R["parent"], R["length"], R["edge_rate"], out / f"{tag}_rate_tree.png",
                               f"{name}: {c} - variable-rate Brownian motion (lambda = {a.rate_lambda:g})")
                order = np.argsort(-R["edge_rate"][1:])[:a.rate_top] + 1
                rate_summary.append({"trait_set": name, "variable": c, "penalised_logLik": R["penalised_logLik"],
                                     "logLik": R["logLik"], "k": R["k"], "lambda": R["lambda"],
                                     "converged": R["converged"],
                                     "fastest": "; ".join(f"{clade_label(R['nodes'][i])} x{R['edge_rate'][i] / med:.1f}"
                                                          for i in order)})
                if not R["converged"]:
                    report.append(f"  rates {c}: optimiser did not converge ({R['message']})")
            if rows:
                with open(out / f"{name}_branch_rates.csv", "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
                report.append(f"Variable rates: {name}_branch_rates.csv and one <set>__<variable>_rate_tree.png per variable")
                for r in rate_summary:
                    if r["trait_set"] == name:
                        report.append(f"  {r['variable']}: fastest branches (rate / median) {r['fastest']}")
        report.append("")

    if "pgls" in a.analyses:
        for f in a.pgls:
            ys, pred = [x.strip() for x in f.split("~")]
            S = sets.get(ys)
            if S is None or "Y" not in S:
                report.append(f"PGLS {f}: trait set {ys} not analysed - skipped"); continue
            if pred == "logCS":
                if S["size"] is None:
                    report.append(f"PGLS {f}: no size column in {ys} (--size_col) - skipped"); continue
                xv = np.log(np.array([S["size"][S["species"].index(t.name)] for t in S["tips"]]))
            else:
                s2, cc = pred.split(":", 1); src = sets.get(s2)
                if src is None or cc not in src["cols"]:
                    report.append(f"PGLS {f}: predictor column not found - skipped"); continue
                xv = np.array([src["raw"][src["species"].index(t.name), src["cols"].index(cc)] for t in S["tips"]])
            r = pgls(S["Y"], xv, S["C"], a.iterations, rng)
            pgls_rows.append({"model": f, **r})
            report.append(f"PGLS {f}: R2 = {r['Rsq']:.4f}, F = {r['F']:.3f}, P = {r['P']:.4f}")

    for name in sets:
        rows = [r for r in model_rows if r["trait_set"] == name]
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{name}_model_fits.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for fn, rows in (("phylogenetic_signal.csv", signal_rows), ("pgls.csv", pgls_rows), ("rate_fits.csv", rate_summary)):
        if rows:
            with open(out / fn, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "phylo_report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
