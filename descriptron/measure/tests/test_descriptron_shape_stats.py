"""Known-answer tests for descriptron_shape_stats (the ANOVA/trajectory/disparity statistics were also checked
against R's RRPP on identical aligned data: SS, R2, F, trajectory distances/angles and disparities identical)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_shape_stats as ss  # noqa: E402

BASE = np.array([[0, 0], [40, 5], [80, 0], [100, 30], [80, 60], [40, 65], [0, 60], [-15, 30]], float)


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def test_gpa_removes_position_rotation_scale():
    rng = np.random.default_rng(0)
    X = np.array([(rot(rng.uniform(0, 6.28)) @ BASE.T).T * rng.uniform(1, 5) + rng.uniform(0, 500, 2) for _ in range(6)])
    Z, cs = ss.gpa(X)
    assert np.abs(Z - Z[0]).max() < 1e-8                       # identical shapes coincide
    assert np.allclose(np.sqrt((Z ** 2).sum(axis=(1, 2))), 1)


def test_anova_detects_group_and_not_noise():
    rng = np.random.default_rng(1)
    eff = {"a": 0, "b": rng.normal(0, 2, BASE.shape)}
    X, g, junk = [], [], []
    for grp in ("a", "b"):
        for _ in range(15):
            X.append(BASE + eff[grp] + rng.normal(0, 0.8, BASE.shape)); g.append(grp); junk.append(str(rng.integers(2)))
    Y = ss.flat(ss.gpa(np.array(X))[0])
    d = ss.Design("group + junk", {"group": g, "junk": junk}, {"group": "factor", "junk": "factor"})
    tab = {r["term"]: r for r in ss.procrustes_anova(Y, d, 199, np.random.default_rng(0))}
    assert tab["group"]["P"] < 0.01 and tab["junk"]["P"] > 0.05
    assert abs(tab["group"]["SS"] + tab["junk"]["SS"] + tab["Residuals"]["SS"] - tab["Total"]["SS"]) < 1e-12


def test_newick_covariance_and_kmult_brownian():
    tips, C = ss.parse_newick("((a:1,b:1):1,(c:1,d:1):1);")
    assert tips == ["a", "b", "c", "d"] and C[0, 0] == 2 and C[0, 1] == 1 and C[0, 2] == 0
    nwk = "(((a:1,b:1):1,(c:1,d:1):1):1,((e:1,f:1):1,(g:1,h:1):1):1);"
    tips, C = ss.parse_newick(nwk)
    L = np.linalg.cholesky(C)
    rng = np.random.default_rng(3)
    ks_bm = [ss.kmult(L @ rng.normal(size=(8, 6)), C) for _ in range(200)]
    ks_rand = [ss.kmult(rng.normal(size=(8, 6)), C) for _ in range(200)]
    assert 0.8 < np.mean(ks_bm) < 1.3 and np.mean(ks_rand) < np.mean(ks_bm) - 0.2


def test_cr_low_for_independent_modules():
    rng = np.random.default_rng(4)
    n = 60
    fa, fb = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    A = fa @ rng.normal(size=(1, 8)) + 0.3 * rng.normal(size=(n, 8))      # module 1: landmarks 1-4 (8 coords)
    B = fb @ rng.normal(size=(1, 8)) + 0.3 * rng.normal(size=(n, 8))      # module 2: landmarks 5-8
    mods = [1, 1, 1, 1, 2, 2, 2, 2]
    r = ss.modularity(np.hstack([A, B]), mods, 199, np.random.default_rng(0))
    assert r["CR"] < 0.4 and r["P_CR (share of random partitions with CR <= observed)"] < 0.05
    f = rng.normal(size=(n, 1))
    one = f @ rng.normal(size=(1, 16)) + 0.3 * rng.normal(size=(n, 16))
    assert ss.cr_coefficient(one, mods) > 0.7


def test_asymmetry_detects_directional_asymmetry():
    sym = np.array([[-30, 0], [-20, 30], [0, 40], [20, 30], [30, 0], [0, -20]], float)   # 1<->5, 2<->4 mirror pairs
    pairs = [(1, 5), (2, 4)]
    rng = np.random.default_rng(5)
    noisy = np.array([sym + rng.normal(0, 0.5, sym.shape) for _ in range(25)])
    shifted = noisy.copy(); shifted[:, 0, 1] += 4                               # left landmark 1 consistently higher
    r0 = ss.asymmetry(noisy, pairs, 199, np.random.default_rng(0))
    r1 = ss.asymmetry(shifted, pairs, 199, np.random.default_rng(0))
    assert r0["P_DA"] > 0.05 and r1["P_DA"] < 0.01 and r1["directional_asymmetry"] > 3 * r0["directional_asymmetry"]


def test_assignment_with_typicality():
    rng = np.random.default_rng(6)
    A = rng.normal(0, 1, (20, 3)); B = rng.normal(0, 1, (20, 3)) + [6, 0, 0]
    unl = np.array([[0.2, 0.1, -0.3], [6.1, 0.3, 0.2], [0, 25, 0]])            # from A, from B, unlike both
    P = np.vstack([A, B, unl])
    groups = ["A"] * 20 + ["B"] * 20 + [None] * 3
    r = ss.assign(P, groups, k=3)
    got = [u["most_likely"] for u in r["unlabelled"]]
    assert got[:2] == ["A", "B"] and got[2].startswith("none") and r["loo_accuracy"] >= 0.95


def test_mantel_detects_geographic_pattern():
    rng = np.random.default_rng(7)
    lat, lon = rng.uniform(-10, 10, 30), rng.uniform(20, 40, 30)
    Y = np.column_stack([lat, lon]) @ rng.normal(size=(2, 10)) + rng.normal(0, 1, (30, 10))
    Dg = ss.haversine_km(lat, lon); Ds = np.linalg.norm(Y[:, None] - Y[None], axis=2)
    r = ss.mantel(Ds, Dg, 199, np.random.default_rng(0))
    assert r["r"] > 0.5 and r["P"] < 0.01


def test_effect_size_matches_rrpp_formula_shape():
    rng = np.random.default_rng(8)
    rand = rng.gamma(2, 0.5, 999)
    assert ss.effect_size(np.median(rand), rand) == pytest.approx(0, abs=0.2)
    assert ss.effect_size(rand.max() * 5, rand) > 3
