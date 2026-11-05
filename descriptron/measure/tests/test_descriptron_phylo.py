"""descriptron_phylo: tree covariance, pruning, Kmult, PGLS, ancestral states and the command line.

Values against geomorph/ape/phytools are checked by validation/validate_phylo_*.py (they need R); these tests pin
the properties that hold without R: known covariances, Kmult = 1 on a star tree with unit branches, PGLS = OLS on a
star tree, the root state = the GLS mean, and identical results whether the tree is pruned before or by the program.
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import descriptron_phylo as dp          # noqa: E402

TREE = "((A:1,B:1):1,(C:0.5,D:0.5):1.5);"


def test_vcv_known_tree():
    t = dp.parse_newick(TREE); tips = dp.tips_of(t)
    assert [x.name for x in tips] == list("ABCD")
    C = dp.vcv(tips, tips)
    np.testing.assert_allclose(C, [[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1.5], [0, 0, 1.5, 2]])


def test_prune_keeps_path_lengths():
    t = dp.prune(dp.parse_newick(TREE), {"A", "B", "C"}); tips = dp.tips_of(t)
    assert sorted(x.name for x in tips) == ["A", "B", "C"]
    C = dp.vcv(tips, tips); i = {x.name: k for k, x in enumerate(tips)}
    assert C[i["C"], i["C"]] == 2 and C[i["A"], i["B"]] == 1 and C[i["A"], i["C"]] == 0


def test_kmult_star_tree_is_one():
    rng = np.random.default_rng(0)
    star = dp.parse_newick("(" + ",".join(f"t{k}:1" for k in range(12)) + ");"); tips = dp.tips_of(star)
    C = dp.vcv(tips, tips)
    for p in (1, 5):
        assert abs(dp.kmult(rng.normal(size=(12, p)), C) - 1) < 1e-12


def test_pgls_star_tree_equals_ols():
    rng = np.random.default_rng(1)
    star = dp.parse_newick("(" + ",".join(f"t{k}:1" for k in range(15)) + ");"); tips = dp.tips_of(star)
    x = rng.normal(size=15); Y = np.outer(x, [1.0, -0.5]) + rng.normal(size=(15, 2))
    r = dp.pgls(Y, x, dp.vcv(tips, tips), 99, np.random.default_rng(2))
    X = np.column_stack([np.ones(15), x]); res = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    Yc = Y - Y.mean(0)
    assert abs(r["Rsq"] - (1 - (res ** 2).sum() / (Yc ** 2).sum())) < 1e-12


def test_root_state_is_gls_mean():
    t = dp.parse_newick(TREE); tips = dp.tips_of(t); nodes = dp.internal_nodes(t)
    Y = np.array([[1.0], [3.0], [10.0], [14.0]])
    anc = dp.ancestral_states(Y, tips, nodes)
    Ci = np.linalg.inv(dp.vcv(tips, tips)); one = np.ones(4)
    assert abs(anc[0, 0] - (one @ Ci @ Y[:, 0]) / (one @ Ci @ one)) < 1e-12      # nodes[0] is the root


def _write(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def test_cli_prunes_tree_itself(tmp_path):
    """Running on a larger tree gives the same numbers as running on the tree cut to the species with data."""
    rng = np.random.default_rng(3)
    big = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(X:1,Y:1):1.5):0.5);"      # X, Y have no data
    sub = dp.prune(dp.parse_newick(big), set("ABCDEF"))
    (tmp_path / "big.nwk").write_text(big)
    (tmp_path / "sub.nwk").write_text(_newick(sub) + ";")
    rows = [[s, f"{s}_{k}"] + list(rng.normal(size=4)) for s in "ABCDEF" for k in range(3)]
    _write(tmp_path / "t.csv", rows, ["species", "file", "v1", "v2", "v3", "v4"])
    outs = []
    for tree in ("big.nwk", "sub.nwk"):
        od = tmp_path / ("out_" + tree)
        dp.main(["--tree", str(tmp_path / tree), "--traits", f"t={tmp_path / 't.csv'}", "--species_col", "species",
                 "--analyses", "signal", "--iterations", "99", "--seed", "1", "--out_dir", str(od)])
        outs.append(sorted(p.read_text() for p in od.glob("*.csv")))
    assert outs[0] == outs[1]


def _newick(n):
    if not n.kids:
        return f"{n.name}:{n.length!r}"
    inner = ",".join(_newick(k) for k in n.kids)
    return f"({inner}):{n.length!r}" if n.parent is not None else f"({inner})"


# ---------------------------------------------------------------- evolutionary models and variable rates
# (values against geiger::fitContinuous and phytools::multirateBM: validation/validate_phylo_models_vs_r.py)
NONULTRA = "(((A:1,B:0.5):0.7,(C:0.4,D:1.2):0.3):0.5,((E:0.9,F:0.6):0.4,G:1.1):0.8);"


def test_model_vcv_limits():
    t = dp.parse_newick(NONULTRA); tips = dp.tips_of(t); C = dp.vcv(tips, tips)
    np.testing.assert_allclose(dp.model_vcv(C, "OU", 1e-10), C, rtol=1e-8, atol=1e-12)       # OU -> BM as alpha -> 0
    np.testing.assert_allclose(dp.model_vcv(C, "EB", -1e-10), C, rtol=1e-8, atol=1e-12)      # EB -> BM as a -> 0
    np.testing.assert_allclose(dp.model_vcv(C, "lambda", 1.0), C)
    np.testing.assert_allclose(dp.model_vcv(C, "lambda", 0.0), np.diag(np.diag(C)))
    d = np.diag(C); a = 0.7                                                                 # OU stationary-from-root form
    np.testing.assert_allclose(np.diag(dp.model_vcv(C, "OU", a)), -np.expm1(-2 * a * d) / (2 * a))


def test_white_noise_loglik_is_closed_form():
    y = np.array([0.3, -1.2, 2.0, 0.1, 0.5, -0.4, 1.1]); n = len(y)
    ll, s2, z0 = dp.profile_loglik(y, np.eye(n))
    s2_ml = ((y - y.mean()) ** 2).mean()
    assert abs(z0 - y.mean()) < 1e-12 and abs(s2 - s2_ml) < 1e-12
    assert abs(ll - (-n / 2 * (np.log(2 * np.pi) + np.log(s2_ml) + 1))) < 1e-10


def test_fit_models_weights_and_nesting():
    rng = np.random.default_rng(4)
    t = dp.parse_newick(NONULTRA); tips = dp.tips_of(t); C = dp.vcv(tips, tips)
    y = np.linalg.cholesky(C) @ rng.normal(size=len(tips))
    fits = {f["model"]: f for f in dp.fit_models(y, C)}
    assert abs(sum(f["weight"] for f in fits.values()) - 1) < 1e-12
    for m in ("OU", "EB", "lambda"):                     # each nests BM, so its maximum is at least BM's
        assert fits[m]["logLik"] >= fits["BM"]["logLik"] - 1e-8
    assert 0 <= fits["lambda"]["P_lambda_vs_0"] <= 1 and 0 <= fits["lambda"]["P_lambda_vs_1"] <= 1


def test_logmean_is_continuous_and_its_derivatives_are_right():
    u = np.array([0.3, 0.3, -1.0, 2.0]); v = np.array([0.3, 0.3 + 1e-8, 0.5, 1.999])
    L, du, dv = dp._logmean(u, v)
    exact = np.where(np.abs(u - v) > 0, (np.exp(u) - np.exp(v)) / np.where(u == v, 1, u - v), np.exp(u))
    np.testing.assert_allclose(L, exact, rtol=1e-7)
    h = 1e-6
    np.testing.assert_allclose(du, (dp._logmean(u + h, v)[0] - dp._logmean(u - h, v)[0]) / (2 * h), rtol=1e-5)
    np.testing.assert_allclose(dv, (dp._logmean(u, v + h)[0] - dp._logmean(u, v - h)[0]) / (2 * h), rtol=1e-5)


def test_multirate_with_heavy_smoothing_is_single_rate_bm():
    """a very large penalty forces one rate everywhere: the Brownian-motion ML rate."""
    rng = np.random.default_rng(5)
    t = dp.parse_newick(NONULTRA); tips = dp.tips_of(t); C = dp.vcv(tips, tips)
    y = np.linalg.cholesky(C) @ rng.normal(size=len(tips))
    R = dp.multirate_bm(t, tips, y, lam=1e6)
    s2 = dp.profile_loglik(y, C)[1]
    assert R["converged"]
    np.testing.assert_allclose(R["node_rate"], s2, rtol=1e-3)


def test_load_tree_tip_regex_and_duplicates(tmp_path):
    (tmp_path / "t.nwk").write_text("((QH1-S1_Abemus chloropterus:1,QH2-S2_Abemus olivaceus:1):1,"
                                    "(QH3-S3_Naddia alexeyi:1,QH4-S4_Abemus chloropterus:2):1);")
    root, notes = dp.load_tree(str(tmp_path / "t.nwk"), r"_([A-Z][a-z]+ .+)$")
    assert sorted(x.name for x in dp.tips_of(root)) == ["Abemus_chloropterus", "Abemus_olivaceus", "Naddia_alexeyi"]
    assert any("several tips" in n for n in notes)


def test_cli_models_and_rates(tmp_path):
    rng = np.random.default_rng(6)
    tree = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);"
    (tmp_path / "t.nwk").write_text(tree)
    rows = [[s, f"{s}_{k}"] + list(rng.normal(size=2)) for s in "ABCDEFGH" for k in range(2)]
    _write(tmp_path / "c.csv", rows, ["species", "file", "L_star", "b_star"])
    od = tmp_path / "out"
    dp.main(["--tree", str(tmp_path / "t.nwk"), "--traits", f"colour={tmp_path / 'c.csv'}", "--species_col", "species",
             "--analyses", "models", "rates", "--out_dir", str(od)])
    fits = list(csv.DictReader(open(od / "colour_model_fits.csv")))
    assert len(fits) == 2 * 5 and {r["model"] for r in fits} == set(dp.MODELS)
    br = list(csv.DictReader(open(od / "colour_branch_rates.csv")))
    assert len(br) == 2 * 14                                  # 8 tips: 14 branches per variable
    assert (od / "colour__L_star_rate_tree.png").exists() and (od / "rate_fits.csv").exists()
