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


# ---------------------------------------------------------------- colour-pattern trees, tanglegram, convergence
# (values against farver, phangorn, ape, vegan, phytools, convevol and windex:
#  validation/validate_phylo_colour_convergence_vs_r.py)
SHARMA_SOME = [  # Sharma, Wu & Dalal (2005) Table 1, pairs 1, 7, 10, 17, 25, 34
    (50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485, 2.0425),
    (50.0, 0.0, 0.0, 50.0, -1.0, 2.0, 2.3669),
    (50.0, 2.49, -0.001, 50.0, -2.49, 0.001, 7.1792),
    (50.0, 2.5, 0.0, 73.0, 25.0, -18.0, 27.1492),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (2.0776, 0.0795, -1.135, 0.9033, -0.0636, -0.5514, 0.9082)]


def test_ciede2000_published_pairs_and_symmetry():
    S = np.array(SHARMA_SOME)
    np.testing.assert_allclose(dp.ciede2000(S[:, :3], S[:, 3:6]), S[:, 6], atol=5e-5)
    np.testing.assert_allclose(dp.ciede2000(S[:, 3:6], S[:, :3]), S[:, 6], atol=5e-5)
    assert np.all(dp.ciede2000(S[:, :3], S[:, :3]) == 0)
    np.testing.assert_allclose(dp.de76(S[:, :3], S[:, 3:6]), np.linalg.norm(S[:, :3] - S[:, 3:6], axis=1))


def test_upgma_recovers_an_ultrametric_tree():
    t = dp.parse_newick("(((A:1,B:1):2,C:3):1,(D:2.5,E:2.5):1.5);"); tips = dp.tips_of(t); names = [x.name for x in tips]
    u = dp.upgma(dp.patristic(tips), names)
    assert dp.tree_keys(u, True) == dp.tree_keys(t, True)
    np.testing.assert_allclose(dp.patristic([{x.name: x for x in dp.tips_of(u)}[s] for s in names]), dp.patristic(tips))


def test_nj_recovers_an_additive_tree():
    t = dp.parse_newick("((A:1,B:2):0.5,(C:0.7,D:1.1):0.9,(E:0.4,F:1.3):0.2);"); tips = dp.tips_of(t)
    names = [x.name for x in tips]
    j = dp.nj(dp.patristic(tips), names)
    assert dp.rf_distance(j, t)["RF"] == 0
    np.testing.assert_allclose(dp.patristic([{x.name: x for x in dp.tips_of(j)}[s] for s in names]), dp.patristic(tips),
                               atol=1e-12)


def test_colour_distance_uses_shared_cells_and_resampled_repeats():
    lab = np.zeros((3, 4, 3)); lab[1, :, 0] = [1, 2, 3, 4]; lab[2, :, 0] = 10
    lab[2, 0] = np.nan                                                   # species 2 lacks cell 0
    Dc = dp.per_cell_distances(lab, "de76")
    D = dp.colour_distance(Dc)
    assert D[0, 1] == 2.5 and D[0, 2] == 10 and abs(D[1, 2] - np.mean([8, 7, 6])) < 1e-12
    Db = dp.colour_distance(Dc, np.array([3, 3, 0]))                     # cell 3 twice, cell 0 once
    assert abs(Db[0, 1] - (4 + 4 + 1) / 3) < 1e-12 and Db[0, 2] == 10    # species 2: only the two copies of cell 3


def test_bootstrap_support_of_well_separated_clusters():
    rng = np.random.default_rng(7)
    centre = {"A": 20.0, "B": 60.0}
    names = [f"{g}{i}" for g in "AB" for i in range(4)]
    lab = np.array([[[centre[n[0]] + rng.normal(0, 1), rng.normal(0, 1), rng.normal(0, 1)] for _ in range(30)] for n in names])
    tree, sup, idx, nfill = dp.bootstrap_support(dp.per_cell_distances(lab), names, "upgma", 50, rng)
    assert idx.shape == (50, 30) and nfill == 0
    full = {s: sup[id(n)] for n, s in dp.clade_sets(tree)}
    assert full[frozenset(names[:4])] == 1.0 and full[frozenset(names[4:])] == 1.0
    # all replicates identical to the full data: every clade is supported by every replicate
    c = dp.support_counts(dp.per_cell_distances(lab), names, "nj", np.tile(np.arange(30), (5, 1)),
                          dp.nj(dp.colour_distance(dp.per_cell_distances(lab)), names))
    c.pop("__filled__"); assert set(c.values()) == {5}


def test_mantel_r_and_ties():
    rng = np.random.default_rng(8)
    P = rng.normal(size=(12, 3)); A = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    B = A + rng.uniform(0, 0.3, A.shape); B = (B + B.T) / 2; np.fill_diagonal(B, 0)
    iu = np.triu_indices(12, 1)
    m = dp.mantel(A, B, "pearson", 199, rng)
    assert abs(m["r"] - np.corrcoef(A[iu], B[iu])[0, 1]) < 1e-12 and m["P"] <= 0.01
    from scipy.stats import spearmanr
    assert abs(dp.mantel(A, B, "spearman", 9, rng)["r"] - spearmanr(A[iu], B[iu])[0]) < 1e-12
    T = A.copy(); T[0, 1] = T[1, 0] = T[0, 2] * (1 + 1e-12); T[0, 2] = T[2, 0] = T[0, 1]  # equal up to noise: tied
    t = dp._tie(np.array([T[0, 1], T[0, 2], T[0, 3]]))
    assert t[0] == t[1] and (t[2] != t[0] or T[0, 3] == T[0, 1])


def test_rf_and_untangle():
    a = dp.parse_newick("(((A:1,B:1):1,(C:1,D:1):1):1,(E:2,(F:1,G:1):1):1);")
    b = dp.parse_newick("((E:2,(G:1,F:1):1):1,((D:1,C:1):1,(B:1,A:1):1):1);")     # same tree, rotated
    assert dp.rf_distance(a, b)["RF"] == 0
    assert dp.crossings([t.name for t in dp.tips_of(a)], [t.name for t in dp.tips_of(b)]) > 0
    assert dp.untangle(a, b) == 0
    c = dp.parse_newick("(((A:1,C:1):1,(B:1,D:1):1):1,(E:2,(F:1,G:1):1):1);")
    assert dp.rf_distance(a, c)["RF"] == 4                                       # AB|CD vs AC|BD: 2 splits each way
    assert dp.untangle(a, c) == 1


CONV_TREE = "((((A:1,B:1):1,C:2):1,(D:1.5,E:1.5):1.5):1,((F:2,G:2):1.5,(H:3,I:3):0.5):0.5);"


def test_convergence_identical_distant_tips():
    """two distant species with the same phenotype: Dtip = 0, so C1 = Ct1 = 1 and C2 = Dmax."""
    t = dp.parse_newick(CONV_TREE); T = dp.ConvTree(t)
    rng = np.random.default_rng(9)
    Y = np.linalg.cholesky(T.C) @ rng.normal(size=(9, 2))
    Y[T.names.index("H")] = Y[T.names.index("A")]
    C, Ct = dp.conv_stats(T, T.all_values(Y), dp._pairs(T, ["A", "H"]))
    assert abs(C[0] - 1) < 1e-12 and abs(Ct[0] - 1) < 1e-12 and C[1] > 0
    # invariance to rotating the trait space
    Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    C2, Ct2 = dp.conv_stats(T, T.all_values(Y @ Q), dp._pairs(T, ["A", "H"]))
    np.testing.assert_allclose(C, C2, atol=1e-12); np.testing.assert_allclose(Ct, Ct2, atol=1e-12)
    r = dp.convergence_test(T, Y, ["A", "H"], {"A": "A", "H": "H"}, 30, rng)
    assert r["n_pairs_C"] == 1 and 0 <= r["P_C"][0] <= 1 and r["P_C"][0] <= 0.2


def test_ancestral_values_match_gls():
    t = dp.parse_newick(CONV_TREE); T = dp.ConvTree(t)
    Y = np.random.default_rng(10).normal(size=(9, 3))
    A = T.all_values(Y)
    inner = [T.nodes[i] for i in T.inner_ix]
    np.testing.assert_allclose(A[T.inner_ix], dp.ancestral_states(Y, T.tips, inner), atol=1e-12)


def test_wheatsheaf_closed_form_and_jackknife():
    t = dp.parse_newick(CONV_TREE); tips = dp.tips_of(t); C = dp.vcv(tips, tips)
    rng = np.random.default_rng(11)
    X = rng.normal(size=(9, 2)); f = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1], bool)
    w = dp.wheatsheaf(X, C, f)
    Xs = X / (X.std(0, ddof=1) / 3)
    P = C / C.max()
    Dm = np.sqrt(((Xs[:, None] - Xs[None]) ** 2).sum(-1)) / (1 - np.log(P + 0.01))
    assert abs(w["w"] - Dm.mean() / Dm[np.ix_(f, f)].mean()) < 1e-12
    # leave-one-out as windex: the species' row and column dropped, the diagonal zero kept
    i = 7; M = Dm.copy(); M[i, :] = np.nan; M[:, i] = np.nan; M[i, i] = 0
    assert abs(w["jackknife"][i] - np.nanmean(M) / np.nanmean(M[np.ix_(f, f)])) < 1e-12
    assert abs(dp.wheatsheaf(X * [3.0, 0.1], C, f)["w"] - w["w"]) < 1e-12        # traits scaled by their SE
    assert dp.quantile7(np.array([1.0, 2.0, np.inf, np.inf]), 0.975) == np.inf


def test_candidate_groups_and_lineages():
    mol = dp.parse_newick(CONV_TREE)
    col = dp.parse_newick("(((A:1,H:1):1,B:2):1,((C:1,D:1):1,(E:1,(F:0.5,(G:0.25,I:0.25):0.25):0.5):1):1);")
    sup = {id(n): 1.0 for n in dp.internal_nodes(col)}
    groups = dp.candidate_groups(col, sup, mol, 0.7, 4)
    mem = [tuple(g["members"]) for g in groups]
    assert ("A", "H") in mem and ("A", "B", "H") in mem
    assert all(len(m) <= 4 for m in mem)
    g = next(g for g in groups if g["members"] == ["A", "B", "H"])
    assert g["lineages"]["A"] == "A+B" and g["lineages"]["H"] == "H"
    assert dp.lineage_sisters(mol, "A+B") == "C"


def test_cli_colour_tree_cophylo_heatmap_convergence(tmp_path):
    rng = np.random.default_rng(12)
    tree = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);"
    (tmp_path / "t.nwk").write_text(tree)
    base = {s: rng.normal([40, 5, 10], [8, 4, 4], size=(6, 3)) for s in "ABCDEFGH"}
    base["G"] = base["A"] + rng.normal(0, 0.3, (6, 3))                      # A and G look alike
    cells = [f"r{1 + k // 3}c{1 + k % 3}" for k in range(6)]
    header = ["filename", "species"] + [f"{c}_{ch}_mean_abs" for c in cells for ch in "Lab"] + ["entropy"]
    rows = []
    for s in "ABCDEFGH":
        for k in range(2):
            v = (base[s] + rng.normal(0, 0.2, (6, 3))).ravel()
            rows.append([f"{s}_{k}.png", s] + list(v) + [rng.random()])
    _write(tmp_path / "c.csv", rows, header)
    (tmp_path / "g.csv").write_text("species,group\nA,AG\nG,AG\n")
    od = tmp_path / "out"
    dp.main(["--tree", str(tmp_path / "t.nwk"), "--traits", f"col={tmp_path / 'c.csv'}", "--species_col", "species",
             "--analyses", "colour_tree", "cophylo", "heatmap", "convergence", "--bootstrap", "20", "--conv_sims", "20",
             "--iterations", "19", "--convergent_groups", str(tmp_path / "g.csv"), "--out_dir", str(od)])
    for f in ("col_colour_distance.csv", "col_colour_tree.nwk", "col_colour_tree.png", "col_colour_tree_support.csv",
              "col_tanglegram.png", "col_tanglegram.pdf", "col_cophylo.csv", "col_colour_heatmap.png",
              "col_convergence.csv", "col_convergent_candidates.csv"):
        assert (od / f).exists(), f
    conv = list(csv.DictReader(open(od / "col_convergence.csv")))
    assert len(conv) == 1 and conv[0]["group"] == "AG" and float(conv[0]["C1"]) > 0.5
    D = list(csv.reader(open(od / "col_colour_distance.csv")))
    assert len(D) == 9 and len(D[0]) == 9
    nwk = (od / "col_colour_tree.nwk").read_text()
    assert dp.rf_distance(dp.parse_newick(nwk), dp.parse_newick(nwk))["RF"] == 0


def test_load_traits_keep_missing(tmp_path):
    _write(tmp_path / "m.csv", [["A", 1, ""], ["A", 3, ""], ["B", 2, 5]], ["species", "x", "y"])
    sp, cols, M, _ = dp.load_traits(str(tmp_path / "m.csv"), "species", None, None)
    assert cols == ["x"]
    sp, cols, M, _ = dp.load_traits(str(tmp_path / "m.csv"), "species", None, None, keep_missing=True)
    assert cols == ["x", "y"] and np.isnan(M[0, 1]) and M[1, 1] == 5 and M[0, 0] == 2


# ---------------------------------------------------------------- palette distance, PCA distance, SURFACE
# (values against emdist, trees.R, cmdscale / ape::pcoa, prcomp, prop.clades, ouch::hansen and surface::runSurface:
#  validation/validate_phylo_palette_surface_vs_r.py)
def test_emd_one_dimensional_closed_form_and_properties():
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(20)
    for _ in range(20):
        m, n = rng.integers(1, 9, 2)
        x, y = rng.normal(0, 20, m), rng.normal(3, 20, n)
        wx, wy = rng.random(m) + 0.01, rng.random(n) + 0.01
        assert abs(dp.emd(wx, x[:, None], wy, y[:, None]) - wasserstein_distance(x, y, wx, wy)) < 1e-10
    A = rng.normal(size=(5, 3)) * 20; w = rng.random(5) + 0.1
    B = rng.normal(size=(4, 3)) * 20; v = rng.random(4) + 0.1
    assert dp.emd(w, A, w, A) < 1e-12
    assert abs(dp.emd(w, A, v, B) - dp.emd(v, B, w, A)) < 1e-10
    assert abs(dp.emd(w * 7, A, v, B) - dp.emd(w, A, v, B)) < 1e-10                 # weights normalised
    c = np.array([[50.0, 0, 0]])                                                   # one colour: weighted mean distance
    assert abs(dp.emd([1.0], c, v, B) - (v / v.sum()) @ np.linalg.norm(B - c, axis=1)) < 1e-10


def test_species_mean_distance_and_specimen_resamples():
    rng = np.random.default_rng(21)
    P = rng.normal(size=(6, 2)); M = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    slot = np.array([0, 0, 1, 2, 2, 2])
    D = dp.species_mean_distance(M, slot, 3)
    assert abs(D[0, 2] - M[np.ix_([0, 1], [3, 4, 5])].mean()) < 1e-12 and D[1, 1] == 0
    row = np.array([1, 1, 2, 5, 5, 3])                                             # a resample: repeats count
    Db = dp.species_mean_distance(M, slot, 3, row)
    assert abs(Db[0, 2] - M[np.ix_([1, 1], [5, 5, 3])].mean()) < 1e-12
    idx = dp.specimen_resamples(slot, 200, rng)
    assert (idx[:, 2] == 2).all()                                                  # one specimen: fixed
    assert set(np.unique(idx[:, :2])) <= {0, 1} and set(np.unique(idx[:, 3:])) <= {3, 4, 5}


def test_pcoa_recovers_a_euclidean_configuration():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(10, 3))
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    Y, ev = dp.pcoa(D)
    assert Y.shape[1] == 3 and ev[3:].max() < 1e-9
    np.testing.assert_allclose(np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1)), D, atol=1e-10)


def test_r_nelder_mead_reproduces_optim():
    """R: optim(c(-1.2, 1), fr) on Rosenbrock's function -> par 1.0002601387256695 1.0005059993037650,
    value 8.8252410967227472e-08, 195 function evaluations (R 4.3.3)."""
    fr = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    r = dp.r_nelder_mead(fr, [-1.2, 1.0])
    np.testing.assert_allclose(r["x"], [1.0002601387256695, 1.0005059993037650], rtol=1e-12)
    assert abs(r["fun"] - 8.8252410967227472e-08) < 1e-18 and r["fncount"] == 195


def test_hansen_one_regime_is_the_stationary_ou_likelihood():
    t = dp.parse_newick(CONV_TREE); T = dp.OUTree(t)
    y = np.random.default_rng(23).normal(size=len(T.names))
    a, s2 = 0.4, 0.8
    dev, th = dp.hansen_deviance(T, y, np.zeros(T.N, int), 1, a, s2)
    V = s2 / (2 * a) * np.exp(-a * (T.T[:, None] + T.T[None, :] - 2 * T.C))
    Vi = np.linalg.inv(V); one = np.ones(len(y)); m = (one @ Vi @ y) / (one @ Vi @ one)
    r = y - m
    assert abs(th[0] - m) < 1e-10
    assert abs(dev - (len(y) * np.log(2 * np.pi) + np.linalg.slogdet(V)[1] + r @ Vi @ r)) < 1e-9


def test_surface_paint_and_a_planted_shift():
    t = dp.parse_newick("(((A:1,B:1):1,(C:1,D:1):1):2,((E:1,F:1):1,(G:1,H:1):1):2);")
    T = dp.OUTree(t)
    cd = next(i for i in range(T.N) if T.tipset[i] == frozenset("CD"))
    reg = T.paint([(0, "a"), (cd, "b")])
    tipreg = {T.names[k]: reg[i] for k, i in enumerate(T.tip_ix)}
    assert tipreg["C"] == tipreg["D"] == "b" and all(tipreg[x] == "a" for x in "ABEFGH")
    rng = np.random.default_rng(24)
    Y = rng.normal(0, 0.05, (8, 2))
    Y[[T.names.index("C"), T.names.index("D")]] += 5.0
    res = dp.run_surface(T, Y)
    assert any(T.tipset[v] == frozenset("CD") for v, _ in res["final"]["shifts"])
    assert res["final"]["aic"] < res["fwd"][0]["aic"]


def test_load_palettes_matches_photo_stems(tmp_path):
    (tmp_path / "pal").mkdir()
    for stem, k in (("p1_A_dorsal", 1), ("p1_AB_dorsal", 2), ("p2_C_dorsal", 3)):
        _write(tmp_path / "pal" / f"recolorize_dE20_{stem}png_{k}_fg_body_palette.csv",
               [[1, 10, 0.6, 1, 1, 1, 0, 0, 0, 30, 5, 5], [2, 5, 0.4, 1, 1, 1, 0, 0, 0, 60, -5, 10]],
               ["colour", "n_pixels", "share", "R", "G", "B", "R01", "G01", "B01", "L", "a", "b"])
    groups = {"p1_A_dorsal.png": "Sp_one", "p1_AB_dorsal.png": "Sp_two", "p2_C_dorsal.png": "Sp_one"}
    pals = dp.load_palettes(str(tmp_path / "pal" / "*_palette.csv"), groups, ["Sp_one", "Sp_two"])
    assert [(p["species"], p["photo"]) for p in pals] == [("Sp_one", "p1_A_dorsal"), ("Sp_one", "p2_C_dorsal"),
                                                          ("Sp_two", "p1_AB_dorsal")]
    np.testing.assert_allclose(pals[0]["w"], [0.6, 0.4])


def test_cli_three_distances_models_on_pcs_and_surface(tmp_path):
    rng = np.random.default_rng(25)
    tree = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);"
    (tmp_path / "t.nwk").write_text(tree)
    cells = [f"r{1 + k // 3}c{1 + k % 3}" for k in range(6)]
    header = ["filename"] + [f"{c}_{ch}_mean_abs" for c in cells for ch in "Lab"] + ["entropy"]
    rows, grows = [], []
    (tmp_path / "pal").mkdir()
    base = {s: rng.normal([40, 5, 10], [8, 4, 4], size=(6, 3)) for s in "ABCDEFGH"}
    base["G"] = base["A"] + rng.normal(0, 0.3, (6, 3))
    for s in "ABCDEFGH":
        for k in range(2 if s in "AG" else 1):
            fn = f"{s}_{k}.png"
            rows.append([fn] + list((base[s] + rng.normal(0, 0.2, (6, 3))).ravel()) + [rng.random()])
            grows.append([fn, s])
            lab = base[s][:3] + rng.normal(0, 0.5, (3, 3))
            _write(tmp_path / "pal" / f"recolorize_dE20_{s}_{k}png_1_fg_col_palette.csv",
                   [[i + 1, 10, w, 0, 0, 0, 0, 0, 0, *lab[i]] for i, w in enumerate([0.5, 0.3, 0.2])],
                   ["colour", "n_pixels", "share", "R", "G", "B", "R01", "G01", "B01", "L", "a", "b"])
    _write(tmp_path / "c.csv", rows, header)
    _write(tmp_path / "g.csv", grows, ["filename", "group_label"])
    od = tmp_path / "out"
    dp.main(["--tree", str(tmp_path / "t.nwk"), "--traits", f"col={tmp_path / 'c.csv'}", "--groups", str(tmp_path / "g.csv"),
             "--palettes", f"col={tmp_path / 'pal' / '*_palette.csv'}", "--colour_distance", "de2000", "palette_emd", "pca",
             "--analyses", "colour_tree", "cophylo", "heatmap", "convergence", "models", "rates", "surface",
             "--model_pcs", "2", "--surface_pcs", "2", "--bootstrap", "10", "--conv_sims", "10", "--iterations", "9",
             "--out_dir", str(od)])
    for d in ("de2000", "palette_emd", "pca"):
        for f in ("colour_distance.csv", "colour_tree.nwk", "tanglegram.png", "colour_heatmap.png", "cophylo.csv"):
            assert (od / f"col__{d}_{f}").exists(), (d, f)
    assert (od / "col__palette_emd_colour_bars.png").exists() and (od / "col__palette_emd_specimen_emd.csv").exists()
    for f in ("col_distance_comparison.csv", "col_model_comparison.csv", "col_surface_path.csv",
              "col_surface_regimes.csv", "col_surface_parameters.csv", "col_chronogram_rates.png",
              "col_chronogram_regimes.png", "col__PC1_rate_tree.png"):
        assert (od / f).exists(), f
    fits = list(csv.DictReader(open(od / "col_model_fits.csv")))
    assert {r["variable"] for r in fits} == {"PC1", "PC2"}
    cmp = {r["model"] for r in csv.DictReader(open(od / "col_model_comparison.csv"))}
    assert cmp == {"BM", "OU1", "EB", "lambda", "SURFACE"}


# ---------------------------------------------------------------- reference scores, Mk, stochastic maps, OUM
def test_adjusted_rand_known_values():
    assert dp.adjusted_rand([1, 1, 2, 2], ["a", "a", "b", "b"]) == 1.0
    # Hubert & Arabie worked value: partitions {1,2,3},{4,5,6} vs {1,2},{3,4},{5,6}: ARI = 0.2424...
    a, b = [1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]
    sij, sa, sb, tot = 2, 6, 3, 15
    exp = sa * sb / tot
    assert abs(dp.adjusted_rand(a, b) - (sij - exp) / (0.5 * (sa + sb) - exp)) < 1e-12


def test_cut_upgma_matches_scipy_average_linkage_and_pam_separates_clusters():
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    rng = np.random.default_rng(31)
    X = np.vstack([rng.normal(c, 0.3, (8, 2)) for c in ((0, 0), (5, 0), (0, 5))])
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    for k in (2, 3, 5):
        f = fcluster(linkage(squareform(D, checks=False), "average"), k, "maxclust")
        assert dp.adjusted_rand(dp.cut_upgma(D, k), f) == 1.0
    truth = np.repeat([1, 2, 3], 8)
    med, cl, obj = dp.pam(D, 3)
    assert dp.adjusted_rand(cl, truth) == 1.0 and len(set(med // 8)) == 3


def test_nn_agreement_ties_and_singletons():
    D = np.array([[0, 1, 1, 5], [1, 0, 2, 5], [1, 2, 0, 5], [5, 5, 5, 0.0]])
    s, n, per = dp.nn_agreement(D, ["x", "x", "y", "z"])
    # A: ties B (x) and C (y) -> 0.5; B -> A (x) 1; C (y, a singleton label) and D (z) not counted
    assert n == 2 and abs(s - 0.75) < 1e-12 and np.isnan(per[2]) and np.isnan(per[3])


def test_mk_loglik_two_state_cherry_closed_form():
    t = dp.parse_newick("(A:0.7,B:1.3);")
    T = dp.MkTree(t)
    q = 0.4
    Q = np.array([[-q, q], [q, -q]])
    P = lambda s, t_: 0.5 + (0.5 if s else -0.5) * np.exp(-2 * q * t_)
    X = np.array([[1.0, 0], [0, 1.0]])
    expect = np.log(0.5 * P(True, 0.7) * P(False, 1.3) + 0.5 * P(False, 0.7) * P(True, 1.3))
    assert abs(dp.mk_loglik(T, X, Q, "equal") - expect) < 1e-12
    assert abs(dp.mk_loglik(T, X, Q, "flat") - expect - np.log(2)) < 1e-12        # ape::ace's form


def test_mk_marginal_equals_brute_force_enumeration():
    from itertools import product
    from scipy.linalg import expm
    t = dp.parse_newick("(((A:1,B:0.5):0.7,C:1.2):0.4,(D:0.9,E:0.3):1.1);")
    T = dp.MkTree(t)
    Q = np.array([[-0.5, 0.3, 0.2], [0.1, -0.4, 0.3], [0.25, 0.05, -0.3]])
    X = np.eye(3)[[0, 1, 1, 2, 0]]
    tipstate = {int(v): int(X[i].argmax()) for i, v in enumerate(T.tip_ix)}
    inner = [int(v) for v in T.inner_ix]
    joint = {}
    for combo in product(range(3), repeat=len(inner)):
        s = dict(zip(inner, combo)); s.update(tipstate)
        p = 1 / 3
        for v in range(1, T.N):
            p *= expm(Q * T.length[v])[s[T.parent[v]], s[v]]
        joint[combo] = p
    tot = sum(joint.values())
    assert abs(np.log(tot) - dp.mk_loglik(T, X, Q)) < 1e-10
    post = dp.mk_marginal(T, X, Q)
    for ii, v in enumerate(inner):
        bf = np.array([sum(p for c, p in joint.items() if c[ii] == s) for s in range(3)]) / tot
        np.testing.assert_allclose(post[v], bf, atol=1e-10)


def test_fit_mk_er_matches_a_grid_and_ard_nests():
    t = dp.parse_newick(CONV_TREE); T = dp.MkTree(t)
    X = np.eye(2)[[0, 0, 1, 0, 1, 1, 1, 0, 1]]
    f = dp.fit_mk(T, X, "ER")
    grid = np.exp(np.linspace(np.log(1e-3), np.log(10), 4001))
    best = max(dp.mk_loglik(T, X, np.array([[-q, q], [q, -q]])) for q in grid)
    assert f["logLik"] >= best - 1e-9
    assert dp.fit_mk(T, X, "ARD", starts=[f["rates"].repeat(2)])["logLik"] >= f["logLik"] - 1e-9


def test_simmaps_are_consistent_with_node_states_and_branch_lengths():
    t = dp.parse_newick(CONV_TREE); T = dp.MkTree(t)
    X = np.eye(3)[[0, 0, 1, 0, 1, 2, 2, 0, 1]]
    Q = dp.fit_mk(T, X, "ER")["Q"]
    maps = dp.simmaps(T, X, Q, 50, np.random.default_rng(32))
    for m in maps:
        assert (m["state"][T.tip_ix] == X.argmax(1)).all()
        for v in range(1, T.N):
            sg = m["segments"][v]
            assert abs(sum(d for _, d in sg) - T.length[v]) < 1e-9
            assert sg[0][0] == m["state"][T.parent[v]] and sg[-1][0] == m["state"][v]
            assert all(a_ != b_ for (a_, _), (b_, _) in zip(sg[:-1], sg[1:]))
        C, tm = dp.simmap_counts(T, m, 3)
        o = dp.simmap_origins(T, m, 3)
        assert abs(tm.sum() - T.length.sum()) < 1e-9
        for s in range(3):                                   # origins: at least 1, at most the species of that state
            assert 1 <= o[s] <= int(X[:, s].sum()) and o[s] <= C[:, s].sum() + (m["state"][0] == s)


def test_bms_and_oum_with_one_regime_reduce_to_bm_and_ou1():
    t = dp.parse_newick(CONV_TREE); T = dp.OUTree(t)
    y = np.random.default_rng(33).normal(size=len(T.names))
    b = dp.bms_fit(T, y, np.zeros(T.N, int), 1)
    assert abs(b["loglik"] - dp.profile_loglik(y, T.C)[0]) < 1e-6
    o = dp.oum_fit(T, y[:, None], ["a"] * T.N)
    f = dp.hansen_fit(T, y, ["a"] * T.N, np.sqrt(2 / T.depth), np.sqrt(np.var(y, ddof=1) / T.depth))
    assert o["lnL"] >= f["loglik"] - 1e-12


def test_cli_reference_scores_mk_maps_and_oum(tmp_path):
    rng = np.random.default_rng(34)
    tree = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);"
    (tmp_path / "t.nwk").write_text(tree)
    cells = [f"r{1 + k // 3}c{1 + k % 3}" for k in range(6)]
    header = ["filename"] + [f"{c}_{ch}_mean_abs" for c in cells for ch in "Lab"]
    sp = list("ABCDEFGHIJ")                                   # I and J are not on the tree
    typ = dict(zip(sp, ["m1", "m1", "m2", "m2", "m1", "m3", "m2", "m3", "m1", "m2"]))
    centre = {"m1": [30, 5, 5], "m2": [60, 10, 30], "m3": [45, -5, -10]}
    rows, grows, refrows = [], [], []
    for s in sp:
        fn = f"{s}_0.png"
        rows.append([fn] + list((np.array(centre[typ[s]]) + rng.normal(0, 3, (6, 3))).ravel()))
        grows.append([fn, s]); refrows.append([s, f"G_{typ[s]}", typ[s]])
    _write(tmp_path / "c.csv", rows, header)
    _write(tmp_path / "g.csv", grows, ["filename", "group_label"])
    _write(tmp_path / "ref.csv", refrows, ["species", "group", "mimicry_type"])
    od = tmp_path / "out"
    dp.main(["--tree", str(tmp_path / "t.nwk"), "--traits", f"col={tmp_path / 'c.csv'}", "--groups", str(tmp_path / "g.csv"),
             "--colour_distance", "de2000", "pca", "--reference_groups", str(tmp_path / "ref.csv"),
             "--reference_columns", "group,mimicry_type", "--discrete_trait", f"{tmp_path / 'ref.csv'}:mimicry_type",
             "--analyses", "reference_scores", "oum", "--mk_models", "ER", "SYM", "--simmaps", "20", "--oum_maps", "5",
             "--surface_pcs", "2", "--iterations", "49", "--out_dir", str(od)])
    sc = list(csv.DictReader(open(od / "col__de2000_reference_scores.csv")))
    assert {(r["species_set"], r["grouping"]) for r in sc} == {(s, g) for s in ("all", "tree") for g in ("group", "mimicry_type")}
    assert {r["n_species"] for r in sc if r["species_set"] == "all"} == {"10"}
    assert {r["n_species"] for r in sc if r["species_set"] == "tree"} == {"8"}
    assert float(next(r for r in sc if r["species_set"] == "all" and r["grouping"] == "group")["NN_agreement"]) == 1.0
    assert (od / "reference_scores_summary.csv").exists() and (od / "col__pca_reference_nearest.csv").exists()
    for f in ("mk_mimicry_type_fits.csv", "mk_mimicry_type_ancestral_states.csv", "mk_mimicry_type_origins.csv",
              "mk_mimicry_type_simmap_transitions.csv", "mk_mimicry_type_simmap_tree.png", "col_oum_model_comparison.csv",
              "col_oum_parameters.csv", "col_oum_maps.csv", "col_oum_optima.png"):
        assert (od / f).exists(), f
    models = [r["model"] for r in csv.DictReader(open(od / "col_oum_model_comparison.csv"))]
    assert {"BM", "OU1", "OUM (MAP painting)", "BMS (MAP painting)", "SURFACE"} <= set(models)
    assert len(list(csv.DictReader(open(od / "col_oum_maps.csv")))) == 5


def test_white_noise_loglik_by_group_is_the_anova_likelihood():
    from scipy.stats import norm
    y = np.array([1.0, 1.5, 0.7, 4.0, 4.4, 3.1])
    g = ["a", "a", "a", "b", "b", "b"]
    mu = np.array([y[:3].mean()] * 3 + [y[3:].mean()] * 3)
    s = np.sqrt(((y - mu) ** 2).mean())
    assert abs(dp.white_noise_loglik(y, g) - norm.logpdf(y, mu, s).sum()) < 1e-12
    assert abs(dp.white_noise_loglik(y) - norm.logpdf(y, y.mean(), y.std()).sum()) < 1e-12


# ---------------------------------------------------------------------------- ancestral colour pattern
def test_bm_ancestral_is_gls_with_fastanc_variances():
    t = dp.parse_newick("((A:1,B:1):1,C:2);"); tips = dp.tips_of(t); nodes = dp.internal_nodes(t)
    y = np.array([[1.0, 10.0], [3.0, 14.0], [8.0, 11.0]])
    R = dp.bm_ancestral(tips, nodes, y)
    np.testing.assert_allclose(R["est"], dp.ancestral_states(y, tips, nodes), atol=1e-12)
    C = dp.vcv(tips, tips); Ci = np.linalg.inv(C); one = np.ones(3)
    mu = (one @ Ci @ y) / (one @ Ci @ one)
    s2 = np.einsum("ij,ik,kj->j", y - mu, Ci, y - mu) / 2               # REML: mean squared contrast
    np.testing.assert_allclose(R["sigsq"], s2, rtol=1e-12)
    # contrasts with the node as root: root (AB 1 + 0.5, C 2) -> 6/7; node AB (A 1, B 1, C 3) -> 3/7
    np.testing.assert_allclose(R["var"], np.outer([6 / 7, 3 / 7], s2), rtol=1e-12)


def test_bm_ancestral_missing_tips_is_the_pruned_tree_and_interpolates_removed_nodes():
    rng = np.random.default_rng(5)
    t = dp.parse_newick("(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);")
    tips = dp.tips_of(t); nodes = dp.internal_nodes(t)
    y = rng.normal(size=(8, 1)); y_m = y.copy(); y_m[[2, 3], 0] = np.nan          # C and D lack the value
    R = dp.bm_ancestral(tips, nodes, y_m)
    tp = dp.prune(dp._copy_tree(t), set("ABEFGH")); tt = dp.tips_of(tp); tn = dp.internal_nodes(tp)
    Rp = dp.bm_ancestral(tt, tn, y[[0, 1, 4, 5, 6, 7]])
    key = lambda n: frozenset(x.name for x in dp.tips_of(n))
    full = {key(n): i for i, n in enumerate(nodes)}
    for j, n in enumerate(tn):                                    # the same node = the MRCA of the same species
        i = min((len(k), v) for k, v in full.items() if key(n) <= k)[1]
        assert abs(R["est"][i, 0] - Rp["est"][j, 0]) < 1e-12 and abs(R["var"][i, 0] - Rp["var"][j, 0]) < 1e-12
    cd = full[frozenset("CD")]; abcd = full[frozenset("ABCD")]; ab = full[frozenset("AB")]
    assert abs(R["est"][cd, 0] - R["est"][abcd, 0]) < 1e-12        # all its species missing: the attachment value
    assert R["var"][cd, 0] > R["var"][abcd, 0]
    assert np.isfinite(R["est"]).all() and R["n_obs"][0] == 6
    assert np.isnan(dp.bm_ancestral(tips, nodes, np.where(np.arange(8)[:, None] < 2, y, np.nan))["est"]).all()
    # ABCD, removed by pruning, lies on the pruned branch root-AB: its estimate is the path-length-weighted mean
    rt = full[frozenset("ABCDEFGH")]                              # root -1- ABCD -1- AB once C, D are gone
    assert abs(R["est"][abcd, 0] - 0.5 * (R["est"][rt, 0] + R["est"][ab, 0])) < 1e-12


def test_k_by_column_is_kmult_of_each_column_and_group():
    rng = np.random.default_rng(7)
    t = dp.parse_newick(TREE); tips = dp.tips_of(t); C = dp.vcv(tips, tips)
    Y = rng.normal(size=(4, 6))
    K, Kg = dp.k_by_column(Y, C, [np.arange(3), np.arange(3, 6)])
    for j in range(6):
        assert abs(K[j] - dp.kmult(Y[:, [j]], C)) < 1e-12
    assert abs(Kg[0] - dp.kmult(Y[:, :3], C)) < 1e-12 and abs(Kg[1] - dp.kmult(Y[:, 3:], C)) < 1e-12


def test_bh_adjust_matches_p_adjust():
    np.testing.assert_allclose(dp.bh_adjust(np.array([0.01, 0.04, 0.03, 0.2, np.nan]))[:4], [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert np.isnan(dp.bh_adjust(np.array([0.5, np.nan]))[1])


def test_lab_to_rgb_round_trips_srgb_and_pattern_image_fills_cells():
    g = np.linspace(0, 1, 5); rgb = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    lab = np.c_[116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])]
    assert np.abs(dp.lab_to_rgb(lab) - rgb).max() < 1e-5
    geom = {"outline": np.array([[0, 0], [2, 0], [2, 1], [0, 1.0]]), "x_lines": [0, 1, 2], "y_lines": [0, 1],
            "cell_rc": {"r0c0": (0, 0), "r0c1": (0, 1)}}
    im = dp.pattern_image(geom, ["r0c0", "r0c1"], np.array([[1, 0, 0], [0, 0, 1.0]]), np.array([True, False]), width=40)
    h, w = im.shape[:2]
    assert (im[h // 2, 10, :3] == [1, 0, 0]).all() and np.allclose(im[h // 2, 30, :3], 0.85)   # not drawn: grey
    assert im[0, 0, 3] == 0 and im[h // 2, w // 2, 3] == 1
    assert dp.pattern_image(geom, ["r0c0", "r0c1"], np.zeros((2, 3)), width=40, rotate=90).shape[:2] == (w, h)


def test_simmap_origin_branches_walks_to_the_last_change():
    t = dp.parse_newick("((A:1,B:1):1,C:2);"); T = dp.MkTree(t)
    ix = {T.nodes[i].name: i for i in T.tip_ix}; ab = T.parent[ix["A"]]
    segs = [[] for _ in range(T.N)]
    segs[ab] = [(0, 0.5), (1, 0.5)]; segs[ix["A"]] = [(1, 1.0)]; segs[ix["B"]] = [(1, 0.4), (0, 0.2), (1, 0.4)]
    segs[ix["C"]] = [(0, 2.0)]
    st = np.zeros(T.N, int); st[ab] = 1; st[ix["A"]] = 1; st[ix["B"]] = 1
    ob = dp.simmap_origin_branches(T, [{"state": st, "segments": segs}])
    got = {T.nodes[ti].name: ob[k] for k, ti in enumerate(T.tip_ix)}
    assert got == {"A": {ab: 1}, "B": {ix["B"]: 1}, "C": {0: 1}}


def test_cli_ancestral_pattern_with_cell_geometry(tmp_path):
    import json
    rng = np.random.default_rng(41)
    tree = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);"
    (tmp_path / "t.nwk").write_text(tree)
    cells = [f"r{r}c{c}" for r in range(3) for c in range(2)]
    feats = ["dL_rel", "L_mean_abs", "a_mean_abs", "b_mean_abs"]
    header = ["filename"] + [f"{c}_{f}" for c in cells for f in feats]
    typ = dict(zip("ABCDEFGH", ["m1", "m1", "m2", "m2", "m1", "m2", "m2", "m1"]))
    rows, grows, refrows, spec = [], [], [], {}
    for s in "ABCDEFGH":
        for k in range(2 if s == "A" else 1):
            fn = f"{s}_{k}.png_{k}"
            rows.append([fn] + [v for c in cells for v in [0.0, *rng.normal([40, 10, 20], 5)]])
            grows.append([f"{s}_{k}.png", s])
            spec[fn] = {"missing": ["r2c1"] if s in "CD" or (s == "A" and k == 0) else []}
        refrows.append([s, typ[s]])
    _write(tmp_path / "c.csv", rows, header)
    _write(tmp_path / "g.csv", grows, ["filename", "group_label"])
    _write(tmp_path / "ref.csv", refrows, ["species", "mimicry_type"])
    geom = {"consensus_outline": [[0, 0], [2, 0], [2, 3], [0, 3]], "x_lines": [0, 1, 2], "y_lines": [0, 1, 2, 3],
            "cells": [{"cell": c, "row": int(c[1]), "col": int(c[3]), "overlap": 1.0} for c in cells],
            "specimens": spec}
    (tmp_path / "geom.json").write_text(json.dumps(geom))
    od = tmp_path / "out"
    dp.main(["--tree", str(tmp_path / "t.nwk"), "--traits", f"col={tmp_path / 'c.csv'}", "--groups", str(tmp_path / "g.csv"),
             "--cell_geometry", f"col={tmp_path / 'geom.json'}", "--discrete_trait", f"{tmp_path / 'ref.csv'}:mimicry_type",
             "--state_codes", "m1=X:#123456", "--analyses", "ancestral_pattern", "--mk_models", "ER", "--simmaps", "10",
             "--iterations", "9", "--pattern_width", "30", "--out_dir", str(od)])
    st = list(csv.DictReader(open(od / "col_ancestral_pattern_states.csv")))
    assert len(st) == (7 + 8) * 6 * 3                                    # 7 nodes + 8 species, 6 cells, L a b
    a_ = [r for r in st if r["node"] == "A" and r["cell"] == "r2c1" and r["channel"] == "L"][0]
    assert abs(float(a_["estimate"]) - float(rows[1][1 + 5 * 4 + 1])) < 1e-6        # A's mean = its specimen that has the cell
    assert {r["n_species_with_cell"] for r in st if r["cell"] == "r2c1"} == {"6"}
    cd = [r for r in st if r["cell"] == "r2c1" and r["is_tip"] == "0" and r["descendants_with_cell"] == "0"]
    assert len(cd) == 3                                                  # the CD node: no species of it has the cell
    for f in ("tree.png", "nodes.png", "cell_maps.png", "convergence.png", "nodes.csv", "cells.csv", "origins.csv"):
        assert (od / f"col_ancestral_pattern_{f}").exists(), f
    org = list(csv.DictReader(open(od / "col_ancestral_pattern_origins.csv")))
    assert sum(int(r["n_descendant_species"]) for r in org) == 8


# ---------------------------------------------------------------------------- rate of pattern change, parts, discrete tree
def test_branch_pattern_changes_is_mean_ciede2000_over_compared_cells_per_unit_length():
    rng = np.random.default_rng(3)
    t = dp.parse_newick("((A:1,B:3):2,C:4);"); tips = dp.tips_of(t); nodes = dp.internal_nodes(t)
    lab = rng.normal([50, 10, 20], 8, size=(3, 4, 3)); lab[2, 3] = np.nan          # C lacks cell 3
    R = dp.bm_ancestral(tips, nodes, lab.reshape(3, -1), min_obs=2)           # cell 3: A and B only
    est = R["est"].reshape(len(nodes), 4, 3)
    desc = np.array([np.isfinite(lab[[tips.index(x) for x in dp.tips_of(nd)], :, 0]).any(0) for nd in nodes])
    br = dp.branch_pattern_changes(t, tips, nodes, est, lab, desc)
    by = {dp.branch_key(b["child"]): b for b in br}
    assert len(br) == 4 and set(by) == {frozenset("A"), frozenset("B"), frozenset("C"), frozenset("AB")}
    ab = nodes[1] if dp.branch_key(nodes[1]) == frozenset("AB") else nodes[0]
    iab, iroot = nodes.index(ab), nodes.index(t)
    b = by[frozenset("B")]
    want = dp.ciede2000(est[iab], lab[1])
    assert abs(b["mean"] - want.mean()) < 1e-12 and abs(b["rate"] - want.mean() / 3) < 1e-12 and b["n_cells"] == 4
    c = by[frozenset("C")]                                       # only the 3 cells C has
    assert c["n_cells"] == 3 and abs(c["mean"] - dp.ciede2000(est[iroot, :3], lab[2, :3]).mean()) < 1e-12
    assert np.isnan(c["de"][3]) and abs(c["rate"] - c["mean"] / 4) < 1e-12
    n_ = by[frozenset("AB")]
    assert abs(n_["mean"] - dp.ciede2000(est[iroot], est[iab]).mean()) < 1e-12 and abs(n_["length"] - 2) < 1e-12
    # identical patterns everywhere: no change, rate 0
    same = np.repeat(lab[:1], 3, axis=0)
    R2 = dp.bm_ancestral(tips, nodes, same.reshape(3, -1))
    br2 = dp.branch_pattern_changes(t, tips, nodes, R2["est"].reshape(len(nodes), 4, 3), same, np.ones((2, 4), bool))
    assert max(b["mean"] for b in br2) < 1e-9


def test_place_node_pictures_never_overlaps():
    t = dp.parse_newick("(((A:1,B:1):0.1,C:1.1):0.05,((D:1,E:1):0.02,F:1.02):0.1);")
    x, y = dp._layout(t)
    pl = dp._place_node_pictures(t, x, y, 0.5, 0.7, 6)
    boxes = [(cx, cy - 0.06, 0.25 * sc, (0.7 * sc + 0.12) / 2) for cx, cy, sc in pl.values()]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes[i], boxes[j]
            assert abs(x1 - x2) >= w1 + w2 - 1e-9 or abs(y1 - y2) >= h1 + h2 - 1e-9


def test_make_pd_keeps_psd_and_repairs_the_diagonal_preserving_rest():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 3)); S = A @ A.T + np.eye(6) * 0.5
    np.testing.assert_allclose(dp.make_pd(S), S, atol=1e-10)
    B = A @ A.T; np.fill_diagonal(B, np.diag(B).mean() * 0.3)               # not positive semidefinite
    assert np.linalg.eigvalsh(B).min() < 0
    P = dp.make_pd(B)
    assert np.linalg.eigvalsh(P).min() > 0 and np.allclose(P, P.T)


def test_multi_evol_rates_closed_form_and_planted_difference():
    rng = np.random.default_rng(11)
    t = dp.parse_newick(TREE8); tips = dp.tips_of(t); C = dp.vcv(tips, tips)
    L = np.linalg.cholesky(C)
    Y1 = L @ rng.normal(size=(8, 6)); Y2 = 2.0 * Y1[:, :4]                    # set 2: 4x the rate of the same changes
    T = dp.multi_evol_rates([Y1[:, :4], Y2], C, 99, rng)
    Ci = np.linalg.inv(C); w = Ci.sum(1)
    Xc = Y1[:, :4] - (w @ Y1[:, :4]) / w.sum()
    s1 = np.trace(Xc.T @ Ci @ Xc) / (8 * 4)
    assert abs(T["rates"][0] - s1) < 1e-12 and abs(T["rates"][1] / T["rates"][0] - 4) < 1e-12
    assert abs(T["ratio"] - 4) < 1e-12 and T["pairs"] == [(0, 1)] and 0 < T["P"] <= 0.05
    T2 = dp.multi_evol_rates([Y1[:, :3], Y1[:, 3:]], C, 99, rng)               # same process: no difference
    assert T2["P"] > 0.05 and abs(T2["global_rate"] - dp._subset_rates(Y1, Ci, [np.arange(6)])[0]) < 1e-12


def test_combine_distances_median_normalised_weighted_sum_and_parse():
    D1 = np.array([[0, 2, 4], [2, 0, 6], [4, 6, 0.0]]); D2 = np.array([[0, 1, 1], [1, 0, 3], [1, 3, 0.0]])
    nm, D = dp.combine_distances([(["a", "b", "c"], D1), (["c", "b", "a", "x"][:3][::-1], D2)], [3.0, 1.0])
    assert nm == ["a", "b", "c"]
    np.testing.assert_allclose(D, (3 * D1 / 4 + D2 / 1) / 4)
    assert dp.combine_distances([(["a", "b", "c"], D1)], [2.0])[1] is not None
    np.testing.assert_allclose(dp.combine_distances([(["a", "b", "c"], D1)], [2.0])[1], D1)
    assert dp.parse_combine_sets(["body=soma:2,abdomen"]) == [("body", [("soma", 2.0), ("abdomen", 1.0)])]


def test_fitch_counts_and_state_clades():
    t = dp.parse_newick("(((A:1,B:1):1,(C:1,D:1):1):1,(E:2,F:2):1);")
    st = dict(zip("ABCDEF", ["x", "x", "y", "y", "x", "z"]))
    score, F = dp.fitch(t, st)
    assert score == 2
    assert dp.state_clades(t, st, "x") == [["A", "B"], ["E"]] and dp.state_clades(t, st, "y") == [["C", "D"]]
    ab = [n for n in dp.internal_nodes(t) if dp.branch_key(n) == frozenset("AB")][0]
    assert F[id(ab)] == {"x"}


TREE8 = "(((A:1,B:1):1,(C:0.5,D:0.5):1.5):1,((E:2,F:2):0.5,(G:1,H:1):1.5):0.5);"


def _flower_inputs(tmp_path, rng):
    """two parts with non-beetle names (petal, sepal), 8 species, cell geometry, a COCO with the parts' masks."""
    import json
    (tmp_path / "t.nwk").write_text(TREE8)
    grows, out = [], {}
    for part, (nr, nc), base in (("petal", (3, 2), [60, 40, 10]), ("sepal", (2, 2), [45, -20, 30])):
        cells = [f"r{r}c{c}" for r in range(nr) for c in range(nc)]
        feats = ["dL_rel", "L_mean_abs", "a_mean_abs", "b_mean_abs"]
        header = ["filename"] + [f"{c}_{f}" for c in cells for f in feats]
        rows = []
        for s in "ABCDEFGH":
            fn = f"{s}_0.png_0"
            rows.append([fn] + [v for c in cells for v in [0.0, *rng.normal(base, 6 if part == "petal" else 2)]])
            if part == "petal":
                grows.append([f"{s}_0.png", s])
        _write(tmp_path / f"{part}.csv", rows, header)
        geom = {"consensus_outline": [[0, 0], [nc, 0], [nc, nr], [0, nr]], "x_lines": list(range(nc + 1)),
                "y_lines": list(range(nr + 1)), "specimens": {f"{s}_0.png_0": {"missing": []} for s in "ABCDEFGH"},
                "cells": [{"cell": c, "row": int(c[1]), "col": int(c[3]), "overlap": 1.0} for c in cells]}
        (tmp_path / f"{part}.json").write_text(json.dumps(geom))
        out[part] = (tmp_path / f"{part}.csv", tmp_path / f"{part}.json")
    _write(tmp_path / "g.csv", grows, ["filename", "group_label"])
    coco = {"images": [{"id": 1, "file_name": "f.png", "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "petal"}, {"id": 2, "name": "sepal"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[40, 10, 60, 10, 60, 40, 40, 40]]},
                            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [[10, 60, 30, 60, 30, 80, 10, 80]]}]}
    (tmp_path / "parts.json").write_text(json.dumps(coco))
    return out


def test_cli_rate_coloured_trees_part_sets_and_body_map_with_any_part_names(tmp_path):
    rng = np.random.default_rng(21)
    P = _flower_inputs(tmp_path, rng)
    od = tmp_path / "out"
    argv = ["--tree", str(tmp_path / "t.nwk"), "--groups", str(tmp_path / "g.csv"), "--analyses", "ancestral_pattern",
            "--anc_branch_colour", "rate_de00", "bm_rate", "--model_pcs", "2", "--iterations", "9",
            "--part_sets", "petal", "sepal", "--part_test_sims", "49", "--body_map_layout", "coco",
            "--body_map_coco", str(tmp_path / "parts.json"), "--pattern_width", "30", "--out_dir", str(od)]
    for part, (c, j) in P.items():
        argv += ["--traits", f"{part}={c}", "--cell_geometry", f"{part}={j}"]
    dp.main(argv)
    for f in ("petal_ancestral_pattern_tree.png", "petal_ancestral_pattern_tree_bm_rate.png",
              "sepal_ancestral_pattern_branches.csv", "body_map_part_rates.png", "part_rates.csv", "part_rate_pairs.csv",
              "part_branch_rate_correlation.csv"):
        assert (od / f).exists(), f
    br = list(csv.DictReader(open(od / "petal_ancestral_pattern_branches.csv")))
    assert len(br) == 14 and all(abs(float(r["dE00_per_Myr"]) * float(r["length"]) - float(r["dE00_mean"])) < 1e-9
                                 for r in br)
    assert all(r["bm_rate"] for r in br) and all(len(r["top_changing_cells"].split(";")) == 5 for r in br)
    pr = {r["part"]: r for r in csv.DictReader(open(od / "part_rates.csv"))}
    assert set(pr) == {"petal", "sepal"}
    assert float(pr["petal"]["sigma2_mult_Lab"]) > float(pr["sepal"]["sigma2_mult_Lab"])   # petals vary 3x more
    rep = (od / "phylo_report.md").read_text()
    assert "Rates of colour-pattern change between body parts (petal, sepal)" in rep


def test_coco_part_offsets_and_layouts():
    import json, tempfile
    coco = {"images": [{"id": 1, "file_name": "a", "width": 50, "height": 50}, {"id": 2, "file_name": "b", "width": 50, "height": 50}],
            "categories": [{"id": 1, "name": "petal"}, {"id": 2, "name": "sepal"}],
            "annotations": [{"image_id": i, "category_id": 1, "segmentation": [[0, 0, 10 * i, 0, 10 * i, 10 * i, 0, 10 * i]]}
                            for i in (1, 2)] +
                           [{"image_id": i, "category_id": 2, "segmentation": [[20 * i, 0, 30 * i, 0, 30 * i, 10 * i, 20 * i, 10 * i]]}
                            for i in (1, 2)]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(coco, fh)
    off = dp.coco_part_offsets(fh.name, ["petal", "sepal"])
    np.testing.assert_allclose(off["petal"] + off["sepal"], 0, atol=1e-9)       # equal areas: centroid in between
    assert off["sepal"][0] > 0 and abs(off["sepal"][1]) < 1e-9
    assert dp.coco_part_offsets(fh.name, ["petal", "stamen"]) is None
    g = {"outline": np.array([[0, 0], [1, 0], [1, 2], [0, 2.0]]), "x_lines": [0, 1], "y_lines": [0, 2]}
    st = dp._part_layout([g, g], "stack_y")
    assert abs(st[1][:, 1].min() - st[0][:, 1].max()) < 1e-9
    sx = dp._part_layout([g, g], "stack_x")
    assert abs(sx[1][:, 0].min() - sx[0][:, 0].max()) < 1e-9


def test_cli_discrete_tree_without_trait_tables(tmp_path):
    (tmp_path / "t.nwk").write_text(TREE8)
    _write(tmp_path / "s.csv", [[s, t_, f"g{i}"] for i, (s, t_) in enumerate(zip("ABCDEFGH", "xxyyxzzz"))],
           ["species", "kind", "group"])
    od = tmp_path / "o"
    dp.main(["--tree", str(tmp_path / "t.nwk"), "--discrete_trait", f"{tmp_path / 's.csv'}:kind", "--analyses",
             "discrete_tree", "--discrete_label_column", "group", "--state_codes", "z=Z:#123456", "--out_dir", str(od)])
    assert (od / "kind_on_tree.png").exists()
    rep = (od / "phylo_report.md").read_text()
    assert "Fitch parsimony: at least" in rep and "Z = z: 3 species in 2 separate clade(s)" in rep
