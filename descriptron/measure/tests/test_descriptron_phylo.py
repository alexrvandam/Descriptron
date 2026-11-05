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
