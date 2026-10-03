"""descriptron_character_signal_v1: the vectorised Blomberg's K equals descriptron_phylo.kmult (validated against
geomorph); a character that tracks the tree has signal and one that does not has none; the table keeps the top N
per structure (all of them when fewer); every pipeline step is a valid --only_steps name."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_character_signal_v1 as cs      # noqa: E402
import descriptron_phylo as dp                    # noqa: E402

TREE = "(((a:1,b:1):2,(c:1,d:1):2):3,((e:1,f:1):2,(g:1,h:1):2):3);"


def test_k_matches_kmult():
    t = dp.parse_newick(TREE); tp = dp.tips_of(t); C = dp.vcv(tp, tp)
    y = np.random.default_rng(3).normal(size=len(tp))
    K, P = cs.blomberg_k_perm(y, C, 99, np.random.default_rng(0))
    assert abs(K - dp.kmult(y[:, None], C)) < 1e-12 and 0 < P <= 1


def test_table_and_signal(tmp_path):
    rng = np.random.default_rng(0); sp = list("abcdefgh"); clade = {s: (0 if s in "abcd" else 5) for s in sp}
    rows, fd = [], []
    for st, nchar in (("wing", 30), ("head", 3)):
        for j in range(nchar):
            fid = f"{st}.c{j}"; fd.append({"feature_id": fid, "category": st, "label": f"{st} char {j}", "family": "ratio", "tier": "key"})
            for s in sp:
                for k in range(3):
                    base = clade[s] if j == 0 else rng.normal(0, 3)       # c0 follows the tree's two clades
                    rows.append({"species": s, "specimen_id": f"{s}_{k}", "feature_id": fid, "value": base + rng.normal(0, 0.3)})
    md = tmp_path / "m"; md.mkdir()
    pd.DataFrame(rows).to_csv(md / "specimen_matrix_long.csv", index=False)
    pd.DataFrame(fd).to_csv(md / "feature_dictionary.tsv", sep="\t", index=False)
    (tmp_path / "t.nwk").write_text(TREE)
    cs.main(["--matrix_dir", str(md), "--tree", str(tmp_path / "t.nwk"), "--top", "25", "--iterations", "999",
             "--figure", "--out_dir", str(tmp_path / "o")])
    A = pd.read_csv(tmp_path / "o" / "character_signal_all.tsv", sep="\t").set_index("feature_id")
    assert A.loc["wing.c0", "K_P"] < 0.05 and A.loc["wing.c0", "K"] > 1
    T = pd.read_csv(tmp_path / "o" / "character_signal_top.tsv", sep="\t")
    assert (T.structure == "wing").sum() == 25 and (T.structure == "head").sum() == 3
    assert (tmp_path / "o" / "character_signal_volcano.png").exists()


def test_robustness_ranking(tmp_path):
    """with a robustness table the order within a structure follows hold-out naming, not eta^2"""
    test_table_and_signal(tmp_path)                                    # builds tmp_path/m and the tree
    A = pd.read_csv(tmp_path / "o" / "character_signal_all.tsv", sep="\t")
    rob = pd.DataFrame({"feature_id": A.feature_id, "top1": np.linspace(0.1, 0.9, len(A)), "top1_ci_low": 0.0,
                        "top1_ci_high": 1.0, "chance_top1": 0.125, "times_chance": 1.0, "novelty_auc": 0.6, "species_tested": 8})
    rob.to_csv(tmp_path / "rob.tsv", sep="\t", index=False)
    cs.main(["--matrix_dir", str(tmp_path / "m"), "--tree", str(tmp_path / "t.nwk"), "--robustness", str(tmp_path / "rob.tsv"),
             "--iterations", "99", "--out_dir", str(tmp_path / "o2")])
    T = pd.read_csv(tmp_path / "o2" / "character_signal_top.tsv", sep="\t")
    for _, g in T.groupby("structure"):
        assert list(g.top1) == sorted(g.top1, reverse=True)
    assert "named %" in (tmp_path / "o2" / "character_signal_top.md").read_text()
    assert (tmp_path / "o2" / "character_phylo_signal.png").exists()       # the companion figure, with tree + robustness


def test_every_step_is_a_valid_name():
    import run_full_pipeline_v2 as pipe
    assert set(pipe.ALL_STEPS) <= set(pipe.STEP_NAMES), set(pipe.ALL_STEPS) - set(pipe.STEP_NAMES)
    assert pipe.V2_WORKFLOW.index("char_signal") == pipe.V2_WORKFLOW.index("key_matrix") + 1
