"""coi_species_tree_v1: midpoint rooting puts the root halfway along the longest path; identical tips get the floor
length; descriptron_phylo matches ids written with underscores for spaces; dna_vs_morphology K2P basics."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "validation"))
import coi_species_tree_v1 as ct            # noqa: E402
import descriptron_phylo as dp              # noqa: E402
import dna_vs_morphology_v1 as dv           # noqa: E402


def test_midpoint_root():
    r = ct.midpoint_root(dp.parse_newick("((A:1,B:1):1,C:10);"))
    d = {t.name: t.depth for t in dp.tips_of(r)}
    assert abs(d["C"] - 6.0) < 1e-9 and abs(d["A"] - 6.0) < 1e-9 and abs(d["B"] - 6.0) < 1e-9


def test_species_tree_floor(tmp_path):
    (tmp_path / "t.nwk").write_text("((s1:0,s2:0):0.1,(s3:0.2,s4:0.1):0.1);")
    import csv
    with open(tmp_path / "sp.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["sequence", "specimen_id", "species", "dna_agrees_with_label", "matrix_named"])
        for s, sp in (("s1", "a"), ("s2", "b"), ("s3", "c"), ("s4", "d")):
            w.writerow([s, s, sp, "", ""])
    ct.main(["--tree", str(tmp_path / "t.nwk"), "--specimens", str(tmp_path / "sp.tsv"), "--out_dir", str(tmp_path / "o")])
    t = dp.parse_newick((tmp_path / "o" / "species_tree.nwk").read_text())
    C = dp.vcv(dp.tips_of(t), dp.tips_of(t))
    assert np.linalg.matrix_rank(C) == 4


def test_key_forms_space_underscore():
    assert "Scan_001_x" in dp._key_forms("Scan 001_x.tif")


def test_k2p_zero_and_transition():
    names, D, _ = dv.k2p_matrix({"a": "ACGTACGTAC", "b": "ACGTACGTAC", "c": "GCGTACGTAC"})
    assert D[0, 1] == 0 and D[0, 2] > 0
