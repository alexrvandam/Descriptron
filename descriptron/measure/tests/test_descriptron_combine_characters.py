"""descriptron_combine_characters_v1: blocks of per-specimen characters side by side (shape PCs + a table),
leave-one-out species ID for every combination, nested best combination, two-block PLS. Synthetic data, CPU only."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_combine_characters_v1 as cc          # noqa: E402


def _write_blocks(tmp_path, n_per=5, groups=("spA", "spB", "spC"), seed=0):
    rng = np.random.default_rng(seed)
    shape = tmp_path / "shape_traits_phylo_part.csv"
    table = tmp_path / "scores.tsv"
    with open(shape, "w", newline="") as fs, open(table, "w", newline="") as ft:
        ws = csv.writer(fs); wt = csv.writer(ft, delimiter="\t")
        ws.writerow(["filename"] + [f"PC{i}" for i in range(1, 7)] + ["centroid_size"])
        wt.writerow(["specimen"] + [f"score{i}" for i in range(1, 7)])
        k = 0
        for gi, g in enumerate(groups):
            for _ in range(n_per):
                k += 1
                sid = f"CASENT{k:07d}"
                pcs = rng.normal(0, 0.05, 6); pcs[0] += gi * 1.0; pcs[1] -= gi * 0.7     # shape separates the groups
                ws.writerow([f"{g}_{sid}_lateral.jpg"] + [f"{v:.5f}" for v in pcs] + ["1234.5"])
                wt.writerow([sid] + [f"{v:.5f}" for v in rng.normal(0, 1, 6)])          # table is noise
    return shape, table


def test_pure_helpers():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 3))
    assert cc.rpls(A, A.copy()) == pytest.approx(1.0, abs=1e-9)
    X = np.r_[rng.normal(0, .1, (6, 2)), rng.normal(3, .1, (6, 2))]; y = np.array(["a"] * 6 + ["b"] * 6)
    assert cc.loo_acc(X, y) == 100.0


def test_load_block_shape_and_table(tmp_path):
    import re
    shape, table = _write_blocks(tmp_path)
    rx = re.compile(r"CASENT\d+")
    X, cols, seen = cc.load_block(str(shape), "shape", rx, 3)
    assert cols == ["PC1", "PC2", "PC3"] and len(X) == 15                    # centroid size never used
    assert seen["CASENT0000001"].startswith("spA_")
    T, tcols, _ = cc.load_block(str(table), "table", rx, 3)
    assert tcols == ["PC1", "PC2", "PC3"] and len(T) == 15                   # 6 columns > 3 -> PCA
    M = np.array(list(T.values()))
    assert np.allclose(M.mean(0), 0, atol=1e-9)


def test_main_end_to_end(tmp_path, monkeypatch):
    shape, table = _write_blocks(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["descriptron_combine_characters_v1.py",
                                      "--block", f"shape={shape}", "--block", f"texture={table}:table",
                                      "--out_dir", str(out), "--pcs_per_block", "3", "--permutations", "9", "--seed", "1"])
    cc.main()
    for f in ("combinations.tsv", "nested_selection.tsv", "integration.tsv", "summary.txt", "combinations.png"):
        assert (out / f).exists(), f
    rows = list(csv.DictReader(open(out / "combinations.tsv"), delimiter="\t"))
    assert [r["blocks"] for r in rows] == ["shape", "texture", "shape + texture"]
    acc = {r["blocks"]: float(r["loo_accuracy_pct"]) for r in rows}
    assert acc["shape"] == 100.0 and acc["texture"] < acc["shape"]
    assert all(0 < float(r["p"]) <= 1 for r in rows)
    integ = list(csv.DictReader(open(out / "integration.tsv"), delimiter="\t"))
    assert len(integ) == 1 and integ[0]["n_specimens"] == "15"
    assert "nested leave-one-out accuracy" in (out / "summary.txt").read_text()


def test_group_labels_override_regex(tmp_path, monkeypatch):
    shape, _ = _write_blocks(tmp_path)
    gl = tmp_path / "group_labels.csv"
    with open(gl, "w", newline="") as f:      # one specimen per group -> nothing testable
        w = csv.writer(f); w.writerow(["filename", "group_label"])
        for k in range(1, 16):
            w.writerow([f"x_CASENT{k:07d}.jpg", f"g{k}"])
    monkeypatch.setattr(sys, "argv", ["x", "--block", f"shape={shape}", "--group_labels", str(gl),
                                      "--out_dir", str(tmp_path / "o"), "--permutations", "0"])
    with pytest.raises(SystemExit):
        cc.main()
