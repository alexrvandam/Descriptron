"""run_full_pipeline_v2's added steps (2.3.0): trait_stats finds a structure's colour table and aligned outline,
runs descriptron_trait_stats and writes its results; phylo without --tree is skipped; neither step can stop the
pipeline (they always return True, even when the program fails)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import run_full_pipeline_v2 as pipe          # noqa: E402


def _base(tmp_path):
    rng = np.random.default_rng(0); base = tmp_path / "run"; cat = "forewing"
    (base / "semilandmarks" / cat).mkdir(parents=True); (base / "color_homology" / cat).mkdir(parents=True)
    (base / "semilandmarks" / cat / f"shape_traits_phylo_{cat}.csv").write_text("x\n")      # marks a finished GPA
    groups, rows, imgs, anns = [["filename", "group_label"]], [], [], []
    for k, (sp, shift) in enumerate([(s, i * 3) for i, s in enumerate(("sp1", "sp2", "sp3")) for _ in range(5)]):
        fn = f"{sp}_{k}.tif"; groups.append([fn, sp])
        rows.append([f"{fn}_{k + 1}"] + list(rng.normal(shift, 1, 3)) + [100 + rng.normal(), 128 + rng.normal(), 128 + rng.normal()])
        ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        pts = np.column_stack([(1 + 0.1 * shift) * np.cos(ang), np.sin(ang)]) + rng.normal(0, 0.01, (20, 2))
        imgs.append({"id": k + 1, "file_name": f"{fn}_{k + 1}"}); anns.append({"id": k + 1, "image_id": k + 1, "segmentation": [pts.flatten().tolist()]})
    with open(base / "color_homology" / cat / f"color_homology_features_{cat}_combined_hw1.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "r0c0_dL_rel", "r0c1_dL_rel", "r1c0_dL_rel", "r0c0_L_mean_abs", "r0c0_a_mean_abs", "r0c0_b_mean_abs"]); w.writerows(rows)
    (base / "semilandmarks" / cat / "aligned_coco.json").write_text(json.dumps({"images": imgs, "annotations": anns}))
    with open(tmp_path / "groups.csv", "w", newline="") as f:
        csv.writer(f).writerows(groups)
    return base, cat


def _cfg(base, tmp_path, **kw):
    return {"output_base": str(base), "group_labels": str(tmp_path / "groups.csv"), "force": False, "dry_run": False,
            "trait_stats_permutations": 99, **kw}


def test_trait_stats_step_writes_results(tmp_path):
    base, cat = _base(tmp_path); logs = tmp_path / "logs"; logs.mkdir()
    assert pipe.step_trait_stats(_cfg(base, tmp_path), sys.executable, logs) is True
    summ = list(csv.DictReader(open(base / "trait_stats" / cat / "summary.csv")))
    assert {r["trait_set"] for r in summ} == {"colour pattern", "outline shape"}
    assert (base / "trait_stats" / cat / "trait_stats_report.md").exists()


def test_phylo_step_without_tree_is_skipped(tmp_path):
    base, cat = _base(tmp_path); logs = tmp_path / "logs"; logs.mkdir()
    assert pipe.step_phylo(_cfg(base, tmp_path, tree=None), sys.executable, logs) is True
    assert not (base / "phylo").exists()


def test_phylo_step_runs_with_a_tree(tmp_path):
    base, cat = _base(tmp_path); logs = tmp_path / "logs"; logs.mkdir()
    tree = tmp_path / "t.nwk"; tree.write_text("((sp1:1,sp2:1):1,(sp3:1.5,sp4:1.5):0.5);")   # sp4 has no data
    # descriptron_phylo needs >= 4 species: with 3 it reports the skip and the step still succeeds
    assert pipe.step_phylo(_cfg(base, tmp_path, tree=str(tree)), sys.executable, logs) is True
    assert (base / "phylo" / cat / "phylo_report.md").exists()


def test_failures_never_stop_the_pipeline(tmp_path):
    base, cat = _base(tmp_path); logs = tmp_path / "logs"; logs.mkdir()
    cfg = _cfg(base, tmp_path); cfg["group_labels"] = str(tmp_path / "missing.csv")      # the program will fail
    assert pipe.step_trait_stats(cfg, sys.executable, logs) is True
