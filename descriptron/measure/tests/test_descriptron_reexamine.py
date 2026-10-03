"""descriptron_reexamine_v1: a planted mislabel is listed for re-examination, a weak key is not counted on its own,
an outlying value is an annotation check (not an identity flag), the map is self-contained, and the pipeline runs the
step after calibrate."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import descriptron_reexamine_v1 as rx        # noqa: E402

PROFILE = "specimen_id:\n  number_regex: '_?(\\d+)_(?:forewing|head)'\nspecies: {}\n"


def _setup(tmp_path, key_rows=None, outliers=None):
    """3 species x 4 specimens, sets size and colour. sp1_0 is filed as sp1 but sits in sp2 in both sets."""
    rng = np.random.default_rng(0)
    cal = tmp_path / "cal"; cal.mkdir()
    rows = []
    for s, sp in enumerate(("sp1", "sp2", "sp3")):
        for k in range(4):
            sid = f"{sp}_{k}"
            true = "sp2" if sid == "sp1_0" else sp
            for st in ("size", "colour"):
                for cand in ("sp1", "sp2", "sp3"):
                    d = 0.2 + rng.random() * 0.1 if cand == true else 2 + rng.random()
                    rows.append([sid, sp, st, cand, d])
    with open(cal / "matrix_identification_distances.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["specimen_id", "species", "set", "candidate_species", "d"]); w.writerows(rows)
    (tmp_path / "profile.yaml").write_text(PROFILE)
    args = ["--calibration_dir", str(cal), "--taxon_profile", str(tmp_path / "profile.yaml"), "--out_dir", str(tmp_path / "out")]
    if key_rows is not None:
        with open(tmp_path / "key.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["specimen_id", "species", "loo"]); w.writerows(key_rows)
        args += ["--key_loo", str(tmp_path / "key.tsv")]
    if outliers is not None:
        with open(tmp_path / "outliers.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["image_base", "species", "category", "feature", "value", "species_median", "robust_z", "note"])
            w.writerows(outliers)
        args += ["--outlier_flags", str(tmp_path / "outliers.tsv")]
    return args


def _read(tmp_path):
    return {r["specimen_id"]: r for r in csv.DictReader(open(tmp_path / "out" / "specimens_to_reexamine.tsv"), delimiter="\t")}


def test_planted_mislabel_is_listed(tmp_path):
    rx.main(_setup(tmp_path))
    t = _read(tmp_path)
    bad = t["sp1_0"]
    assert bad["status"] == "re-examine" and bad["matrix_named"] == "sp2" and bad["sets_agreeing"] == "0"
    assert "matrix names it sp2" in bad["reasons"]
    assert all(r["status"] == "fits" for k, r in t.items() if k != "sp1_0")
    summ = json.load(open(tmp_path / "out" / "reexamine_summary.json"))
    assert summ["re_examine"] == 1 and summ["matrix_named_back"] == 11


def test_weak_key_alone_is_not_counted(tmp_path):
    # the key names sp3_1 as sp1 and gets half the specimens wrong -> less reliable than the matrix
    rows = [[f"{sp}_{k}", sp, sp if k % 2 == 0 else "unresolved"] for sp in ("sp1", "sp2", "sp3") for k in range(4)]
    rows = [r if r[0] != "sp3_1" else ["sp3_1", "sp3", "sp1"] for r in rows]
    rx.main(_setup(tmp_path, key_rows=rows))
    t = _read(tmp_path)
    assert t["sp3_1"]["status"] == "fits" and "not counted" in t["sp3_1"]["reasons"]


def test_outlying_value_reaches_its_specimen(tmp_path):
    out = [["img_sp2_3_forewing.tif", "sp2", "forewing", "meas_length_mm", 2.5, 1.0, 12.5, ""]] * 2   # same value twice
    out += [["img_sp3_2_forewing.tif", "sp3", "forewing", "meas_length_mm", 1.3, 1.0, 4.0, ""]]       # mild: listed only
    rx.main(_setup(tmp_path, outliers=out))
    t = _read(tmp_path)
    # 2.5x the species median: an ANNOTATION check, never an identity flag; 1.3x: nothing
    assert t["sp2_3"]["status"] == "check annotation" and t["sp2_3"]["identity"] == "fits"
    assert t["sp2_3"]["n_outlying_values"] == "1" and "2.5x" in t["sp2_3"]["annotation"]
    assert t["sp3_2"]["status"] == "fits" and t["sp3_2"]["annotation"] == ""


def test_corroboration_alone_does_not_flag(tmp_path):
    """an outline flag or a key disagreement with no matrix disagreement is listed but does not raise a flag;
    together with the matrix it makes 're-examine'"""
    rows = [[f"{sp}_{k}", sp, sp] for sp in ("sp1", "sp2", "sp3") for k in range(4)]
    rows = [["sp3_1", "sp3", "sp1"] if r[0] == "sp3_1" else r for r in rows]          # a reliable key, one disagreement
    rx.main(_setup(tmp_path, key_rows=rows))
    t = _read(tmp_path)
    assert t["sp3_1"]["status"] == "check"            # reliable key (11/12 >= matrix) counts as a check on its own
    assert t["sp1_0"]["status"] == "re-examine"


def test_map_is_self_contained(tmp_path):
    rx.main(_setup(tmp_path))
    h = (tmp_path / "out" / "reexamine_map.html").read_text()
    assert "const DATA=" in h and '"n_species":3' in h
    assert "<script src" not in h and "http://" not in h and "https://" not in h and "<link" not in h


def test_pipeline_runs_reexamine_after_calibrate():
    sys.path.insert(0, str(HERE))
    import run_full_pipeline_v2 as pipe
    assert pipe.ALL_STEPS["reexamine"][0] > pipe.ALL_STEPS["calibrate"][0]
    for order in (pipe.V2_STEPS, pipe.V2_WORKFLOW):
        assert order.index("reexamine") == order.index("calibrate") + 1
