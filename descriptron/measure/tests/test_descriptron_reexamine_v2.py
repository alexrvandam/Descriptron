"""descriptron_reexamine_v2: the analysis outputs are byte-identical to v1 on the same input, the map stays
self-contained, the optional DNA layer reaches its specimens without changing any status, and the new layout
(call strength, nearest species, de-overlapped discs) is sound."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import descriptron_reexamine_v1 as rx1       # noqa: E402
import descriptron_reexamine_v2 as rx2       # noqa: E402

PROFILE = "specimen_id:\n  number_regex: '_?(\\d+)_(?:forewing|head)'\nspecies: {}\n"
SPECIES = ("sp1", "sp2", "sp3")


def _setup(tmp_path, key=True, outliers=True):
    """3 species x 4 specimens, sets size and colour. sp1_0 is filed as sp1 but sits in sp2 in both sets
    (as in tests/test_descriptron_reexamine.py), plus a key table and outlying values."""
    rng = np.random.default_rng(0)
    cal = tmp_path / "cal"; cal.mkdir()
    rows = []
    for sp in SPECIES:
        for k in range(4):
            sid = f"{sp}_{k}"
            true = "sp2" if sid == "sp1_0" else sp
            for st in ("size", "colour"):
                for cand in SPECIES:
                    d = 0.2 + rng.random() * 0.1 if cand == true else 2 + rng.random()
                    rows.append([sid, sp, st, cand, d])
    with open(cal / "matrix_identification_distances.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["specimen_id", "species", "set", "candidate_species", "d"]); w.writerows(rows)
    (tmp_path / "profile.yaml").write_text(PROFILE)
    args = ["--calibration_dir", str(cal), "--taxon_profile", str(tmp_path / "profile.yaml")]
    if key:
        krows = [[f"{sp}_{k}", sp, sp] for sp in SPECIES for k in range(4)]
        krows = [["sp3_1", "sp3", "sp1"] if r[0] == "sp3_1" else ["sp1_0", "sp1", "sp2"] if r[0] == "sp1_0" else r for r in krows]
        with open(tmp_path / "key.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t"); w.writerow(["specimen_id", "species", "loo"]); w.writerows(krows)
        args += ["--key_loo", str(tmp_path / "key.tsv")]
    if outliers:
        out = [["img_sp2_3_forewing.tif", "sp2", "forewing", "meas_length_mm", 2.5, 1.0, 12.5, ""]] * 2
        out += [["img_sp3_2_forewing.tif", "sp3", "forewing", "meas_length_mm", 1.3, 1.0, 4.0, ""]]
        with open(tmp_path / "outliers.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["image_base", "species", "category", "feature", "value", "species_median", "robust_z", "note"])
            w.writerows(out)
        args += ["--outlier_flags", str(tmp_path / "outliers.tsv")]
    return args


def _dna(tmp_path):
    d = tmp_path / "dna"; d.mkdir()
    rows = [["Diaphorina-sp1-0", "sp1_0", "sp1", "sp2", 0.003, "False", "sp2", "False"],
            ["Diaphorina-sp2-1", "sp2_1", "sp2", "sp2", -0.0, "True", "sp2", "True"],
            ["Diaphorina-sp3-9", "sp3_9", "sp3", "sp3", 0.0, "True", "", ""]]          # COI only, no morphology record
    with open(d / "specimens_dna_vs_matrix.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["sequence", "specimen_id", "species", "dna_nearest", "dna_nearest_k2p", "dna_agrees_with_label",
                    "matrix_named", "matrix_agrees_with_label"]); w.writerows(rows)
    with open(d / "barcode_gap.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["species", "sequences", "max_intra", "min_inter", "nearest_species", "gap"])
        w.writerows([["sp1", 1, "", 0.003, "sp2", ""], ["sp2", 2, 0.0, 0.003, "sp1", 0.003]])
    return d / "specimens_dna_vs_matrix.tsv"


def _data(html):
    m = re.search(r"const DATA=(.*?);\n", html)
    assert m, "no DATA blob"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_tsv_outputs_identical_to_v1(tmp_path):
    args = _setup(tmp_path)
    rx1.main(args + ["--out_dir", str(tmp_path / "v1")])
    rx2.main(args + ["--out_dir", str(tmp_path / "v2")])
    for f in ("specimens_to_reexamine.tsv", "species_summary.tsv", "reexamine_report.md"):
        assert (tmp_path / "v1" / f).read_bytes() == (tmp_path / "v2" / f).read_bytes(), f
    s1 = json.load(open(tmp_path / "v1" / "reexamine_summary.json"))
    s2 = json.load(open(tmp_path / "v2" / "reexamine_summary.json"))
    assert s2.pop("map_version") == "2.0" and s1 == s2
    t = {r["specimen_id"]: r for r in csv.DictReader(open(tmp_path / "v2" / "specimens_to_reexamine.tsv"), delimiter="\t")}
    assert t["sp1_0"]["status"] == "re-examine" and t["sp2_3"]["status"] == "check annotation"


def test_map_is_self_contained(tmp_path):
    rx2.main(_setup(tmp_path) + ["--out_dir", str(tmp_path / "out")])
    h = (tmp_path / "out" / "reexamine_map.html").read_text()
    assert "const DATA=" in h and '"n_species":3' in h
    assert "<script src" not in h and "http://" not in h and "https://" not in h and "<link" not in h
    d = _data(h)
    assert d["has_dna"] is False and d["map_version"] == "2.0"
    assert 'id="L_dna"' in h and 'id="L_matrix"' in h and 'id="L_key"' in h and 'id="L_ann"' in h
    assert "data-status" in h and len(d["specimens"]) == 12


def test_dna_layer_present_and_never_changes_a_status(tmp_path):
    args = _setup(tmp_path)
    rx2.main(args + ["--out_dir", str(tmp_path / "plain")])
    rx2.main(args + ["--out_dir", str(tmp_path / "dna"), "--dna_table", str(_dna(tmp_path))])
    for f in ("specimens_to_reexamine.tsv", "species_summary.tsv", "reexamine_report.md"):
        assert (tmp_path / "plain" / f).read_bytes() == (tmp_path / "dna" / f).read_bytes(), f
    d = _data((tmp_path / "dna" / "reexamine_map.html").read_text())
    assert d["has_dna"] is True and d["n_dna"] == 2 and d["n_dna_bad"] == 1
    p = {s["id"]: s for s in d["specimens"]}
    assert p["sp1_0"]["dna"]["verdict"] == "disagrees" and p["sp1_0"]["dna"]["nearest_other"] == ["sp2"]
    assert p["sp1_0"]["dna"]["seqs"][0]["k2p"] == 0.003
    assert p["sp2_1"]["dna"]["verdict"] == "agrees" and p["sp2_1"]["dna"]["seqs"][0]["k2p"] == 0.0
    assert "dna" not in p["sp3_0"]
    assert d["dna_only"] == {"sp3": ["sp3_9"]}
    assert d["gaps"]["sp2"]["gap"] == 0.003 and d["gaps"]["sp1"]["gap"] is None
    summ = json.load(open(tmp_path / "dna" / "reexamine_summary.json"))
    assert "1 nearest to another species" in summ["dna"]


def test_call_lines_neighbours_and_layout(tmp_path):
    rx2.main(_setup(tmp_path) + ["--out_dir", str(tmp_path / "out")])
    d = _data((tmp_path / "out" / "reexamine_map.html").read_text())
    p = {s["id"]: s for s in d["specimens"]}
    # the planted mislabel sits in sp2 in both sets: a decisive call; specimens that fit carry no call strength
    assert p["sp1_0"]["named"] == "sp2" and p["sp1_0"]["strength"] > 0.5
    assert all(s["strength"] is None for k, s in p.items() if k != "sp1_0")
    # with 3 species each one's 2 other species are its neighbours: the 3 pairs, each with a distance
    assert sorted((a, b) for a, b, _, _ in d["nbr"]) == [("sp1", "sp2"), ("sp1", "sp3"), ("sp2", "sp3")]
    assert all(dist > 0 for _, _, dist, _ in d["nbr"])
    sp = {s["id"]: s for s in d["species"]}
    for a in sp.values():
        for b in sp.values():
            if a["id"] < b["id"]:
                assert np.hypot(a["x"] - b["x"], a["y"] - b["y"]) >= a["r"] + b["r"] - 1e-6
    # the planted specimen takes the ring slot facing the species the matrix names it as
    s1, s2 = sp["sp1"], sp["sp2"]
    v = np.array([s2["x"] - s1["x"], s2["y"] - s1["y"]])
    mates = [k for k in p if k.startswith("sp1_")]
    cos = {k: np.dot([p[k]["x"] - s1["x"], p[k]["y"] - s1["y"]], v) for k in mates}
    assert max(cos, key=cos.get) == "sp1_0"


def test_declutter_separates_coincident_species():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.001, 0.0], [1.0, 1.0], [-1.0, 0.5]])
    P, r = rx2.declutter(X, [5, 5, 5, 2, 8])
    assert np.abs(P).max() <= 1.0 + 1e-9
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            assert np.hypot(*(P[i] - P[j])) >= r[i] + r[j] - 1e-6
