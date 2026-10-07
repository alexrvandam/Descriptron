"""biorag_coded_states_from_coco_v1: GUI / GBIF-annotator attributes -> coded-state TSV and matrix columns;
the key builder then separates two species on a recorded character."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable


def _setup(tmp):
    imgs, anns, gl = [], [], ["filename,group_label"]
    k = 0
    for sp, setae, count in (("spA", "dense", 12), ("spB", "sparse", 3)):
        for i in range(1, 4):
            k += 1
            fn = f"Project_morph_{sp}_{i}_elytron_m_ch00_SV.tif"
            imgs.append({"id": k, "file_name": fn, "width": 100, "height": 100})
            anns.append({"id": k, "image_id": k, "category_id": 1, "segmentation": [[1, 1, 50, 1, 50, 50]],
                         "attributes": {"setae": setae, "setae_count": str(count + i % 2), "_observer": "x"}})
            gl.append(f"{fn},{sp}")
    coco = {"images": imgs, "categories": [{"id": 1, "name": "elytron"}], "annotations": anns}
    (tmp / "gui.json").write_text(json.dumps(coco))
    (tmp / "gl.csv").write_text("\n".join(gl) + "\n")
    (tmp / "profile.yaml").write_text("specimen_id:\n  number_regex: '_(\\d+)_elytron'\n")
    # a minimal Tier-1 matrix: one uninformative measurement, same in both species
    m = tmp / "matrix"; m.mkdir()
    rows = [{"species": sp, "specimen_id": f"{sp}_{i}", "sex": "male", "category": "elytron", "base_category": "elytron",
             "column": "meas_length_mm", "feature_id": "elytron.length_mm", "tier": "key", "family": "length",
             "value": 1.0 + 0.01 * i, "n_images": 1} for sp in ("spA", "spB") for i in range(1, 4)]
    pd.DataFrame(rows).to_csv(m / "specimen_matrix_long.csv", index=False)
    pd.DataFrame([{"feature_id": "elytron.length_mm", "category": "elytron", "base_category": "elytron",
                   "column": "meas_length_mm", "tier": "key", "family": "length", "label": "elytron: length",
                   "definition": "length", "unit": "mm", "structure_sex": "both", "section": "elytron",
                   "key_priority": 1, "n_species": 2, "n_specimens": 6, "conversion": ""}]).to_csv(
        m / "feature_dictionary.tsv", sep="\t", index=False)
    return tmp


def _run(tmp, *extra):
    return subprocess.run([PY, str(HERE / "biorag_coded_states_from_coco_v1.py"), "--coco", str(tmp / "gui.json"),
                           "--group_labels", str(tmp / "gl.csv"), "--taxon_profile", str(tmp / "profile.yaml"), *extra], capture_output=True, text=True)


def test_tsv_and_matrix(tmp_path):
    t = _setup(tmp_path)
    r = _run(t, "--matrix_dir", str(t / "matrix"), "--out_dir", str(t / "out"))
    assert r.returncode == 0, r.stderr
    tsv = pd.read_csv(t / "out/coded_states_by_specimen.tsv", sep="\t")
    assert set(tsv["character"]) == {"setae", "setae_count"}               # "_observer" ignored
    assert set(tsv["source"]) == {"human"} and set(tsv["specimen_id"]) == {f"{s}_{i}" for s in ("spA", "spB") for i in (1, 2, 3)}
    fd = pd.read_csv(t / "out/feature_dictionary.tsv", sep="\t")
    added = fd[fd["column"] == "coded_state"]
    assert set(added["feature_id"]) == {"elytron.coded_setae__dense", "elytron.coded_setae__sparse", "elytron.coded_setae_count"}
    assert "elytron: number of setae" in set(added["label"]) and set(added.dropna(subset=["state"])["state"]) >= {"dense", "sparse"}
    assert set(added["tier"]) == {"key"} and set(added["family"]) == {"coded_state", "meristic"}
    long = pd.read_csv(t / "out/specimen_matrix_long.csv")
    dense = long[long["feature_id"] == "elytron.coded_setae__dense"].set_index("specimen_id")["value"]
    assert dense["spA_1"] == 1.0 and dense["spB_1"] == 0.0
    # the source matrix is untouched
    assert len(pd.read_csv(t / "matrix/feature_dictionary.tsv", sep="\t")) == 1


def test_refuses_to_overwrite_source(tmp_path):
    t = _setup(tmp_path)
    r = _run(t, "--matrix_dir", str(t / "matrix"), "--out_dir", str(t / "matrix"))
    assert r.returncode != 0 and "refusing" in (r.stderr + r.stdout)


def test_gbif_annotator_trait_only_annotation(tmp_path):
    t = _setup(tmp_path)
    d = json.loads((t / "gui.json").read_text())
    for a in d["annotations"]:                                  # the annotator's region-level, geometry-free form
        a["is_trait_only"] = True; a.pop("segmentation")
    (t / "gui.json").write_text(json.dumps(d))
    r = _run(t, "--out_dir", str(t / "out2"))
    assert r.returncode == 0, r.stderr
    assert len(pd.read_csv(t / "out2/coded_states_by_specimen.tsv", sep="\t")) == 12


def test_key_builder_separates_species_on_a_coded_character(tmp_path):
    t = _setup(tmp_path)
    assert _run(t, "--matrix_dir", str(t / "matrix"), "--out_dir", str(t / "out")).returncode == 0
    r = subprocess.run([PY, str(HERE / "biorag_key_builder_v1.py"), "--matrix_dir", str(t / "out"),
                        "--output_dir", str(t / "key"), "--llm-backend", "none", "--no-loo"],
                       capture_output=True, text=True, cwd=str(HERE))
    assert r.returncode == 0, r.stderr[-2000:]
    tree = json.loads((t / "key/key_tree.json").read_text())
    assert "coded_setae" in json.dumps(tree)


def _coded_matrix(tmp):
    t = _setup(tmp)
    assert _run(t, "--matrix_dir", str(t / "matrix"), "--out_dir", str(t / "out")).returncode == 0
    long = pd.read_csv(t / "out/specimen_matrix_long.csv")
    g = long.groupby(["species", "feature_id"])["value"]
    pd.DataFrame({"n": g.count(), "min": g.min(), "max": g.max(), "mean": g.mean(), "median": g.median()}
                 ).reset_index().to_csv(t / "out/species_feature_summary.csv", index=False)
    return t


def test_key_couplet_says_the_state_in_words(tmp_path):
    t = _coded_matrix(tmp_path)
    r = subprocess.run([PY, str(HERE / "biorag_key_builder_v1.py"), "--matrix_dir", str(t / "out"),
                        "--output_dir", str(t / "key"), "--llm-backend", "none", "--no-loo"],
                       capture_output=True, text=True, cwd=str(HERE))
    assert r.returncode == 0, r.stderr[-1500:]
    txt = json.dumps(json.loads((t / "key/key_tree.json").read_text()))
    assert "setae dense" in txt or "setae sparse" in txt, txt[:1500]
    assert "present" not in txt.lower().replace("presented", "")


def _evidence(t):
    sys.path.insert(0, str(HERE))
    import biorag_description_refiner_v1 as rf
    from types import SimpleNamespace
    args = SimpleNamespace(matrix_dir=str(t / "out"), compiled_dir=None, key_tree=None, localities=None, prior_cache=None)
    profile = {"species": {}, "structures": {"elytron": {"term": "elytron", "section": "Thorax"}},
               "description_sections": ["Thorax"]}
    return rf, rf.Evidence(args, profile)


def test_data_sheet_gives_recorded_states_in_words(tmp_path):
    t = _coded_matrix(tmp_path)
    rf, ev = _evidence(t)
    sheet, allowed = rf.build_sheet(ev, "spA")
    assert "DESCRIPTIVE CHARACTERS RECORDED BY THE TAXONOMIST" in sheet
    assert "elytron, setae: dense (3 of 3 specimens)" in sheet and "other species: sparse in spB" in sheet
    assert "[D1] elytron with setae dense: recorded in no other species" in sheet
    assert "= dense:" not in sheet                         # never as a 0-1 number in the Tier-1 list
    assert 3.0 in allowed["t1"]


def test_audit_checks_named_states(tmp_path):
    t = _coded_matrix(tmp_path)
    sys.path.insert(0, str(HERE))
    import biorag_confabulation_checker_v2 as ck
    from types import SimpleNamespace
    ev = SimpleNamespace(fdict=pd.read_csv(t / "out/feature_dictionary.tsv", sep="\t").set_index("feature_id"),
                         long=pd.read_csv(t / "out/specimen_matrix_long.csv"),
                         profile={"structures": {"elytron": {"term": "elytron"}}})
    issues = []
    ok = ck.verify_coded_states("Diagnosis", "Elytron with setae dense.", "spA", ev, issues)
    assert ok and all(r["status"] == "ok" for r in ok) and not issues
    bad = ck.verify_coded_states("Diagnosis", "Elytron with setae sparse.", "spA", ev, issues)
    assert any(r["type"] == "coded_state_mismatch" and r["status"] == "error" for r in bad)
    neg = ck.verify_coded_states("Thorax", "Elytron: setae not dense.", "spA", ev, issues)
    assert any(r["type"] == "coded_state_contradicted" for r in neg)


def test_unknown_character_with_whole_numbers_is_a_count(tmp_path, monkeypatch):
    sys.path.insert(0, str(HERE))
    import biorag_coded_states_from_coco_v1 as cs
    t = _setup(tmp_path)
    d = json.loads((t / "gui.json").read_text())
    for a in d["annotations"]:
        a["attributes"] = {"spine_number": str(4 + a["id"] % 2)}
    (t / "gui.json").write_text(json.dumps(d))
    df, _ = cs.read_states([t / "gui.json"], t / "gl.csv", {}, {"characters": {}})
    assert set(df["type"]) == {"count"}
