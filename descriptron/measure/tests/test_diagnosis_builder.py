"""biorag_diagnosis_builder_v1: a diagnosis written by fixed rules from the key path and the lineage-specific
characters (recorded states / absences, counts, unique measurements), and its use by the description step."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
PROFILE = HERE / "biorag_prompts" / "taxon_profiles" / "TEMPLATE_taxon_profile.yaml"


def matrix(tmp):
    """4 species x 3 specimens: spA has lost its spine (presence) and has 8 antennal segments (others 10-11);
    spB alone has dense setae; body length separates spC from spD (the key's last couplet)"""
    rows, fd = [], []
    def add(fid, cat, col, fam, tier, unit, label, vals, character="", state=""):
        fd.append({"feature_id": fid, "category": cat, "base_category": cat, "column": col, "tier": tier,
                   "family": fam, "label": label, "unit": unit, "character": character, "state": state,
                   "character_label": character, "definition": label, "structure_sex": "both", "section": cat,
                   "key_priority": 1})
        for sp, vs in vals.items():
            for i, v in enumerate(vs, 1):
                rows.append({"species": sp, "specimen_id": f"{sp}_{i}", "sex": "unknown", "category": cat,
                             "base_category": cat, "column": col, "feature_id": fid, "tier": tier, "family": fam,
                             "value": v, "n_images": 1})
    sp = ["spA", "spB", "spC", "spD"]
    add("spine.coded_presence__absent", "spine", "coded_state", "coded_state", "key", "", "spine = absent",
        {s: [1.0] * 3 if s == "spA" else [0.0] * 3 for s in sp}, "presence", "absent")
    add("spine.coded_presence__present", "spine", "coded_state", "coded_state", "key", "", "spine = present",
        {s: [0.0] * 3 if s == "spA" else [1.0] * 3 for s in sp}, "presence", "present")
    add("pronotum.coded_setae__dense", "pronotum", "coded_state", "coded_state", "key", "", "pronotum: setae = dense",
        {s: [1.0] * 3 if s == "spB" else [0.0] * 3 for s in sp}, "setae", "dense")
    add("pronotum.coded_setae__sparse", "pronotum", "coded_state", "coded_state", "key", "",
        "pronotum: setae = sparse", {s: [0.0] * 3 if s == "spB" else [1.0] * 3 for s in sp}, "setae", "sparse")
    add("antenna.coded_segments", "antenna", "coded_state", "meristic", "key", "count", "antenna: segments",
        {"spA": [8, 8, 8], "spB": [10, 11, 10], "spC": [11, 10, 11], "spD": [10, 10, 11]})
    add("body.length_mm", "body", "length_mm", "length", "key", "mm", "body length",
        {"spA": [1.0, 1.1, 1.05], "spB": [1.2, 1.3, 1.25], "spC": [1.5, 1.6, 1.55], "spD": [2.0, 2.1, 2.05]})
    md = tmp / "m"; md.mkdir()
    long = pd.DataFrame(rows); fdd = pd.DataFrame(fd)
    long.to_csv(md / "specimen_matrix_long.csv", index=False)
    fdd.to_csv(md / "feature_dictionary.tsv", sep="\t", index=False)
    ss = (long.groupby(["species", "feature_id"])["value"]
          .agg(n="count", min="min", max="max", mean="mean", sd="std", median="median").reset_index())
    ss.merge(fdd[["feature_id", "tier", "unit", "category", "column"]], on="feature_id").to_csv(
        md / "species_feature_summary.csv", index=False)
    def ch(fid, label, unit, fam, thr, a_op, a, b, an, bn, binary=False):
        return {"role": "primary", "feature_id": fid, "label": label, "definition": label, "unit": unit,
                "family": fam, "column": "", "structure_sex": "both", "threshold": thr, "threshold_text": str(thr),
                "A_operator": a_op, "B_operator": ">" if a_op == "<=" else "<=", "A_range": a, "B_range": b,
                "perfect_separation": True, "balanced_accuracy": 1.0, "binary": binary, "A_n": an, "B_n": bn}
    key = {"meta": {}, "species": sp, "validation": {}, "couplets": [
        {"number": 1, "A_species": ["spA"], "B_species": ["spB", "spC", "spD"], "A_goto": "spA", "B_goto": 2,
         "characters": [ch("spine.coded_presence__present", "spine = present", "", "coded_state", 0.5, "<=",
                           [0, 0], [1, 1], 3, 9, True)]},
        {"number": 2, "A_species": ["spB"], "B_species": ["spC", "spD"], "A_goto": "spB", "B_goto": 3,
         "characters": [ch("body.length_mm", "body length", "mm", "length", 1.4, "<=", [1.2, 1.3], [1.5, 2.1], 3, 6)]},
        {"number": 3, "A_species": ["spC"], "B_species": ["spD"], "A_goto": "spC", "B_goto": "spD",
         "characters": [ch("body.length_mm", "body length", "mm", "length", 1.8, "<=", [1.5, 1.6], [2.0, 2.1], 3, 3)]}]}
    (tmp / "key_tree.json").write_text(json.dumps(key))
    return md, tmp / "key_tree.json"


def build(tmp):
    md, key = matrix(tmp)
    r = subprocess.run([sys.executable, str(HERE / "biorag_diagnosis_builder_v1.py"), "--matrix_dir", str(md),
                        "--key_tree", str(key), "--taxon_profile", str(PROFILE), "--out_dir", str(tmp / "d")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads((tmp / "d" / "diagnoses.json").read_text()), md, key


def test_key_path_and_lineage_specific_characters(tmp_path):
    D, _, _ = build(tmp_path)
    a = D["spA"]["text"]
    assert "spine absent" in a and "key couplet 1a" in a and "recorded in no other species, 3 scored" in a
    assert "segments 8 (n = 3): fewer than in" in a and "Pronotum setae sparse" not in a    # meristic; sparse is shared by 2 of 3
    assert "setae dense" in D["spB"]["text"] and "recorded in no other species" in D["spB"]["text"]
    c = D["spC"]["text"]
    assert "key couplet 3a" in c and "observed 1.500–1.600 mm; n = 3" in c        # the species' own range
    assert "key couplet 1b" not in c                                           # 'spine present' says nothing here
    assert all(x["citations"] for x in D.values() if isinstance(x, dict) and "citations" in x)


def test_deterministic(tmp_path):
    D1, _, _ = build(tmp_path / "1" if (tmp_path / "1").mkdir() is None else tmp_path)
    D2, _, _ = build(tmp_path / "2" if (tmp_path / "2").mkdir() is None else tmp_path)
    assert {k: v["text"] for k, v in D1.items() if k != "_meta"} == {k: v["text"] for k, v in D2.items() if k != "_meta"}


def test_description_step_uses_it_verbatim(tmp_path):
    import biorag_description_refiner_v1 as rf
    D, md, key = build(tmp_path)
    (tmp_path / "d" / "diagnoses.json")
    args = types.SimpleNamespace(deterministic_diagnosis=str(tmp_path / "d" / "diagnoses.json"))
    assert rf.deterministic_diagnosis(args, "spA")["text"] == D["spA"]["text"]
    assert rf.deterministic_diagnosis(types.SimpleNamespace(deterministic_diagnosis=None), "spA") is None
