"""2.7.8 character loss: a structure recorded ABSENT (GUI v84 Record absent / Lost keypoint) is a character state;
one that is not visible or simply not annotated is missing data. Checks the completeness check, the presence /
absence characters, landmark GPA on the shared landmarks, and the keypoint distances keeping their numbers."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_check_completeness_v1 as cc   # noqa: E402

SQ = [[10, 10, 60, 10, 60, 60, 10, 60]]


def coco(specimens):
    """specimens: {name: {"masks": [...], "status": {...}, "kp": [(x, y, v), ...], "kstat": [...]}}"""
    cats = [{"id": 1, "name": "eye"}, {"id": 2, "name": "antenna"}, {"id": 3, "name": "pronotum"},
            {"id": 9, "name": "line_keypoints", "supercategory": "keypoints"}]
    cid = {c["name"]: c["id"] for c in cats}
    imgs, anns, k = [], [], 1
    for i, (name, d) in enumerate(specimens.items(), 1):
        im = {"id": i, "file_name": f"{name}.tif", "width": 100, "height": 100}
        if d.get("status"):
            im["structure_status"] = d["status"]
        imgs.append(im)
        for m in d.get("masks", []):
            anns.append({"id": k, "image_id": i, "category_id": cid[m], "segmentation": SQ}); k += 1
        if d.get("kp"):
            a = {"id": k, "image_id": i, "category_id": 9, "keypoints": [c for p in d["kp"] for c in p],
                 "num_keypoints": sum(p[2] > 0 for p in d["kp"])}
            if d.get("kstat"):
                a["keypoint_status"] = d["kstat"]
            anns.append(a); k += 1
    return {"images": imgs, "annotations": anns, "categories": cats}


KP_FULL = [(10, 10, 2), (20, 10, 2), (30, 20, 2), (40, 40, 2), (15, 50, 2)]
KP_LOST = [(10, 11, 2), (21, 10, 2), (0, 0, 0), (41, 40, 2), (15, 51, 2)]


def data():
    return coco({
        "spA_1": {"masks": ["eye", "antenna", "pronotum"], "kp": KP_FULL},
        "spA_2": {"masks": ["eye", "pronotum"], "kp": KP_FULL},                       # antenna forgotten
        "spB_1": {"masks": ["antenna", "pronotum"], "status": {"eye": "absent"}, "kp": KP_LOST,
                  "kstat": ["present", "present", "absent", "present", "present"]},
        "spB_2": {"masks": ["antenna", "pronotum"], "status": {"eye": "absent"}, "kp": KP_LOST,
                  "kstat": ["present", "present", "absent", "present", "present"]},
        "spC_1": {"masks": ["eye", "antenna"], "status": {"pronotum": "unknown"},
                  "kp": [(10, 10, 2), (20, 10, 2), (0, 0, 0), (40, 40, 2), (15, 50, 2)]},   # negative point
        "spC_2": {"masks": ["eye", "antenna", "pronotum"], "status": {"eye": "absent"}},  # conflict
    })


def test_states_and_gaps():
    st, chars, conflicts = cc.structure_states([data()])
    assert st["spB_1"]["eye"] == "absent" and st["spA_1"]["eye"] == "present"
    assert "antenna" not in st["spA_2"]                                # a gap: nothing recorded
    assert st["spC_1"]["pronotum"] == "unknown"                        # recorded not visible = missing data
    assert st["spB_1"]["line_keypoints keypoint 3"] == "absent"        # Lost keypoint
    assert st["spC_1"]["line_keypoints keypoint 3"] == "unknown"       # negative point
    assert st["spA_1"]["line_keypoints keypoint 3"] == "present"
    assert st["spC_2"]["eye"] == "present" and conflicts and conflicts[0]["image"] == "spC_2"
    sp = cc.species_map(list(st), species_regex=r"^(sp[A-Z])_")
    rows, work, per, chars = cc.check(st, chars, sp)
    first = [w for w in work if w["priority"] == 1]
    assert any(w["image"] == "spA_2" and w["character"] == "antenna" for w in first)  # conspecific has it
    eye = next(p for p in per if p["character"] == "eye")
    assert eye["absent"] == 2 and eye["present"] == 4 - 0 and eye["share_scored"] == 1.0


def test_command_line_and_strict(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps(data()))
    r = subprocess.run([sys.executable, str(HERE / "descriptron_check_completeness_v1.py"), "--coco",
                        str(tmp_path / "c.json"), "--species_regex", r"^(sp[A-Z])_", "--out_dir", str(tmp_path / "o"),
                        "--strict"], capture_output=True, text=True)
    assert r.returncode == 1, r.stderr[-1500:]                         # gaps and a conflict remain
    for f in ("completeness_matrix.tsv", "completeness_worklist.csv", "completeness_conflicts.csv",
              "completeness_by_character.tsv", "completeness_heatmap.png", "completeness_summary.json"):
        assert (tmp_path / "o" / f).exists(), f
    s = json.loads((tmp_path / "o/completeness_summary.json").read_text())
    assert s["states"]["absent"] == 4 and s["conflicts"] == 1          # spC_2: recorded absent but has a mask


def test_presence_characters_reach_the_matrix(tmp_path):
    import biorag_coded_states_from_coco_v1 as cs
    (tmp_path / "c.json").write_text(json.dumps(data()))
    gl = tmp_path / "gl.csv"
    pd.DataFrame({"filename": [f"{n}.tif" for n in ("spA_1", "spA_2", "spB_1", "spB_2", "spC_1", "spC_2")],
                  "group_label": ["A", "A", "B", "B", "C", "C"]}).to_csv(gl, index=False)
    df, _ = cs.read_states([str(tmp_path / "c.json")], str(gl), {}, {"characters": {}})
    pres = df[df["character"] == "presence"]
    eye = pres[pres["category"] == "eye"].set_index("image")["state"].to_dict()
    assert eye["spB_1"] == "absent" and eye["spA_1"] == "present"
    assert set(pres["category"]) == {"eye", "line_keypoints_keypoint_3"}   # only structures recorded absent
    kp3 = pres[pres["category"] == "line_keypoints_keypoint_3"].set_index("image")["state"].to_dict()
    assert "spC_1" not in kp3                                           # a negative point is NOT a loss
    per = cs.one_per_specimen(pres)
    fd, long, dropped = cs.to_features(per, "key", 2, None)
    assert "eye = absent" in set(fd["label"]) and "eye = present" in set(fd["label"])


def test_gpa_keeps_specimens_on_shared_landmarks(tmp_path):
    import landmark_gpa_V2 as g
    rng = np.random.default_rng(0)
    base = np.array([[0, 0], [10, 0], [10, 5], [5, 9], [0, 6]], float)
    imgs, anns = [], []
    for i in range(8):
        pts = base + rng.normal(0, 0.3, base.shape)
        kp = []
        for j, (x, y) in enumerate(pts):
            v = 0 if (i < 3 and j == 2) else 2                          # three specimens lack landmark 3
            kp += [float(x), float(y), v] if v else [0, 0, 0]
        imgs.append({"id": i, "file_name": f"s{i}.png"})
        anns.append({"id": i, "image_id": i, "category_id": 1, "keypoints": kp, "num_keypoints": 5})
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "wing"}]}))
    g.LANDMARK_IDS.clear()
    configs, _ = g.load_keypoints(str(p))["wing"]
    assert len(configs) == 5                                           # default: incomplete specimens left out
    configs, fns = g.load_keypoints(str(p), present_in_all=True)["wing"]
    assert len(configs) == 8 and configs[0].shape == (4, 2) and g.LANDMARK_IDS["wing"] == [1, 2, 4, 5]


def test_keypoint_distances_keep_their_numbers():
    src = (HERE / "measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "process_keypoints")
    ns = {"np": np, "logging": __import__("logging")}
    exec(compile(ast.Module([fn], []), "V35", "exec"), ns)
    kp = [c for p in KP_LOST for c in p]
    m = ns["process_keypoints"]({"keypoints": kp, "num_keypoints": 4, "category_id": 9}, 1, 0, "line_keypoints")
    assert m is not None                                               # used to be rejected as 'inconsistent'
    D = np.array(m["keypoints_distance_matrix"], dtype=float)
    assert D.shape == (5, 5) and np.isnan(D[2]).all() and m["keypoint_ids"] == [1, 2, 4, 5]
    assert D[3, 4] == pytest.approx(np.hypot(41 - 15, 40 - 51))        # keypoints 4-5, not renumbered


def test_gui_v84_saves_states():
    """the saving code writes absent / not visible as COCO v=0 plus keypoint_status (static check of v84)"""
    gui = HERE.parent / "descriptron-v2-v84.py"
    if not gui.exists():
        pytest.skip("GUI source not beside measure/ (pip layout)")
    s = gui.read_text()
    assert 'point_label_options = ["Positive", "Negative", "Lost"]' in s
    assert '"keypoint_status": [KP_STATUS.get(v, "unknown")' in s and "'keypoint_status': [KP_STATUS.get(" in s
    assert 'img["structure_status"] = dict(sorted(rec.items()))' in s
    assert "coco_kps.extend([coord[0], coord[1], 2])" not in s        # every manual point was saved visible


def test_images_of_one_specimen_are_merged_and_sex_limited_structures_skipped(tmp_path):
    """found on real data: specimens photographed in several images (head, leg, terminalia) must be one unit,
    and a female is not 'missing' an aedeagus"""
    d = coco({"spA_1_head_m": {"masks": ["eye"]}, "spA_1_leg_m": {"masks": ["antenna"]},
              "spA_2_head_f": {"masks": ["eye"]}, "spA_2_leg_f": {"masks": ["antenna"]}})
    d["categories"].append({"id": 4, "name": "aedeagus"})
    d["annotations"].append({"id": 99, "image_id": 1, "category_id": 4, "segmentation": SQ})
    st, chars, _ = cc.structure_states([d])
    sp = cc.species_map(list(st), species_regex=r"^(sp[A-Z])_")
    spec = cc.specimen_map(list(st), sp, specimen_regex=r"^(sp[A-Z]_\d+)_")
    merged, images_of, conflicts = cc.merge_specimens(st, spec)
    assert set(merged) == {"spA|spA_1", "spA|spA_2"} and merged["spA|spA_1"]["eye"] == "present"
    species = {spec[s][0]: sp[s] for s in spec}
    rows, work, per, _ = cc.check(merged, chars, species, sex_of={"spA|spA_2": "female", "spA|spA_1": "male"},
                                  structure_sex={"aedeagus": "male"})
    assert not [w for w in work if w["character"] == "aedeagus"]          # not expected in the female
    assert not [w for w in work if w["character"] in ("eye", "antenna")]  # merged: no per-image gaps


def test_via_bookkeeping_is_not_a_character_and_summaries_are_kept(tmp_path):
    import biorag_coded_states_from_coco_v1 as cs
    d = data()
    for a in d["annotations"]:
        a["attributes"] = {"via_attribute_group": "female terminalia" if a["image_id"] % 2 else "male terminalia"}
    (tmp_path / "c.json").write_text(json.dumps(d))
    gl = tmp_path / "gl.csv"
    pd.DataFrame({"filename": [f"{n}.tif" for n in ("spA_1", "spA_2", "spB_1", "spB_2", "spC_1", "spC_2")],
                  "group_label": ["A", "A", "B", "B", "C", "C"]}).to_csv(gl, index=False)
    df, _ = cs.read_states([str(tmp_path / "c.json")], str(gl), {}, {"characters": {}})
    assert not df["character"].str.startswith("via_").any()
    # a minimal matrix folder: the converter must carry its other files and rebuild the summaries
    md = tmp_path / "m"; md.mkdir()
    sids = [f"{g}_{i}" for g in "ABC" for i in (1, 2)]
    pd.DataFrame({"species": [s[0] for s in sids], "specimen_id": sids, "sex": "unknown", "category": "pronotum",
                  "base_category": "pronotum", "column": "length_mm", "feature_id": "pronotum.length_mm",
                  "tier": "key", "family": "length", "value": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1], "n_images": 1}
                 ).to_csv(md / "specimen_matrix_long.csv", index=False)
    pd.DataFrame([{"feature_id": "pronotum.length_mm", "category": "pronotum", "base_category": "pronotum",
                   "column": "length_mm", "tier": "key", "family": "length", "label": "pronotum length",
                   "unit": "mm"}]).to_csv(md / "feature_dictionary.tsv", sep="\t", index=False)
    (md / "x_diagnostic_report.json").write_text("{}")
    import biorag_feature_policy as pol
    import unittest.mock as um
    with um.patch.object(pol, "specimen_id", lambda stem, sp, prof: f"{sp}_{stem.split('_')[1]}"):
        sys.argv = ["x", "--coco", str(tmp_path / "c.json"), "--group_labels", str(gl), "--matrix_dir", str(md),
                    "--out_dir", str(tmp_path / "o")]
        cs.main()
    for f in ("species_feature_summary.csv", "coverage_species_by_structure.tsv", "x_diagnostic_report.json"):
        assert (tmp_path / "o" / f).exists(), f
    ss = pd.read_csv(tmp_path / "o/species_feature_summary.csv")
    assert "eye.coded_presence__absent" in set(ss["feature_id"])
