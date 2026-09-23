#!/usr/bin/env python3
"""
Tests for the BioRAG v2 evidence policy, key builder and checkers.
Synthetic data only — no API calls, no external files.

Run:  python measure/tests/test_biorag_v2.py        (plain python)
  or: python -m pytest measure/tests/test_biorag_v2.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biorag_feature_policy as pol  # noqa: E402
import biorag_key_builder_v1 as kb  # noqa: E402
from biorag_llm_backend import load_prompt_library, parse_json_response  # noqa: E402


# ── policy ─────────────────────────────────────────────────────────────────
def test_key_tier_is_a_whitelist():
    assert pol.classify_column("meas_length_mm")[0] == pol.TIER_KEY
    assert pol.classify_column("ratio_MF/MT")[0] == pol.TIER_KEY
    assert pol.classify_column("lmkrel_3_8")[0] == pol.TIER_KEY
    assert pol.classify_column("cie_L")[0] == pol.TIER_KEY
    for col in ("shape_PC1", "colhom_phylo_UMAP2", "tex_r1c2_glcm_contrast", "lmk_dist_8_15",
                "shape_centroid_size_mm", "colhom_phylo_silhouette", "imd_to_LAB2_centroid_mm",
                "color_adaptive_hue_mean", "some_brand_new_statistic"):
        assert pol.classify_column(col)[0] == pol.TIER_REMARKS, col
    assert pol.classify_column("meas_orientation_degrees")[0] == pol.TIER_EXCLUDE
    assert pol.classify_column("colhom_chroma_ab")[0] == pol.TIER_EXCLUDE


def test_opencv_lab_conversion():
    enc = pol.detect_lab_encoding([10, 200, 255], [120, 130, 140])
    assert math.isclose(enc["L_scale"], 100 / 255) and enc["ab_offset"] == 128
    L, a, b, C, h = pol.cie_from_encoded(255, 128, 228, enc)
    assert math.isclose(L, 100) and math.isclose(a, 0) and math.isclose(b, 100)
    assert math.isclose(C, 100) and math.isclose(h, 90)
    enc2 = pol.detect_lab_encoding([10, 60, 95], [-5, 3, 10])
    assert enc2["L_scale"] == 1.0 and enc2["ab_offset"] == 0


def test_colour_words():
    assert pol.colour_name(10, 0, 0) == "blackish"
    assert pol.colour_name(90, 1, 2) == "whitish"
    assert "yellow" in pol.colour_name(85, 0, 50)
    assert pol.lightness_term(25) == "dark"


def test_tier2_text_detection():
    assert pol.find_tier2_terms("differs from sp3 (p = 0.012) in PC1")
    assert pol.find_tier2_terms("Procrustes distance 0.37")
    assert not pol.find_tier2_terms("metatibia 1.2x as long as metafemur, dark brown")


def test_number_checker():
    allowed = [0.412, 1.25, 6]
    assert pol.number_is_allowed("0.412", allowed)
    assert pol.number_is_allowed("0.41", allowed)          # coarser rounding of an allowed value
    assert not pol.number_is_allowed("0.52", allowed)
    assert pol.numbers_in("MP/PL ≤ 1.25x; n = 6; sp1 and LAB2") == ["1.25", "6"]


def test_specimen_ids():
    prof = {"specimen_id": {"number_regex": r'_?(\d+)_(?:forewing|head|rostrum|metaleg|te+rminalia)'}}
    assert pol.specimen_id("Project_sp22_sp22_morph_22_3_metaleg_m_ch00_SV.tif", "sp22", prof) == "sp22_3"
    assert pol.specimen_id("Project001_morph_acok3_metaleg_m_ch00_SV.tif", "acok", prof) == "acok_3"
    assert pol.specimen_sex("x_morph_1_head_f_ch00.tif", prof) == "female"


# ── prompts ────────────────────────────────────────────────────────────────
def test_prompt_library_and_json():
    lib = load_prompt_library()
    for sec in ("evidence_policy", "describe_category", "refine_treatment", "key_couplet_wording",
                "describe_category_user", "synthesis_diagnosis"):
        assert sec in lib.sections, sec
    txt = lib.get("key_couplet_wording", taxon_context="X")
    assert '"lead_a"' in txt and "{" in txt
    assert "Genus" not in lib.raw("evidence_policy")          # universal file stays taxon-agnostic
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Here: {"a": 2} trailing {"b": 3}') == {"a": 2}


# ── key builder ────────────────────────────────────────────────────────────
def synthetic_matrix(n_species=9, n_spec=6, seed=1):
    rng = np.random.default_rng(seed)
    rows, fd = [], []
    feats = {"tibia.length_mm": ("tibia", "meas_length_mm", "length", "mm", "both"),
             "tibia.aspect_ratio": ("tibia", "meas_aspect_ratio", "aspect_ratio", "x", "both"),
             "proportions.ratio_MF/MT": ("proportions", "ratio_MF/MT", "ratio", "x", "both"),
             "paramere.length_mm": ("paramere", "meas_length_mm", "length", "mm", "male")}
    for s in range(n_species):
        sp = f"sp{s + 1}"
        for k in range(n_spec):
            sid = f"{sp}_{k + 1}"
            sex = "male" if k % 2 == 0 else "female"
            vals = {"tibia.length_mm": 0.30 + 0.05 * s + rng.normal(0, 0.004),
                    "tibia.aspect_ratio": 5.0 + 0.4 * ((s * 7) % n_species) + rng.normal(0, 0.03),
                    "proportions.ratio_MF/MT": 0.6 + 0.03 * ((s * 4) % n_species) + rng.normal(0, 0.002)}
            if sex == "male":
                vals["paramere.length_mm"] = 0.2 + 0.02 * s + rng.normal(0, 0.002)
            for fid, v in vals.items():
                cat, col, fam, unit, fsex = feats[fid]
                rows.append({"species": sp, "specimen_id": sid, "sex": sex, "category": cat,
                             "base_category": cat, "column": col, "feature_id": fid, "tier": "key",
                             "family": fam, "value": v, "n_images": 1})
    for fid, (cat, col, fam, unit, fsex) in feats.items():
        fd.append({"feature_id": fid, "category": cat, "base_category": cat, "column": col, "tier": "key",
                   "family": fam, "label": fid, "definition": fid, "unit": unit, "structure_sex": fsex,
                   "section": "x", "key_priority": 1, "n_species": n_species, "n_specimens": 0,
                   "conversion": ""})
    return pd.DataFrame(rows), pd.DataFrame(fd)


def test_key_reaches_every_species_once_and_thresholds_hold():
    long, fdict = synthetic_matrix()
    M = kb.Matrix(long, fdict, {})
    recs = kb.build_records(kb.KeyBuilder(M, {}).build(), M)
    steps = kb.paths_to_terminals(recs)
    assert sorted(steps) == sorted(M.species)
    assert all(len(v) == 1 for v in steps.values())
    assert len(recs) == len(M.species) - 1
    assert kb.recheck_thresholds(recs, M) == []
    ok = [kb.identify(v, M.spec_sex.get(s), recs)[0] == M.spec_species[s] for s, v in M.values.items()]
    assert np.mean(ok) > 0.95
    assert 0 < kb.e_dicho({s: v[0] for s, v in steps.items()}) <= 1


def test_nice_threshold_inside_gap():
    t, txt = kb.nice_threshold(0.412, 0.431, "mm")
    assert 0.412 < t < 0.431 and txt == "0.42"


def test_wording_checker_rejects_bad_wording():
    long, fdict = synthetic_matrix()
    M = kb.Matrix(long, fdict, {})
    recs = kb.build_records(kb.KeyBuilder(M, {}).build(), M)
    r = recs[0]
    c = r["characters"][0]
    good = r["A_template"]
    assert kb.check_wording(good, r["characters"], [], "A") == [] or \
        all("abbreviation" in p for p in kb.check_wording(good, r["characters"], [], "A"))
    bad = good.replace(c["threshold_text"], "9.99")
    assert any("missing" in p or "not in couplet" in p for p in kb.check_wording(bad, r["characters"], [], "A"))
    assert kb.check_wording(good + " (PC1 = 2.1)", r["characters"], [], "A")


def test_subjective_colour_checks():
    import biorag_subjective_checks_v1 as sc
    m = {"cie_L": 20.0, "cie_a": 10.0, "cie_b": 20.0, "cie_L_darkest_third": 10.0,
         "cie_L_palest_third": 30.0, "n": 3, "C": math.hypot(10, 20), "h": 63.4}
    assert sc.judge_colour("metafemur dark brown", m)[0] == "consistent"
    assert sc.judge_colour("metafemur pale yellow", m)[0] == "contradicts"
    assert sc.judge_colour("paler than in D. sp. 3", m)[0] == "not_verifiable"
    assert sc.judge_colour("with a pale apex", m)[0] == "check_image"
    assert sc.judge_colour("mottled with brown patches", m)[0] in ("consistent", "not_verifiable")
    assert sc.judge_colour("dark brown", None)[0] == "not_verifiable"
    namer = sc.ColourNamer(sc.DEFAULT_LOOKUP_SCRIPT)
    assert namer.name(0, 0, 0)[0] == "black"
    assert namer.name(100, 0, 0)[0] in ("white", "snow", "ghost white")


# ── independent confabulation audit (checker v2) ────────────────────────────

def _checker_env(tmp: Path):
    """Tiny two-species matrix + profile for the audit tests."""
    import biorag_confabulation_checker_v2 as ck
    rows = []
    for sp, tib, lab in (("spA", [0.50, 0.54, 0.58], [0.20, 0.21, 0.22]),
                         ("spB", [0.80, 0.84, 0.88], [0.30, 0.31, 0.32])):
        for i, (t, l) in enumerate(zip(tib, lab), 1):
            rows += [{"species": sp, "specimen_id": f"{sp}_{i}", "sex": "male", "category": "tibia",
                      "base_category": "tibia", "column": "meas_length_mm", "feature_id": "tibia.length_mm",
                      "tier": "key", "family": "length", "value": t, "n_images": 1},
                     {"species": sp, "specimen_id": f"{sp}_{i}", "sex": "male", "category": "LAB1",
                      "base_category": "LAB1", "column": "meas_length_mm", "feature_id": "LAB1.length_mm",
                      "tier": "key", "family": "length", "value": l, "n_images": 1}]
    pd.DataFrame(rows).to_csv(tmp / "specimen_matrix_long.csv", index=False)
    fd = [{"feature_id": "tibia.length_mm", "category": "tibia", "base_category": "tibia",
           "column": "meas_length_mm", "tier": "key", "family": "length", "label": "metatibia length",
           "definition": "", "unit": "mm", "structure_sex": "both", "section": "Metaleg", "key_priority": 1,
           "n_species": 2, "n_specimens": 6, "conversion": ""},
          {"feature_id": "LAB1.length_mm", "category": "LAB1", "base_category": "LAB1",
           "column": "meas_length_mm", "tier": "key", "family": "length",
           "label": "median labial segment length", "definition": "", "unit": "mm", "structure_sex": "both",
           "section": "Rostrum", "key_priority": 1, "n_species": 2, "n_specimens": 6, "conversion": ""}]
    pd.DataFrame(fd).to_csv(tmp / "feature_dictionary.tsv", sep="\t", index=False)
    prof = tmp / "profile.yaml"
    prof.write_text(
        'profile_version: 1\n'
        'taxon: {genus: Testia, family: Testidae}\n'
        'species:\n  spA: {name: "Testia alpha", status: undescribed}\n'
        '  spB: {name: "Testia beta", status: undescribed}\n'
        'structures:\n'
        '  tibia: {term: "metatibia", section: "Metaleg", sex: both}\n'
        '  LAB1:  {term: "median labial segment", section: "Rostrum", sex: both}\n'
        'description_sections: ["Rostrum", "Metaleg"]\n')
    import argparse
    return ck, ck.Evidence(argparse.Namespace(taxon_profile=str(prof), matrix_dir=str(tmp),
                                              key_tree=None, localities=None))


def _audit(ck, ev, text, citations, section="Metaleg"):
    treat = {"description": [{"section": section, "text": text}], "citations": citations,
             "diagnosis": "", "sexual_dimorphism": "", "remarks": ""}
    return ck.audit_species("spA", treat, "", ev)["records"]


def test_audit_accepts_correct_numbers(tmp_path=None):
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    ck, ev = _checker_env(tmp)
    recs = _audit(ck, ev, "Metatibia 0.50–0.58 mm long (mean 0.54; n=3).",
                  [{"value": "0.50–0.58 mm", "id": "tibia.length_mm"}])
    assert [r["status"] for r in recs if r["kind"] == "measurement"] == ["ok"]


def test_audit_catches_column_confusion(tmp_path=None):
    """A real value of another structure, printed next to this one."""
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    ck, ev = _checker_env(tmp)
    recs = _audit(ck, ev, "Metatibia 0.20–0.22 mm long (n=3).",
                  [{"value": "0.20–0.22 mm", "id": "tibia.length_mm"}])
    m = [r for r in recs if r["kind"] == "measurement"][0]
    assert m["status"] == "error" and m["type"] == "column_confusion"


def test_audit_catches_fabrication_and_sample_inflation(tmp_path=None):
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    ck, ev = _checker_env(tmp)
    m = [r for r in _audit(ck, ev, "Metatibia 1.90–2.10 mm long (n=3).",
                           [{"value": "1.90–2.10 mm", "id": "tibia.length_mm"}])
         if r["kind"] == "measurement"][0]
    assert m["status"] == "error" and m["type"] == "value_fabrication"
    m = [r for r in _audit(ck, ev, "Metatibia 0.50–0.58 mm long (mean 0.54; n=9).",
                           [{"value": "0.50–0.58 mm", "id": "tibia.length_mm"}])
         if r["kind"] == "measurement"][0]
    assert m["status"] == "error" and m["type"] == "sample_inflation"


def test_audit_checks_comparisons_against_the_matrix(tmp_path=None):
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    ck, ev = _checker_env(tmp)
    good = [r for r in _audit(ck, ev, "Metatibia 0.50–0.58 mm long (n=3), smaller than in Testia beta.",
                              [{"value": "0.50–0.58 mm", "id": "tibia.length_mm"}])
            if r["kind"] == "comparison"][0]
    assert good["status"] == "ok"
    bad = [r for r in _audit(ck, ev, "Metatibia 0.50–0.58 mm long (n=3), greater than in Testia beta.",
                             [{"value": "0.50–0.58 mm", "id": "tibia.length_mm"}])
           if r["kind"] == "comparison"][0]
    assert bad["status"] == "error" and bad["type"] == "fabricated_comparison"


def test_audit_flags_statistics_and_identifiers(tmp_path=None):
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    ck, ev = _checker_env(tmp)
    recs = _audit(ck, ev, "Metatibia 0.50–0.58 mm long (n=3) [tibia.length_mm]; PC1 separates it.",
                  [{"value": "0.50–0.58 mm", "id": "tibia.length_mm"}])
    kinds = {(r["kind"], r["type"]) for r in recs}
    assert ("wording", "statistic_outside_remarks") in kinds
    assert ("format", "internal_identifier_in_text") in kinds


def test_category_aliases_are_merged(tmp_path=None):
    """One structure under two names (punctuation stripped) is merged."""
    import biorag_key_feature_filter_v2 as f2
    df = pd.DataFrame([
        {"image_base": "img1", "category": "cell-c+sc", "group_label": "spA", "meas_length_mm": 1.0,
         "cie_L": np.nan, "has_measurement": True, "has_color": False},
        {"image_base": "img1", "category": "cell-csc", "group_label": "spA", "meas_length_mm": np.nan,
         "cie_L": 55.0, "has_measurement": False, "has_color": True},
    ])
    report = {}
    out = f2.merge_category_aliases(df.copy(), {"category_aliases": {"cell-csc": "cell-c+sc"}}, report)
    assert list(out["category"]) == ["cell-c+sc"] and len(out) == 1
    assert out.iloc[0]["meas_length_mm"] == 1.0 and out.iloc[0]["cie_L"] == 55.0
    auto = f2.merge_category_aliases(df.copy(), {}, {})
    assert len(auto) == 1                      # detected without a profile entry


# ── ontology annotation (annotator v2) ──────────────────────────────────────

def test_ontology_lookup_and_broader_terms(tmp_path=None):
    import biorag_ontology_annotator_v2 as oa
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    (tmp / "toy.obo").write_text(
        "format-version: 1.2\ndata-version: releases/2026-01-01\n\n"
        "[Term]\nid: TOY:0000001\nname: metatibia\n\n"
        "[Term]\nid: TOY:0000002\nname: labium\n\n"
        "[Term]\nid: TOY:0000003\nname: length\n\n"
        "[Term]\nid: TOY:0000004\nname: epiproct\nsynonym: \"proctiger\" EXACT []\n\n"
        "[Typedef]\nid: part_of\n")
    onto = oa.Ontologies(tmp, ["toy"])
    assert onto.versions["toy"] == "releases/2026-01-01"
    assert onto.lookup("metatibia")[:3] == ("TOY:0000001", "metatibia", "exact")
    assert onto.lookup("proctiger")[2] == "exact_synonym"
    assert onto.lookup("circumanal ring") is None
    prof = {"structures": {"tibia": {"term": "metatibia"},
                           "LAB1": {"term": "median labial segment", "ontology_broader": "TOY:0000002"},
                           "ring": {"term": "circumanal ring", "ontology_broader": "TOY:0000999"}},
            "ontology_entity_prefixes": ["TOY"]}
    rep = {}
    assert oa.entity_for("tibia", prof, onto, rep)["match"] == "exact"
    assert oa.entity_for("LAB1", prof, onto, rep)["match"] == "broader"
    assert oa.entity_for("ring", prof, onto, rep) is None       # invalid CURIE is reported, not used
    assert rep["invalid_curies"]
    assert oa.quality_for("meas_length_mm", onto, rep)["curie"] == "TOY:0000003"



# ── novelty scoring ────────────────────────────────────────────────────────

def _novelty_env(tmp: Path):
    """Three tight species plus one clearly different series."""
    import biorag_novelty_score_v1 as nv
    rows = []
    base = {"spA": 1.00, "spB": 1.30, "spC": 1.60, "spX": 4.00}
    for sp, mu in base.items():
        for i in range(1, 5):
            for f, off in (("tibia.length_mm", 0.0), ("tibia.height_mm", 0.6), ("tibia.aspect_ratio", 0.5),
                           ("LAB1.length_mm", 0.2), ("LAB1.height_mm", 0.7), ("LAB1.aspect_ratio", 0.1),
                           ("whole_head.length_mm", 0.3), ("whole_head.height_mm", 0.8),
                           ("whole_head.aspect_ratio", 0.4)):
                rows.append({"species": sp, "specimen_id": f"{sp}_{i}", "sex": "male",
                             "category": f.split(".")[0], "base_category": f.split(".")[0],
                             "column": f.split(".")[1], "feature_id": f,
                             "tier": "key", "family": "length" if f.endswith("_mm") else "aspect_ratio",
                             "value": mu + off + 0.01 * i, "n_images": 1})
    pd.DataFrame(rows).to_csv(tmp / "specimen_matrix_long.csv", index=False)
    fd = [{"feature_id": f, "category": f.split(".")[0], "base_category": f.split(".")[0],
           "column": f.split(".")[1], "tier": "key",
           "family": "length" if f.endswith("_mm") else "aspect_ratio", "label": f, "definition": "",
           "unit": "mm" if f.endswith("_mm") else "x", "structure_sex": "both", "section": "S",
           "key_priority": 1, "n_species": 4, "n_specimens": 16, "conversion": ""}
          for f in ("tibia.length_mm", "tibia.height_mm", "tibia.aspect_ratio",
                    "LAB1.length_mm", "LAB1.height_mm", "LAB1.aspect_ratio",
                    "whole_head.length_mm", "whole_head.height_mm", "whole_head.aspect_ratio")]
    pd.DataFrame(fd).to_csv(tmp / "feature_dictionary.tsv", sep="\t", index=False)
    prof = {"taxon": {"genus": "Testia"}, "species": {k: {"name": f"Testia {k}"} for k in base},
            "structures": {}}
    return nv, nv.Reference(tmp, prof)


def test_novelty_series_gap_separates_a_distant_series(tmp_path=None):
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    nv, ref = _novelty_env(tmp)
    ids_x = [f"spX_{i}" for i in range(1, 5)]            # a series the reference has never seen
    ids_a = [f"spA_{i}" for i in range(1, 3)]            # a sub-series of a known species
    far = nv.group_score(ref, "size", ids_x, exclude_species=("spX",))
    near = nv.group_score(ref, "size", ids_a)             # the rest of spA stays in the reference
    assert near["nearest"] == "spA"
    assert far["G"] > near["G"] * 3, (far["G"], near["G"])


def test_novelty_thresholds_hit_the_requested_error_rate(tmp_path=None):
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    nv, ref = _novelty_env(tmp)
    cal = nv.calibrate(ref, nv.Descriptive(None), fpr=0.05)
    for name, c in cal.items():
        assert c["observed_fpr"] <= 0.25, (name, c["observed_fpr"])
        assert c["threshold"] > 0


def test_descriptive_states_are_ordinal_and_graded(tmp_path=None):
    """smooth -> rugose must count as a bigger step than smooth -> alutaceous."""
    import biorag_novelty_score_v1 as nv
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    f = tmp / "flags.tsv"
    f.write_text("species\tfield\tstructure\tkind\twords\n"
                 "spA\tdescription\ttibia\ttexture\tsmooth\n"
                 "spB\tdescription\ttibia\ttexture\talutaceous\n"
                 "spC\tdescription\ttibia\ttexture\trugose\n"
                 "spD\tdescription\ttibia\ttexture\tdeeply, rugose\n")
    d = nv.Descriptive(f)
    assert d.ok
    near = d.compare("spA", "spB")["d"]
    far = d.compare("spA", "spC")["d"]
    assert far > near > 0
    assert d.compare("spA", "spD")["d"] > far        # the modifier adds half a step
    assert "steps on a" in d.compare("spA", "spC")["differences"][0]
    assert d.compare("spA", "spA")["d"] == 0



def test_descriptive_matrix_becomes_a_character_set(tmp_path=None):
    """Per-specimen states -> numeric columns: ordinal scaled 0-1, nominal one-hot,
    'not assessable' left missing."""
    import biorag_novelty_score_v1 as nv
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    f = tmp / "states.tsv"
    f.write_text(
        "specimen_id\tspecies\tcategory\tstructure\tcharacter\ttype\tstate\timage\n"
        "spA_1\tspA\ttibia\tmetatibia\tsurface roughness\tordinal\tsmooth\ti1\n"
        "spA_2\tspA\ttibia\tmetatibia\tsurface roughness\tordinal\tpunctate\ti2\n"
        "spB_1\tspB\ttibia\tmetatibia\tsurface roughness\tordinal\trugose\ti3\n"
        "spB_1\tspB\ttibia\tmetatibia\toutline shape\tnominal\tfalcate\ti3\n"
        "spA_1\tspA\ttibia\tmetatibia\toutline shape\tnominal\tclavate\ti1\n"
        "spA_2\tspA\ttibia\tmetatibia\toutline shape\tnominal\tnot assessable\ti2\n")
    t = nv.descriptive_table(f, nv.DESCRIPTIVE_CHARACTERS)
    col = "tibia|surface roughness"
    assert col in t.columns
    assert 0.0 <= t.loc["spA_1", col] < t.loc["spA_2", col] < t.loc["spB_1", col] <= 1.0
    onehot = [c for c in t.columns if c.startswith("tibia|outline shape=")]
    assert len(onehot) == len(nv.canonical_states(nv.DESCRIPTIVE_CHARACTERS["outline shape"]))
    assert t.loc["spA_1", "tibia|outline shape=clavate"] == 1.0
    assert t.loc["spB_1", "tibia|outline shape=clavate"] == 0.0
    assert t.loc["spA_2", onehot].isna().all()        # not assessable stays missing



def test_coarse_bands_merge_neighbouring_states(tmp_path=None):
    """The coarse table scores the named band, not the exact word: two specimens
    one step apart inside a band become identical, across a band stay apart."""
    import biorag_novelty_score_v1 as nv
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    f = tmp / "states.tsv"
    f.write_text(
        "specimen_id\tspecies\tcategory\tstructure\tcharacter\ttype\tstate\timage\n"
        "spA_1\tspA\ttibia\tmetatibia\tsurface roughness\tordinal\tpolished\ti1\n"
        "spA_2\tspA\ttibia\tmetatibia\tsurface roughness\tordinal\tsmooth\ti2\n"
        "spB_1\tspB\ttibia\tmetatibia\tsurface roughness\tordinal\trugose\ti3\n"
        "spB_1\tspB\ttibia\tmetatibia\toutline shape\tnominal\tfalcate\ti3\n")
    col = "tibia|surface roughness"
    fine = nv.descriptive_table(f, nv.DESCRIPTIVE_CHARACTERS)
    assert fine.loc["spA_1", col] < fine.loc["spA_2", col]        # one step apart
    coarse = nv.descriptive_table(f, nv.DESCRIPTIVE_CHARACTERS, coarse=True)
    assert coarse.loc["spA_1", col] == coarse.loc["spA_2", col]   # same band
    assert coarse.loc["spB_1", col] > coarse.loc["spA_1", col]    # different band
    assert set(nv.coarse_map("surface roughness").values()) == \
        set(nv.COARSE_BANDS["surface roughness"])


def test_reliability_gate_drops_flagged_characters(tmp_path=None):
    """A character the retest could not reproduce is kept out of the table."""
    import biorag_novelty_score_v1 as nv
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    f = tmp / "states.tsv"
    f.write_text(
        "specimen_id\tspecies\tcategory\tstructure\tcharacter\ttype\tstate\timage\n"
        "spA_1\tspA\ttibia\tmetatibia\tsurface roughness\tordinal\tsmooth\ti1\n"
        "spA_1\tspA\ttibia\tmetatibia\toutline shape\tnominal\tfalcate\ti1\n")
    both = nv.descriptive_table(f, nv.DESCRIPTIVE_CHARACTERS)
    assert any(c.startswith("tibia|outline shape") for c in both.columns)
    gated = nv.descriptive_table(f, nv.DESCRIPTIVE_CHARACTERS,
                                 allowed={"surface roughness"})
    assert list(gated.columns) == ["tibia|surface roughness"]


def test_every_band_member_is_a_known_state():
    """A band may only contain words the character actually allows (or their
    synonyms) — a typo in COARSE_BANDS would silently drop specimens."""
    import biorag_novelty_score_v1 as nv
    for character, bands in nv.COARSE_BANDS.items():
        cfg = nv.DESCRIPTIVE_CHARACTERS[character]
        known = set(nv.state_ranks(cfg))
        for band, members in bands.items():
            unknown = [m for m in members if m.lower() not in known]
            assert not unknown, (character, band, unknown)
        covered = {m.lower() for ms in bands.values() for m in ms}
        assert known <= covered, (character, sorted(known - covered))


def test_ablation_table_reads_the_four_runs(tmp_path=None):
    """The investigation table is built from the calibration files of the runs that exist,
    in the order the argument is made, and skips the ones that were not run."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("rf", Path(__file__).resolve().parent.parent /
                                                  "biorag_reliability_figures_v1.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    tmp = Path(tmp_path or __import__("tempfile").mkdtemp())
    for run, tpr in (("ungated", 0.181), ("coarse_only", 0.067), ("gate_only", 0.329)):
        d = tmp / run
        d.mkdir(parents=True)
        (d / "calibration.json").write_text(json.dumps(
            {"descriptive": {"tpr": tpr, "observed_fpr": 0.056, "threshold": 2.0}}))
        (d / "calibration_series.json").write_text(json.dumps({"descriptive": {"tpr": tpr / 2}}))
    t = rf.read_ablation(tmp)
    assert list(t["run"]) == ["ungated", "coarse_only", "gate_only"]      # declared order, missing skipped
    assert t.set_index("run").loc["gate_only", "unseen_species_caught_specimen"] == 0.329
    assert t["unseen_species_caught_series"].notna().all()


def _gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cg", Path(__file__).resolve().parent.parent / "biorag_character_gate_v1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gate_removes_only_clauses_that_say_nothing_else():
    """A clause that is only the unreliable character goes whole; in a clause that says
    more the statement is removed with its connector, never cut into "Cell cu1 to"."""
    cg = _gate()
    import biorag_novelty_score_v1 as nv
    rel = {"use": ["vestiture"], "use_coarse": ["surface roughness"], "flag": ["outline shape"]}
    action, bands = cg.decisions(nv.DESCRIPTIVE_CHARACTERS, rel)
    assert action.get("falcate") == "strike"
    vocab = set(cg.vocabulary(nv.DESCRIPTIVE_CHARACTERS))
    edits = []
    out = cg.edit_block("Paramere falcate; disc setose. Cell cu1 falcate to cuneate.",
                        action, bands, vocab, edits, {"species": "x", "field": "d", "section": ""})
    assert "falcate" not in out.lower()                 # every statement about it is gone
    assert "disc setose" in out                         # its neighbour survives untouched
    assert "Cell cu1 cuneate" in out                    # the coordination collapsed cleanly
    assert " to cuneate" not in out and "cu1 to" not in out          # no dangling connector
    assert {e["action"] for e in edits} == {"clause removed", "statement removed"}


def test_gate_never_leaves_a_clause_hanging():
    """Deleting a state word must not strand a connector, an adverb or a bracket; where it
    would, the clause is reported for rewriting instead of being damaged."""
    cg = _gate()
    mods = "|".join(sorted(__import__("biorag_novelty_score_v1").MODIFIERS, key=len, reverse=True))
    assert cg.delete_word("smooth to weakly striate surface", "striate", mods).split() == \
        ["smooth", "surface"]
    assert cg.delete_word("cuneate to broadly triangular", "triangular", mods).strip() == "cuneate"
    assert cg.delete_word("strongly arcuate (falcate)", "falcate", mods).strip() == "strongly arcuate"
    assert cg.delete_word("(globular/triangular/elongate)", "triangular", mods) == "(globular/elongate)"
    # a clause that already ended in an adverb is not damaged by the edit
    assert not cg.dangling("slightly compressed dorso-ventrally",
                           "subcylindrical and slightly compressed dorso-ventrally")
    assert cg.dangling("cuneate to broadly", "cuneate to broadly triangular")


def test_gate_leaves_band_level_characters_alone_by_default():
    """The bands score a character; they do not reword a description. A character that
    passes at band level keeps the words it was given."""
    cg = _gate()
    import biorag_novelty_score_v1 as nv
    action, _ = cg.decisions(nv.DESCRIPTIVE_CHARACTERS,
                             {"use": [], "use_coarse": ["apex"], "flag": []})
    assert action.get("acuminate") == "coarsen"        # the machinery knows the band ...
    gated = {w: ("keep" if v == "coarsen" else v) for w, v in action.items()}   # ... default drops it
    assert gated["acuminate"] == "keep"


def test_gate_word_shared_by_two_characters_is_left_alone():
    """crenulate is both a surface pattern and a kind of margin incision: it may only be
    struck when every character it could belong to was flagged."""
    cg = _gate()
    import biorag_novelty_score_v1 as nv
    assert len(cg.vocabulary(nv.DESCRIPTIVE_CHARACTERS)["crenulate"]) > 1
    action, _ = cg.decisions(nv.DESCRIPTIVE_CHARACTERS,
                             {"use": ["margin incision"], "use_coarse": [], "flag": ["surface pattern"]})
    assert action["crenulate"] == "keep"
    action2, _ = cg.decisions(nv.DESCRIPTIVE_CHARACTERS,
                              {"use": [], "use_coarse": [],
                               "flag": ["surface pattern", "margin incision"]})
    assert action2["crenulate"] == "strike"


def test_rewrite_must_be_a_deletion():
    """The model editing a sentence may only remove words: a reply that introduces a value,
    a name or a new claim is refused, so nothing can be confabulated during the edit."""
    cg = _gate()
    before = "Paramere falcate, apex acuminate, length 0.24 mm."
    assert cg.is_deletion_only(before, "Paramere, apex acuminate, length 0.24 mm.")
    assert cg.is_deletion_only(before, "")
    assert not cg.is_deletion_only(before, "Paramere straight, apex acuminate, length 0.24 mm.")
    assert not cg.is_deletion_only(before, "Paramere falcate, apex acuminate, length 0.28 mm.")
    assert not cg.is_deletion_only(before, "apex acuminate, paramere falcate.")   # reordered


def test_gate_refuses_an_edit_that_removes_more_than_it_was_asked_to():
    """The subsequence check stops a model ADDING; it does not stop it deleting a neighbouring
    statement, and the numeric audit cannot see that because no number changes. Guard 4 does."""
    cg = _gate()
    before = "Vertex punctate and shining, genal processes falcate, length 0.24 mm."
    keepable = {"punctate", "shining"}
    asked = "Vertex punctate and shining, genal processes, length 0.24 mm."
    greedy = "Vertex, genal processes, length 0.24 mm."
    assert cg.is_deletion_only(before, greedy)            # a subsequence: guard 1 lets it through
    assert cg.collateral(before, asked, keepable) == []   # removed only what was asked
    assert set(w.lower() for w in cg.collateral(before, greedy, keepable)) == {"punctate", "shining"}


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    print(f"{'ALL PASSED' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
