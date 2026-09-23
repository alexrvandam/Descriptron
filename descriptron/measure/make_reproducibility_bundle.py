#!/usr/bin/env python3
"""
make_reproducibility_bundle.py — collect everything needed to repeat the work
============================================================================

Copies the BioRAG v2 scripts, prompts, schemas, ontology releases and tests into
one folder, with a manifest (SHA-256 of every file), the package versions of the
environment that produced the results, and a README that says what each script
is for and the order to run them in. Nothing is modified in place.

  python make_reproducibility_bundle.py --out_dir ~/Desktop/biorag_v2_bundle \\
      [--include_paper] [--results "/media/.../Diaphorina_monograph"]

--include_paper  also copies the figure and manuscript scripts
--results        also copies the small result files (reports, calibrations,
                 matrices) — not the images or the DOCX
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../gui/measure
GUI = HERE.parent

# what each file is for (the rest are copied as supporting modules)
PURPOSE = {
    # --- the v2 pipeline --------------------------------------------------
    "biorag_feature_policy.py": "evidence tiers, units, colour words, number checks (shared by everything)",
    "biorag_llm_backend.py": "LLM backends: Anthropic API or a Claude Code subscription; prompt loader; call log",
    "biorag_key_feature_filter_v2.py": "builds the Tier-1 (hand-checkable) data matrix and repairs the known "
                                       "compilation faults; outlier and label-swap screen",
    "biorag_key_builder_v1.py": "computes the identification key from the matrix; the model only words the "
                                "couplets; validation, JSON-LD and SDD export",
    "biorag_description_refiner_v1.py": "writes the species treatments from a deterministic data sheet; "
                                        "--repair_existing runs the audit and repairs only what it flags",
    "biosyslit_rag_retrieval_v2.py": "image-based per-structure description with the v2 prompts and tiers",
    # --- evaluation -------------------------------------------------------
    "biorag_confabulation_checker_v2.py": "EVALUATION: independent audit of the treatments — every number "
                                          "attributed to one feature, every comparison re-derived, every "
                                          "Remarks statement traced to its source",
    "biorag_subjective_checks_v1.py": "EVALUATION: colour words against the measured colour (KD-tree names), "
                                      "and texture/shape/setation words against the per-specimen states",
    "biorag_descriptive_scoring_v1.py": "scores the descriptive characters per specimen from the "
                                        "isolated-structure images (one state per character, no free text)",
    "biorag_character_reliability_v1.py": "EVALUATION: how repeatable is each descriptive character "
                                          "(test-retest, fine / coarse band) and does it agree with the "
                                          "measured metrics? flags the ones not to lean on for this taxon",
    "biorag_character_gate_v1.py": "EVALUATION: reliability gate on the wording — statements whose "
                                  "state a second reading of the same image does not reproduce are "
                                  "removed from the diagnoses and descriptions (not because they are "
                                  "wrong, but because they cannot be offered as confirmable "
                                  "observations); rules or model, the model may only delete",
    "biorag_graph_identify_v1.py": "EVALUATION: identification and novelty from the PUBLISHED treatments "
                                  "read as assertions (constraint satisfaction), so key, graph and "
                                  "matrix can be compared on the same specimens",
    "biorag_outgroup_batch_check_v1.py": "EVALUATION: is an out-of-reference set separated by biology or "
                                        "by how it was imaged and annotated?",
    "biorag_key_fuzzy_v1.py": "EVALUATION: grades the same key instead of only branching — how "
                              "decisively each couplet was decided and how well the value fits the "
                              "range on the side taken; reports how far an identification can be "
                              "trusted, and shows why grading does NOT detect novelty",
    "biorag_calibrate_v1.py": "CALIBRATION: measures the cutoffs on the reference set in hand rather than inheriting them — leave-one-specimen-out accuracy by series size (read from the key builder's own hold-out, never resubstitution), leave-one-species-out scoring of four candidate novelty rules, and the character-reliability verdicts. Scores the key AND the character matrix on both hold-out arms (in-sample and nested margins, leave-one-out identification, agreement between the two), recommends which instrument to use for a name and which for a warning, and warns when the reference set is too thin to calibrate at all. Writes both arms per specimen so other scripts can re-score them. Pipeline step 22.5, before any treatment is written.",
    "biorag_calibration_figures_v1.py": "CALIBRATION: draws the calibration on two sheets — the key as a tree with every terminal coloured by its own leave-one-specimen-out accuracy (support values, in the phylogenetic sense), the accuracy to expect by series size, every novelty rule with its detection and false-alarm rate; and a character sheet showing which characters carry the key and which descriptive words survived the retest. Written so nobody has to read a TSV to judge a key.",
    "biorag_key_holdout_v1.py": "EVALUATION: rebuilds the key without each species in turn and runs "
                               "that species down it — does the key itself notice a taxon it has never "
                               "seen, or does it give a confident wrong answer?",
    "biorag_gate_compare_v1.py": "EVALUATION: runs the two gate engines side by side against the "
                                 "ungated text so the choice is made on evidence per taxon",
    "biorag_reliability_figures_v1.py": "EVALUATION: the character investigation as tables and figures "
                                        "(repeatability, congruence, the gating ablation, the character "
                                        "sets side by side) — what the Methods and Results cite",
    "biorag_ontology_annotator_v2.py": "entity-quality ontology annotation resolved from the OBO releases",
    # --- delimitation -----------------------------------------------------
    "biorag_novelty_score_v1.py": "PREDICTION: is a candidate outside every described species? five character "
                                  "sets, thresholds calibrated by error rate, series-level gap, key path",
    # --- housekeeping -----------------------------------------------------
    "biorag_make_taxon_profile_v1.py": "drafts a taxon profile from a COCO file and group labels",
    "biorag_rename_species_v1.py": "applies final species names across a finished monograph",
    "biorag_collaborator_forms_v1.py": "builds the workbook sent to collectors: species names, "
                                       "localities and type designations, pre-filled with what is "
                                       "already known",
    "biorag_type_material_v1.py": "turns the returned designations into the Code-compliant type "
                                  "statement, and refuses to write one that would leave the name "
                                  "unavailable (holotype / syntypes / lectotype and the rest)",
    "build_species_treatment_docx_v2.py": "assembles the monograph DOCX",
    "build_monograph_exports_v1.py": "TaxPub, Darwin Core Archives, SDD, JSON-LD, Markdown exports",
    "run_full_pipeline_v2.py": "orchestrator: runs the whole workflow, one step per script",
    "biosyslit_rag_retrieval.py": "the v1 describer: literature retrieval (FAISS over PDFs and Plazi "
                                  "treatments) and the image-reading calls; biosyslit_rag_retrieval_v2.py wraps "
                                  "it and cannot start without it",
    "biorag_key_feature_filter.py": "v1 feature filter; the v2 filter imports one function from it when a "
                                    "prior cache is cleaned (--cache_dir)",
    "biorag_specimen_id.py": "specimen identifier from an image file name (shared; imported by the "
                             "frame, screen and character scripts — without it they do not start)",
    "biorag_annotation_screen_v1.py": "SCREEN: does every polygon sit on the specimen? coverage, containment "
                                      "and COCO dimension checks; writes an exclusion list in the format the "
                                      "matrix builder reads, merged with the manual one (pipeline step 20.5)",
    "biorag_instrument_compare_v3.py": "EVALUATION: key, knowledge graph and character matrix on BOTH questions "
                                       "(naming a described specimen; recognising a species never seen), every "
                                       "operating point, nested choice of threshold, one denominator, agreement "
                                       "between instruments. Tables 1-2, S7a-d and Figure 2 of the paper",
    "biorag_instrument_compare_v2.py": "EVALUATION (superseded by v3, kept to reproduce the 19 Sep table): "
                                       "novelty only, on a 60-quantile threshold grid",
    "biorag_congruence_compare_v1.py": "EVALUATION: does an extra character set add to the matrix? exact sweep, "
                                       "nested margin, paired test, and the character-selection leak closed "
                                       "(characters proposed from a withheld species are dropped for its fold)",
    "biorag_add_discrete_characters_v1.py": "appends present/absent characters (coded at the bench or scored by a "
                                            "model) to a copy of the Tier-1 matrix and reports how many features "
                                            "the key builder will read before and after",
    "biorag_rerun_descriptive_tables_v1.sh": "recomputes Supplementary Tables S5-S6 (six character sets; the four-cell "
                                             "gating ablation) on the current matrix; no model calls",
    "biorag_frame_orientation_v1.py": "OPTIONAL ARM: which homology frames lie end-for-end (a Procrustes fit cannot "
                                      "tell the two ends of a near-symmetric elongate outline apart)",
    "biorag_homology_frame_v1.py": "OPTIONAL ARM: puts every specimen of a structure in one frame by a similarity "
                                   "transform to the Procrustes consensus (shape untouched); TPS on the grid only",
    "biorag_vlm_characters_v1.py": "OPTIONAL ARM: a vision-language model proposes present/absent characters from "
                                   "contrast sets, scores every specimen blind, reports where it looked, retests",
    "biorag_vlm_roi_figures_v1.py": "OPTIONAL ARM: each reported location ringed on the specimen, with its "
                                    "homologous cell",
    "biorag_annotation_reorient_v1.py": "REPAIR: an image rotated or mirrored after it was annotated - finds the rigid motion "
                                        "that puts its polygons (and landmarks) back on the specimen; nothing overwritten",
    "biorag_label_conflict_check_v1.py": "CHECK: is any photograph counted under two species in the compiled table? "
                                         "(exit status 1 if so; first step of the rebuild)",
    "biorag_rebuild_downstream_v1.sh": "ONE COMMAND for everything downstream of the compiled table, in the only valid order, "
                                       "stopping at the first failure; no model calls with backend 'none'",
    "biorag_character_robustness_v1.py": "EVALUATION: what ONE character is worth under the same two hold-outs "
                                         "(naming a withheld specimen; telling a described species from an unseen one)",
    "biorag_key_jackknife_support_v1.py": "EVALUATION: delete-one jackknife support for every couplet of the key - recovered in "
                                          "the rebuilt keys, and withheld specimens routed the right way; printed on the key sheet",
    "biorag_outline_shape_v1.py": "outline shape from the annotation polygons: mirror-aware, label-free registration and "
                                  "Procrustes coordinates re-derived per fold (the withheld item never shapes its own frame)",
    "biorag_computed_sets_v1.py": "per-specimen colour-pattern and texture tables from a compiled directory, for testing as "
                                  "extra character sets (batch ordinations excluded, and why)",
    "biorag_frame_orientation_calibrate_v1.py": "OPTIONAL ARM: ground truth for frame orientation from anatomical adjacency "
                                                "(the tibia end nearest the femur is proximal), scores the applied turns, "
                                                "cross-validates the thresholds, validates the per-image mirror vote",
    "biorag_tier2_summary_v1.py": "joins the novelty comparison, the identification results and the paired tests of a "
                                  "congruence run into the one table the paper prints (Table S7e)",
    "biorag_retest_compare_v1.py": "EVALUATION: the same images read more than once - exact agreement, Cohen's and "
                                   "weighted kappa, and the DIRECTION of the disagreements, for every pair of "
                                   "readings (tells within-session noise from between-session drift); no model calls",
    "biorag_methods_summary_figure_v1.py": "FIGURE 3: every method tried side by side (matrix, key crisp and graded, "
                                           "knowledge graph, model-proposed characters alone / with the matrix / in the "
                                           "key), naming and novelty, in-sample against nested",
    "biorag_key_fuzzy_figures_v1.py": "FIGURE S: crisp against graded key - where they part, which is right, and "
                                      "what the graded score is worth as a warning",
    "biorag_graph_support_figure_v1.py": "FIGURE 4: distances between species, and hold-out support of the characters "
                                         "the knowledge graph and the key use to separate them",
    "biorag_vlm_reliability_figure_v1.py": "FIGURE 5: reliability of the model's readings - descriptive categorical "
                                           "characters within and between sessions; model-proposed binary characters "
                                           "(agreement, kappa, relocation distance, attrition)",
    "biorag_vlm_heatmap_figures_v1.py": "OPTIONAL ARM: density of reported locations over the mean aligned specimen "
                                        "(Figure 6 and one atlas per structure); spread, offset, share off the specimen",
    "biorag_autapomorphy_v1.py": "OPTIONAL ARM: states fixed within a species and absent outside it — proposed on "
                                 "half of each series, confirmed on the rest, FDR over every test attempted",
    # --- feature extraction the matrix is built from ----------------------
    "texture_phenomics_homology.py": "texture features (GLCM, LBP, Gabor/FFT) per TPS-warped homologous cell",
    "texture_phenomics_from_coco.py": "texture features per sclerite, with PCA/UMAP and clustering",
    "texture_phylo_mapping_v9.R": "R: texture phenomics against a phylogeny (phylosignal, ancestral states) — "
                                  "the texture classifier the descriptive words are compared with",
    "color_phenomics_homology_v2_1.py": "colour-homology grid: CIE L*a*b* per homologous cell",
    "color_extraction_with_names-v8.py": "colour extraction and the KD-tree colour-name lookup",
    "compile_specimen_data.py": "joins the feature pipelines into one specimen matrix",
    "landmark_gpa_V1.py": "Procrustes GPA for discrete landmarks",
    "measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py":
        "morphometrics with Florence-2 scale-bar calibration",
    "semi_landmark_and_kpts_procrustesV34_GPA.py": "semilandmark GPA and outline shape",
    "FHS_and_CLAHE_V21.py": "colour extraction with background normalisation",
}
PAPER = {
    "generate_biorag_workflow_v6.py": "the workflow figure (editable SVG), numbers read from the reports",
    "generate_deconfab_figures_v2.py": "the audit and delimitation figures",
    "generate_script_inventory_figure_v2.py": "supplementary figure: every script by stage, which call a model, "
                                              "which were added 18-21 September; reads the source files",
    "manuscript_v18_references_v17_full.md": "reference list as it stood in v17",
    "manuscript_v18_references_main.md": "reference list of the main text (every entry cited, every citation listed)",
    "manuscript_v18_references_supplement.md": "reference list of the supplement",
    "build_manuscript_v18.py": "builds the manuscript and its supplement from the two templates; every "
                               "number is a token filled from a report, and an unknown token stops the build",
    "manuscript_v18_main.md.tmpl": "main text template",
    "manuscript_v18_supplement.md.tmpl": "supplement template",
    "manuscript_numbers_v18.tsv": "every number in the paper with the report file it came from",
    "collect_figures_and_outputs.py": "copies the main figures, the supplementary figures and the summary statistics "
                                      "behind them into three folders, each with an index of where the original lives",
    "apply_biorag_v2_manuscript_updates.py": "SUPERSEDED: the inserting generator used for drafts v4-v17",
}
RESULT_GLOBS = [
    "compiled_key_tier/*.json", "compiled_key_tier/feature_dictionary.tsv",
    "compiled_key_tier/outlier_flags.tsv", "compiled_key_tier/coverage_species_by_structure.tsv",
    "key/key_validation_report.json", "key/taxonomic_key.txt", "key/key_tree.json",
    "descriptions/confabulation_report_v2/confabulation_summary.json",
    "descriptions/confabulation_report_v2_round1_as_generated/confabulation_summary.json",
    "descriptions/ontology_coverage.json", "descriptions/ontology_structure_terms.tsv",
    "descriptions/refine_validation_summary.tsv", "descriptions/subjective_check_summary.json",
    "descriptive_states/scoring_report.json", "descriptive_states/repeatability.json",
    "descriptive_states/character_reliability.json", "descriptive_states/character_reliability.tsv",
    "descriptive_states/figures/*.tsv", "descriptive_states/figures/*.png",
    "descriptive_states/figures/figures_index.json",
    "descriptions/character_gate/character_gate_report.json",
    "descriptions_v2/character_gate/character_gate_report.json",
    "descriptive_states/gating_ablation/*/calibration*.json",
    "novelty_gated/calibration*.json", "novelty_gated/novelty_summary.tsv",
    "calibration/calibration.json", "calibration/calibration_report.txt",
    "calibration/novelty_rules.tsv", "calibration/per_species.tsv",
    "calibration/novelty_tradeoff.tsv", "calibration/novelty_specimens.tsv",
    "calibration/matrix_detection_scores.tsv", "calibration/matrix_false_alarm_scores.tsv",
    "calibration/matrix_identification_distances.tsv", "calibration/matrix_tradeoff.tsv",
    "key/identification_test.tsv",
    "annotation_screen/*.json", "annotation_screen/*.tsv", "annotation_screen/*.csv",
    "instrument_comparison_v3/*.tsv", "instrument_comparison_v3/*.json",
    "instrument_comparison_v3/fig_instrument_comparison_v3.png",
    "key_fuzzy_loo/fuzzy_key_summary.json", "key_fuzzy_loo/fuzzy_key_specimens.tsv",
    "key_holdout_v2/*.json", "key_holdout_v2/*.tsv",
    "graph_test_loo/*.json", "graph_test_loo/*.tsv",
    "congruence_compare/*.tsv", "congruence_compare/*.json",
    "compiled_key_tier_plus_vlm/discrete_characters_report.json",
    "key_plus_vlm/key_validation_report.json", "key_plus_vlm/key_tree.json",
    "key_plus_vlm/identification_test.tsv", "key_plus_vlm/fuzzy_loo/fuzzy_key_summary.json",
    "vlm_combined/vlm_character_states.tsv", "vlm_combined/vlm_proposed_characters.tsv",
    "vlm_combined/fixed_within_species.tsv",
    "vlm_characters_aligned*/vlm_character_reliability.tsv",
    "vlm_characters_aligned*/vlm_characters_summary.json",
    "autapomorphies_vlm_final/*.json", "autapomorphies_vlm_final/*.tsv",
    "vlm_combined/figures_heatmap/vlm_heatmap_index.tsv",
    "vlm_combined/figures_heatmap/vlm_heatmap_summary.json",
    "vlm_combined/figures_heatmap/summary_vlm_attention.png",
    "descriptions/confabulation_report_v2_after_screen_20260920/confabulation_summary.json",
    "descriptions/confabulation_report_v2_after_screen_20260920/confabulation_issues.tsv",
    "audit_20260920/**/*.tsv", "audit_20260920/**/*.json", "audit_20260920/**/*.log",
    "descriptions/confabulation_report_v2_after_screen_repair_20260920/confabulation_summary.json",
    "descriptions/character_gate_check_after_screen_repair_20260920/*.json",
    "homology_frames/frame_orientation.tsv", "homology_frames/homology_frames_summary.json",
    "methods_summary/*.tsv", "methods_summary/*.json", "methods_summary/*.png",
    "annotation_reorient/*.tsv", "annotation_reorient/*.json", "annotation_reorient/*.jpg",
    "annotation_screen_v1_1_original_coco/*.json", "annotation_screen_v1_1_original_coco/*.tsv",
    "annotation_screen_v1_1_original_coco/*.csv", "annotation_screen_v1_1_original_coco/*.jpg",
    "audit_20260921/**/*.json", "audit_20260921/**/*.tsv",
    "label_conflicts/*.json", "label_conflicts/*.tsv",
    "character_robustness/*.tsv", "character_robustness/*.json", "character_robustness/*.png",
    "key_jackknife_support/*.tsv", "key_jackknife_support/*.json",
    "tier2_sets_*/*.tsv", "tier2_sets_*/*.json", "tier2_sets_*/*.sh",
    "tier2_sets_*/outline_shape/*.tsv", "tier2_sets_*/outline_shape/*.json",
    "tier2_sets_*/compare/*.tsv", "tier2_sets_*/compare/*.json",
    "descriptive_states_session2_20260921/descriptive_states_by_specimen.tsv",
    "descriptive_states_session2_20260921/scoring_report.json",
    "descriptive_states_session2_20260921/character_reliability.*",
    "descriptive_states_session2_20260921/retest/descriptive_states_by_specimen.tsv",
    "descriptive_states_session2_20260921/retest/scoring_report.json",
    "descriptive_states_session2_20260921/reliability_*/*.tsv", "descriptive_states_session2_20260921/reliability_*/*.json",
    "descriptive_states_session2_20260921/figures/*.tsv",
    "descriptive_states/retest_compare_five_readings_20260921/*.tsv",
    "descriptive_states/retest_compare_five_readings_20260921/*.json",
    "frame_orientation_calibration_20260921/*.tsv", "frame_orientation_calibration_20260921/*.json",
    "graph_support/*.tsv", "graph_support/*.json", "graph_support/*.png",
    "key_fuzzy_loo/figures/*.tsv", "key_fuzzy_loo/figures/*.json", "key_fuzzy_loo/figures/*.png",
    "vlm_reliability/*.tsv", "vlm_reliability/*.json", "vlm_reliability/*.png",
    "vlm_location_retest_20260920/vlm_character_states_retest.tsv",
    "vlm_location_retest_20260920/vlm_character_reliability.tsv",
    "vlm_location_retest_20260920/vlm_characters_summary.json",
    "descriptive_states/retest/descriptive_states_by_specimen.tsv", "descriptive_states/retest/scoring_report.json",
    "descriptive_states_retestB_20260920/retest/descriptive_states_by_specimen.tsv",
    "descriptive_states_retestB_20260920/retest/scoring_report.json",
    "descriptive_states/retest_compare/*.tsv", "descriptive_states/retest_compare/*.json",
    "descriptive_states/reliability_retest70_20260920/*.tsv", "descriptive_states/reliability_retest70_20260920/*.json",
    "vlm_combined/figures_heatmap_before_orientation_20260920/vlm_heatmap_index.tsv",
    "vlm_combined/figures_heatmap_before_orientation_20260920/vlm_heatmap_summary.json",
    "descriptive_states/figures_pre_screen_20260920/*.tsv",
    "calibration/fig_key_report.png", "calibration/fig_character_report.png",

    "key_external/*/key_external_summary.json", "key_external/*/key_external_outcomes.tsv",

    "outgroup_check*/outgroup_batch_check.json", "outgroup_check*/outgroup_z_by_character.tsv",
    "novelty_*/external_scores.tsv", "novelty_*/novelty_summary.tsv",
    "descriptions/character_gate/character_gate_edits.tsv",
    "novelty*/calibration*.json", "novelty*/novelty_summary.tsv",
    "novelty*/character_discrimination*.tsv",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path, manifest: list, purpose: str = ""):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append({"path": str(dst), "sha256": sha256(src),
                     "bytes": src.stat().st_size, "purpose": purpose})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--include_paper", action="store_true")
    ap.add_argument("--paper_dir", default=None,
                    help="folder holding the manuscript builder and figure generators "
                         "(default: ~/Desktop/Towley_paper)")
    ap.add_argument("--results", default=None, help="monograph directory; small reports are copied")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest: list = []

    for name, purpose in PURPOSE.items():
        src = HERE / name
        if src.exists():
            copy(src, out / "scripts" / name, manifest, purpose)
        else:
            print(f"  missing (skipped): {name}")
    if a.include_paper:
        paper_dir = Path(a.paper_dir).expanduser() if a.paper_dir else Path.home() / "Desktop" / "Towley_paper"
        for name, purpose in PAPER.items():
            for src in (HERE / name, paper_dir / name):
                if src.exists():
                    copy(src, out / "scripts_paper" / name, manifest, purpose)
                    break
    for sub, purpose in (("biorag_prompts", "universal system prompts, taxon profiles, schemas, ontologies"),
                         ("tests", "tests (synthetic data only, no model calls)")):
        for src in sorted((HERE / sub).rglob("*")):
            if src.is_file() and ".bak" not in src.name and "__pycache__" not in str(src):
                # inside scripts/, as in the working folder: the scripts and the tests look for them there
                copy(src, out / "scripts" / sub / src.relative_to(HERE / sub), manifest, purpose)
    for doc in ("BIORAG_V2_README.md",):
        if (HERE / doc).exists():
            copy(HERE / doc, out / doc, manifest, "how everything fits together and how to rerun it")
    if (GUI / "descriptron-v2-v73.py").exists():
        copy(GUI / "descriptron-v2-v73.py", out / "scripts" / "descriptron-v2-v73.py", manifest,
             "the GUI; its BioRAG button runs run_full_pipeline_v2.py")

    n_results = 0
    if a.results:
        res = Path(a.results)
        for pat in RESULT_GLOBS:
            for src in sorted(res.glob(pat)):
                if src.is_file() and src.stat().st_size < 20_000_000:
                    copy(src, out / "results" / src.relative_to(res), manifest, "result file")
                    n_results += 1

    env = ""
    try:
        env = subprocess.run([a.python, "-m", "pip", "freeze"], capture_output=True, text=True,
                             timeout=120).stdout
    except Exception:  # noqa: BLE001
        pass
    (out / "environment.txt").write_text(
        f"python: {subprocess.run([a.python, '--version'], capture_output=True, text=True).stdout.strip()}\n"
        f"generated: {datetime.now().isoformat()}\n\n{env}")
    with open(out / "MANIFEST.tsv", "w") as f:
        f.write("path\tbytes\tsha256\tpurpose\n")
        for m in sorted(manifest, key=lambda m: m["path"]):
            rel = Path(m["path"]).relative_to(out)
            f.write(f"{rel}\t{m['bytes']}\t{m['sha256']}\t{m['purpose']}\n")

    readme = [f"# BioRAG v2 — reproducibility bundle", "",
              f"Generated {datetime.now():%Y-%m-%d} from `{HERE}`. Every file is listed in `MANIFEST.tsv` "
              f"with its SHA-256; `environment.txt` records the interpreter and package versions that "
              f"produced the results.", "",
              "## Order to run", "",
              "1. `run_full_pipeline_v2.py --workflow v2` — the whole workflow; each step is one script below "
              "(step 20.5 screens the annotations before the matrix is built).",
              "2. Evaluation of the text: `biorag_confabulation_checker_v2.py`, then "
              "`biorag_description_refiner_v1.py --repair_existing`, then the checker again.",
              "3. Descriptive characters: `biorag_descriptive_scoring_v1.py` (per specimen), then "
              "`biorag_character_reliability_v1.py` (which characters repeat, for this taxon), then "
              "`biorag_reliability_figures_v1.py` (the tables and figures of that investigation), then "
              "`biorag_subjective_checks_v1.py --descriptive_matrix ...`.",
              "4. Instruments: `biorag_calibrate_v1.py` (key and matrix, both arms held out), "
              "`biorag_graph_identify_v1.py --arm leave_one_specimen_out --ranges_from matrix`, "
              "`biorag_key_fuzzy_v1.py --known_arm leave_one_specimen_out`, then "
              "`biorag_instrument_compare_v3.py` (Tables 1-2 and Figure 2). A candidate series is scored with "
              "`biorag_novelty_score_v1.py` (add `--descriptive_matrix`, `--computed_dir` and `--reliability "
              "descriptive_states/character_reliability.json`, which drops the characters that did not repeat "
              "and keeps the rest at their full scale).",
              "5. Optional arm, present/absent characters: `biorag_homology_frame_v1.py`, "
              "`biorag_vlm_characters_v1.py`, `biorag_autapomorphy_v1.py`, `biorag_add_discrete_characters_v1.py`, "
              "`biorag_congruence_compare_v1.py`, `biorag_vlm_heatmap_figures_v1.py`.",
              "6. Outputs: `build_species_treatment_docx_v2.py`, `build_monograph_exports_v1.py`; the paper: "
              "`build_manuscript_v18.py`.", "",
              "Results folders ending `_pre_screen_20260920`, and `audit_20260920/`, are not in the bundle's "
              "main line: they record the analyses as they stood before the annotation screen was applied "
              "and before the hold-out corrections, and are described in Supplementary Text S7-S8.", "",
              "`scripts/tests/test_biorag_holdout_v1.py` states the hold-out invariant (changing the data of whatever is "
              "withheld must not change its score) for the matrix, the key and the graph.", "",
              "Everything taxon-specific lives in a taxon profile "
              "(`scripts/biorag_prompts/taxon_profiles/`); the system prompts are taxon-agnostic. "
              "`scripts/` mirrors the working folder, so every script and test runs from there as it stands "
              "(`python scripts/tests/test_biorag_v2.py`).", "",
              "## What each script is for", ""]
    for name, purpose in list(PURPOSE.items()) + (list(PAPER.items()) if a.include_paper else []):
        readme.append(f"- `{name}` — {purpose}")
    readme += ["", f"Files: {len(manifest)}" + (f"; result files copied: {n_results}" if a.results else "")]
    (out / "README_BUNDLE.md").write_text("\n".join(readme))
    print(f"bundle -> {out}  ({len(manifest)} files, {n_results} result files)")
    print(f"  manifest: {out / 'MANIFEST.tsv'}")


if __name__ == "__main__":
    main()
