#!/usr/bin/env python3
"""
Descriptron → BioRAG Full Pipeline Orchestrator (v2)

--workflow v2 (DEFAULT, recommended; taxon-agnostic):
  data steps 0-7 → key_matrix (Tier-1 matrix, repairs + outlier screen)
  → biorag (image-based per-structure descriptions, evidence-tiered prompts,
    --llm_backend api|claude-code) → key_build (data-driven key)
  → describe_v2 (treatments: Diagnosis/Description/Sexual dimorphism/Remarks)
  → subjective_check → confab_check_v2 (independent audit of every number,
    comparison and Remarks statement against the matrix; problems are repaired
    by describe_v2 --repair_existing) → ontology_v2 (EQ annotation from OBO
    releases) → figure_plates → treatments_docx (+ machine-readable exports)
  A draft taxon profile is generated when --taxon_profile is not given.
--workflow v1: the original 20-step sequence below (LLM-written key).

Chains 20 sequential steps from COCO annotations through to formal
taxonomic descriptions. Each step is run as a subprocess using the
same scripts that the Descriptron GUI buttons call.

Steps:
  0. COCO JSON Cleanup          — validate and fix COCO JSON for pipeline compatibility
  1. Measurements (V35)         — morphometrics + Florence-2 scale bars
  2. Semilandmarks + GPA (V34)  — Procrustes shape analysis + foreground masks
  3. Color Extraction (V21)     — FHS + CLAHE per-category color features
  4. Color Homology (v2.1)      — grid-based CIE LAB per homologous region
  5. Texture Homology           — GLCM + LBP per homologous region
  6. Landmark GPA (optional)    — discrete homologous landmark analysis
  7. Compile Specimen Data      — join all pipelines, diagnostic feature selection
  8. BioRAG Descriptions        — VLM-based taxonomic descriptions + key
  9. Confabulation Check (r1)   — validate numeric claims against compiled data
 10. Confabulation Fix (r2)     — re-describe with prompt-constrained numeric fidelity
 11. Ontology Assignment        — auto-assign OBO Foundry URIs to description terms
 12. Key Feature Filter         — create human-measurable-only compiled data for key
 13. Key Regeneration           — regenerate key using filtered compiled directory
 14. Key Numeric Check          — validate numeric thresholds in key against data
 15. Key Qualitative Check      — validate qualitative claims, E_Dicho efficiency
 16. Confabulation Comparison   — comparison figures/tables across checker rounds
 17. Key Comparison & Figures   — key check comparison + qualitative check figures
 18. Figure Plates              — per-species annotated specimen plates + figure ref injection
 19. Figure Captions            — generate captions CSV/TXT for all pipeline figures

Usage:
  python run_full_pipeline.py \\
    --coco_json /path/to/coco.json \\
    --image_dir /path/to/images \\
    --group_labels /path/to/group_labels.csv \\
    --output_base /path/to/output \\
    --family Liviidae \\
    --api_key_file /path/to/key.txt

  # Dry run (prints commands without executing):
  python run_full_pipeline.py ... --dry_run

  # Resume (skips completed steps):
  python run_full_pipeline.py ...   # just re-run same command

  # Force re-run of specific steps:
  python run_full_pipeline.py ... --only_steps compile biorag --force

  # Run only confabulation checking and key validation:
  python run_full_pipeline.py ... --only_steps confab_check key_filter key_regen key_num_check key_qual_check
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent

STEP_NAMES = [
    "coco_clean",
    "measurements",
    "semilandmarks",
    "color",
    "color_homology",
    "texture",
    "landmark_gpa",
    "compile",
    "biorag",
    "confab_check",
    "confab_fix",
    "ontology",
    "key_filter",
    "key_regen",
    "key_num_check",
    "key_correct",
    "key_qual_check",
    "confab_compare",
    "key_compare",
    "figure_plates",
    "figure_captions",
    "annotation_screen",
    "key_matrix",
    "key_build",
    "calibrate",
    "describe_v2",
    "descriptive_states",
    "char_reliability",
    "char_figures",
    "subjective_check",
    "confab_check_v2",
    "type_material",
    "char_gate",
    "ontology_v2",
    "treatments_docx",
]

logger = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _python_bin(conda_env: str) -> str:
    conda_base = Path(sys.executable).resolve()
    while conda_base.name != "envs" and conda_base != conda_base.parent:
        conda_base = conda_base.parent
    if conda_base.name == "envs":
        candidate = conda_base / conda_env / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    for search in [
        Path.home() / "Desktop" / "Descriptron" / "conda" / "envs" / conda_env / "bin" / "python",
        Path(f"/home/{os.getenv('USER', 'localuser')}/Desktop/Descriptron/conda/envs/{conda_env}/bin/python"),
    ]:
        if search.exists():
            return str(search)
    return f"conda run -n {conda_env} python"


def _discover_categories(coco_json_path: str) -> List[str]:
    with open(coco_json_path) as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    ann_counts: Dict[int, int] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        ann_counts[cid] = ann_counts.get(cid, 0) + 1
    categories = []
    for cid, name in sorted(cat_id_to_name.items()):
        if ann_counts.get(cid, 0) > 0:
            categories.append(name)
    logger.info(f"Discovered {len(categories)} categories with annotations")
    return categories


def _has_keypoints(coco_json_path: str) -> bool:
    with open(coco_json_path) as f:
        coco = json.load(f)
    for ann in coco["annotations"]:
        if ann.get("keypoints") and any(v != 0 for v in ann["keypoints"]):
            return True
    return False


def _run_step(cmd: List[str], step_name: str, log_dir: Path,
              dry_run: bool = False) -> bool:
    cmd_str = " \\\n  ".join(cmd)
    logger.info(f"  Command:\n    {cmd_str}")

    if dry_run:
        logger.info(f"  [DRY RUN] Skipping execution")
        return True

    log_file = log_dir / f"{step_name}.log"
    with open(log_file, "w") as lf:
        lf.write(f"# {step_name}\n# {datetime.now().isoformat()}\n")
        lf.write(f"# {cmd_str}\n\n")
        lf.flush()
        t0 = time.time()
        result = subprocess.run(
            cmd, stdout=lf, stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR.parent),
        )
        elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"  FAILED (exit code {result.returncode}, {elapsed:.0f}s)")
        logger.error(f"  See log: {log_file}")
        return False
    logger.info(f"  Completed in {elapsed:.0f}s")
    return True


def _find_fg_categories(semi_dir: Path) -> List[str]:
    categories = []
    if not semi_dir.exists():
        return categories
    for cat_dir in sorted(semi_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        inner = cat_dir / cat_dir.name
        if inner.is_dir():
            has_fg = any(inner.glob("*_fg_*"))
            has_bin = any(inner.glob("*_bin_*"))
            if has_fg and has_bin:
                categories.append(cat_dir.name)
    return categories


def _find_gpa_categories(semi_dir: Path) -> List[str]:
    categories = []
    if not semi_dir.exists():
        return categories
    for cat_dir in sorted(semi_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if any(cat_dir.glob("shape_traits_phylo_*.csv")):
            categories.append(cat_dir.name)
    return categories


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step_coco_clean(cfg: Dict, python: str, log_dir: Path) -> bool:
    coco_json = Path(cfg["coco_json"])
    cleaned = coco_json.parent / (coco_json.stem + "_cleaned.json")

    if cleaned.exists() and not cfg["force"]:
        logger.info("  Already cleaned (%s exists), using cleaned version", cleaned.name)
        cfg["coco_json"] = str(cleaned)
        return True

    script = SCRIPT_DIR.parent / "clean_coco_json.py"
    if not script.exists():
        logger.warning("  clean_coco_json.py not found at %s, skipping cleanup", script)
        return True

    cmd = [
        python, str(script),
        str(coco_json),
        "--fix",
        "--output", str(cleaned),
    ]
    if cfg.get("image_dir"):
        cmd += ["--images_dir", cfg["image_dir"]]

    ok = _run_step(cmd, "step0_coco_clean", log_dir, cfg["dry_run"])
    if ok and cleaned.exists():
        cfg["coco_json"] = str(cleaned)
        logger.info("  Pipeline will use cleaned JSON: %s", cleaned)
    elif ok:
        logger.info("  No fixes needed, using original JSON")
    return ok


def step_measurements(cfg: Dict, python: str, log_dir: Path) -> bool:
    out_dir = Path(cfg["output_base"]) / "measurements"
    out_dir.mkdir(parents=True, exist_ok=True)
    check_file = out_dir / "all_metrics.csv"

    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (all_metrics.csv exists), skipping")
        return True

    script = SCRIPT_DIR / "measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py"
    cmd = [
        python, str(script),
        "--json", cfg["coco_json"],
        "--image_dir", cfg["image_dir"],
        "--output_dir", str(out_dir),
        "--method", "pca",
        "--save_results",
        "--output_file", str(out_dir / "all_metrics.csv"),
        "--jsonl_output", str(out_dir / "all_metrics.jsonl"),
    ]
    if cfg.get("grouping_file"):
        cmd += ["--grouping_file", cfg["grouping_file"]]

    return _run_step(cmd, "step1_measurements", log_dir, cfg["dry_run"])


def step_semilandmarks(cfg: Dict, python: str, log_dir: Path) -> bool:
    out_dir = Path(cfg["output_base"]) / "semilandmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_cats = _find_gpa_categories(out_dir)
    if existing_cats and not cfg["force"]:
        logger.info(f"  Already complete ({len(existing_cats)} categories), skipping")
        return True

    # V42 (2026-09-25) = V34 + mirror-image outlines reflected before GPA (checked against geomorph); V34 fallback
    script = SCRIPT_DIR / "semi_landmark_and_kpts_procrustesV42_GPA.py"
    align = "reflect_mirrored"
    if not script.is_file():
        script, align = SCRIPT_DIR / "semi_landmark_and_kpts_procrustesV34_GPA.py", "without_reflection"
    cmd = [
        python, str(script),
        "--json", cfg["coco_json"],
        "--image_dir", cfg["image_dir"],
        "--output_dir", str(out_dir),
        "--num_landmarks", str(cfg.get("num_landmarks", 100)),
        "--anchor_method", "none",
        "--alignment_method", align,
        *(["--slide_method", cfg["slide_method"]] if cfg.get("slide_method", "none") != "none" and "V42" in script.name else []),
        "--perform_manova",
    ]
    if cfg.get("group_labels"):
        cmd += ["--group_labels", cfg["group_labels"]]

    return _run_step(cmd, "step2_semilandmarks", log_dir, cfg["dry_run"])


def step_color_extraction(cfg: Dict, python: str, log_dir: Path) -> bool:
    semi_dir = Path(cfg["output_base"]) / "semilandmarks"
    out_base = Path(cfg["output_base"]) / "color_extraction"
    out_base.mkdir(parents=True, exist_ok=True)

    categories = _find_fg_categories(semi_dir)
    if not categories:
        logger.warning("  No foreground masks found in semilandmarks output")
        return True

    script = SCRIPT_DIR / "FHS_and_CLAHE_V21.py"
    all_ok = True

    for cat in categories:
        out_dir = out_base / cat
        check_file = list(out_dir.glob(f"color_traits_phylo_{cat}_adaptive.csv"))
        if check_file and not cfg["force"]:
            logger.info(f"    [{cat}] already complete, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        input_dir = semi_dir / cat / cat

        has_fhs = any(f.name.startswith("fhs_") and f.suffix == ".png"
                      for f in out_dir.iterdir() if f.is_file())

        cmd = [
            python, str(script),
            "--input_dir", str(input_dir),
            "--output_dir", str(out_dir),
        ]
        if has_fhs:
            cmd.append("--skip_preprocessing")
        cmd += [
            "--run_lab_average",
            "--median_threshold",
            "--run_segmentation",
            "--run_normalization",
            "--remove_shine",
            "--run_pattern_analysis",
            "--run_efa",
            "--category_name", cat,
        ]
        if cfg.get("bg_normalize", True):
            cmd += ["--bg_normalize"]
            if cfg.get("image_dir"):
                cmd += ["--bg_original_dir", cfg["image_dir"]]
            if cfg.get("coco_json"):
                cmd += ["--bg_coco_json", cfg["coco_json"]]

        logger.info(f"    [{cat}] Running color extraction...")
        ok = _run_step(cmd, f"step3_color_{cat}", log_dir, cfg["dry_run"])
        if not ok:
            all_ok = False
            logger.error(f"    [{cat}] FAILED")

    return all_ok


def step_color_homology(cfg: Dict, python: str, log_dir: Path) -> bool:
    semi_dir = Path(cfg["output_base"]) / "semilandmarks"
    out_base = Path(cfg["output_base"]) / "color_homology"
    out_base.mkdir(parents=True, exist_ok=True)

    categories = _find_gpa_categories(semi_dir)
    if not categories:
        logger.warning("  No GPA output found in semilandmarks dir")
        return True

    script = SCRIPT_DIR / "color_phenomics_homology_v2_1.py"
    all_ok = True

    for cat in categories:
        out_dir = out_base / cat
        check_file = list(out_dir.glob(f"color_homology_features_{cat}.csv"))
        if check_file and not cfg["force"]:
            logger.info(f"    [{cat}] already complete, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        gpa_dir = semi_dir / cat

        cmd = [
            python, str(script),
            "--gpa_dir", str(gpa_dir),
            "--json", cfg["coco_json"],
            "--image_dir", cfg["image_dir"],
            "--output_dir", str(out_dir),
            "--category_name", cat,
            "--color_mode", "combined",
        ]

        logger.info(f"    [{cat}] Running color homology...")
        ok = _run_step(cmd, f"step4_colhom_{cat}", log_dir, cfg["dry_run"])
        if not ok:
            all_ok = False
            logger.error(f"    [{cat}] FAILED")

    return all_ok


def step_texture_homology(cfg: Dict, python: str, log_dir: Path) -> bool:
    semi_dir = Path(cfg["output_base"]) / "semilandmarks"
    out_base = Path(cfg["output_base"]) / "texture_homology"
    out_base.mkdir(parents=True, exist_ok=True)

    categories = _find_gpa_categories(semi_dir)
    if not categories:
        logger.warning("  No GPA output found in semilandmarks dir")
        return True

    script = SCRIPT_DIR / "texture_phenomics_homology.py"
    all_ok = True

    for cat in categories:
        out_dir = out_base / cat
        check_file = list(out_dir.glob(f"texture_homology_features_{cat}.csv"))
        if check_file and not cfg["force"]:
            logger.info(f"    [{cat}] already complete, skipping")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        gpa_dir = semi_dir / cat

        cmd = [
            python, str(script),
            "--gpa_dir", str(gpa_dir),
            "--json", cfg["coco_json"],
            "--image_dir", cfg["image_dir"],
            "--output_dir", str(out_dir),
            "--category_name", cat,
        ]

        logger.info(f"    [{cat}] Running texture homology...")
        ok = _run_step(cmd, f"step5_texhom_{cat}", log_dir, cfg["dry_run"])
        if not ok:
            all_ok = False
            logger.error(f"    [{cat}] FAILED")

    return all_ok


def step_landmark_gpa(cfg: Dict, python: str, log_dir: Path) -> bool:
    kpts_json = cfg.get("keypoints_json")
    if not kpts_json:
        if _has_keypoints(cfg["coco_json"]):
            kpts_json = cfg["coco_json"]
        else:
            logger.info("  No keypoint annotations found, skipping")
            return True

    out_dir = Path(cfg["output_base"]) / "landmark_gpa"
    out_dir.mkdir(parents=True, exist_ok=True)
    # landmark_gpa_V1.py writes <out>/<category>/<category>_pc_scores.csv (there is no <out>/pc_scores.csv)
    if any(out_dir.glob("*/*_pc_scores.csv")) and not cfg["force"] and not cfg.get("extra_keypoints_json"):
        logger.info("  Already complete (*_pc_scores.csv exists), skipping")
        return True

    # V2 (2026-09-25) reflects mirror-image specimens before GPA (V1 let them dominate PC1); V1 as fallback
    script = SCRIPT_DIR / "landmark_gpa_V2.py"
    if not script.is_file():
        script = SCRIPT_DIR / "landmark_gpa_V1.py"
    cmd = [
        python, str(script),
        "--json", kpts_json,
        "--output_dir", str(out_dir),
    ]
    if cfg.get("group_labels"):
        cmd += ["--group_labels", cfg["group_labels"]]

    ok = True
    if cfg["force"] or not any(out_dir.glob("*/*_pc_scores.csv")):
        ok = _run_step(cmd, "step6_landmark_gpa", log_dir, cfg["dry_run"])
    # further landmark sets (another body part with its own landmark scheme): one GPA each, side by side
    for extra in (cfg.get("extra_keypoints_json") or []):
        xd = Path(cfg["output_base"]) / f"landmark_gpa_{Path(extra).stem}"
        if any(xd.glob("*/*_pc_scores.csv")) and not cfg["force"]:
            continue
        xd.mkdir(parents=True, exist_ok=True)
        xcmd = [python, str(script), "--json", extra, "--output_dir", str(xd)]
        if cfg.get("group_labels"):
            xcmd += ["--group_labels", cfg["group_labels"]]
        ok = _run_step(xcmd, f"step6_landmark_gpa_{Path(extra).stem}", log_dir, cfg["dry_run"]) and ok
    return ok


def step_compile(cfg: Dict, python: str, log_dir: Path) -> bool:
    base = Path(cfg["output_base"])
    out_dir = base / "compiled"
    out_dir.mkdir(parents=True, exist_ok=True)
    check_file = out_dir / "diaphorina_full_features.csv"

    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (full_features.csv exists), skipping")
        return True

    color_dirs = [str(base / "color_extraction")]
    colhom_dirs = [str(base / "color_homology")]
    tex_dirs = [str(base / "texture_homology")]
    lmk_dirs = []

    # every landmark_gpa* folder that holds a GPA result (the test used to look for a file that is never written,
    # so landmarks were silently left out of every table compiled through this orchestrator)
    for d in sorted(base.glob("landmark_gpa*")):
        if d.is_dir() and any(d.glob("*/*_pc_scores.csv")):
            lmk_dirs.append(str(d))

    script = SCRIPT_DIR / "compile_specimen_data.py"
    cmd = [
        python, str(script),
        "--measurements_dir", str(base / "measurements"),
        "--semilandmarks_dir", str(base / "semilandmarks"),
        "--color_dirs", *color_dirs,
        "--color_homology_dirs", *colhom_dirs,
        "--texture_dirs", *tex_dirs,
        "--coco_json", cfg["coco_json"],
        "--group_labels", cfg["group_labels"],
        "--output_dir", str(out_dir),
        "--alpha", str(cfg.get("alpha", 0.05)),
    ]
    if lmk_dirs:
        cmd += ["--landmark_gpa_dirs", *lmk_dirs]
    if cfg.get("exclude_list"):
        cmd += ["--exclude_list", cfg["exclude_list"]]
    if cfg.get("ratio_config"):
        cmd += ["--ratio_config", cfg["ratio_config"]]
    if cfg.get("species_pattern"):
        cmd += ["--species_pattern", cfg["species_pattern"]]

    return _run_step(cmd, "step7_compile", log_dir, cfg["dry_run"])


def step_biorag(cfg: Dict, python: str, log_dir: Path) -> bool:
    base = Path(cfg["output_base"])
    out_dir = base / "biorag_descriptions"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = out_dir / "biorag_summary.json"
    if summary.exists() and not cfg["force"]:
        logger.info("  Already complete (biorag_summary.json exists), skipping")
        return True

    semi_dirs = [str(base / "semilandmarks")]

    v2 = cfg.get("workflow", "v2") == "v2"
    script = SCRIPT_DIR / ("biosyslit_rag_retrieval_v2.py" if v2 else "biosyslit_rag_retrieval.py")
    cmd = [
        python, str(script), "describe-biorag",
        "--compiled-dir", str(base / "compiled"),
        "--semilandmarks-dir", *semi_dirs,
        "--image-dir", cfg["image_dir"],
        "--coco-json", cfg["coco_json"],
        "--output-dir", str(out_dir),
        "--vision-mode", "foreground",
        "--no-florence2",
        "--family", cfg.get("family", ""),
        "--k", str(cfg.get("k", 3)),
        "--model", cfg.get("model", "claude-sonnet-4-6"),
        "--group-labels", cfg["group_labels"],
    ]
    if cfg.get("api_key_file"):
        cmd += ["--api-key-file", cfg["api_key_file"]]
    if cfg.get("pdf_dir"):
        cmd += ["--pdf-dir", cfg["pdf_dir"]]
    if cfg.get("rag_index"):
        # a prebuilt literature index (BioSysLit and/or PDFs); describe-biorag uses it
        # INSTEAD of --pdf-dir, so PDFs wanted as well must be indexed into it
        cmd += ["--index", cfg["rag_index"]]
        if cfg.get("pdf_dir"):
            logger.warning("  --rag_index and --pdf_dir both given: only the index is used. "
                           "Add the PDFs to the index with: biosyslit_rag_retrieval_v2 index "
                           "--pdf-dir <dir> --taxon <taxon> --output <index.json>")
    if cfg.get("user_prompts"):
        cmd += ["--user-prompts", cfg["user_prompts"]]
    if cfg.get("species"):
        cmd += ["--species"] + cfg["species"]
    if cfg.get("label_dir"):
        cmd += ["--label-dir", cfg["label_dir"]]
    if v2:
        # evidence-tiered data sheets + chosen backend; the key is built later (key_build)
        cmd += ["--matrix-dir", str(base / "compiled_key_tier")] + \
            [a for a in _v2_backend_args(cfg) if a != "none"]
        if cfg.get("llm_backend") == "none":
            # no model: skip the image descriptions rather than halt the pipeline. Everything
            # computed (matrix, key, delimitation, audits) still runs, and describe_v2 writes
            # the evidence sheets without prose, as it already does under 'none'.
            logger.warning("  --llm_backend none: skipping the image-based descriptions "
                           "(this step needs a model: api or claude-code)"
                           + ("; the literature given is not used" if cfg.get("pdf_dir") or cfg.get("rag_index") else ""))
            return True
    else:
        cmd += ["--generate-key"]

    return _run_step(cmd, "step8_biorag", log_dir, cfg["dry_run"])


# ═══════════════════════════════════════════════════════════════════════════════
# CONFABULATION & KEY VALIDATION STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step_confab_check(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 9: Run confabulation checker on BioRAG descriptions (round 1)."""
    base = Path(cfg["output_base"])
    desc_dir = base / "biorag_descriptions"
    out_dir = desc_dir / "confabulation_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_file = out_dir / "confabulation_summary.json"
    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (confabulation_summary.json exists), skipping")
        return True

    script = SCRIPT_DIR / "biorag_confabulation_checker.py"
    cmd = [
        python, str(script),
        "--descriptions_dir", str(desc_dir),
        "--compiled_dir", str(base / "compiled"),
        "--output_dir", str(out_dir),
    ]
    ok = _run_step(cmd, "step9_confab_check", log_dir, cfg["dry_run"])
    if not ok:
        return False

    # Generate summary figures directly in the confabulation_report directory
    viz_script = SCRIPT_DIR / "biorag_confabulation_compare.py"
    viz_cmd = [
        python, str(viz_script),
        "--round1", str(out_dir),
        "--output_dir", str(out_dir),
        "--labels", "Round 1",
    ]
    logger.info("  Generating confabulation summary figures...")
    _run_step(viz_cmd, "step9_confab_figures", log_dir, cfg["dry_run"])
    return True


def step_confab_fix(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 10: Re-describe with prompt-constrained numeric fidelity (round 2).

    Uses a separate cache directory so round-1 descriptions are preserved.
    Only runs if Step 9 found actual confabulation errors (not just warnings).
    """
    base = Path(cfg["output_base"])

    r1_summary_path = base / "biorag_descriptions" / "confabulation_report" / "confabulation_summary.json"
    if r1_summary_path.exists():
        with open(r1_summary_path) as f:
            r1_summary = json.load(f)
        r1_errors = r1_summary.get("errors", 0)
        if r1_errors == 0:
            logger.info(f"  Step 9 found 0 confabulation errors (only {r1_summary.get('warnings', 0)} warnings) — skipping re-description")
            return True

    out_dir = base / "biorag_descriptions_r2"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = out_dir / "biorag_summary.json"
    if summary.exists() and not cfg["force"]:
        logger.info("  Already complete (biorag_summary.json exists), skipping")
        return True

    semi_dirs = [str(base / "semilandmarks")]

    script = SCRIPT_DIR / "biosyslit_rag_retrieval_prompt_constrained.py"
    cmd = [
        python, str(script), "describe-biorag",
        "--compiled-dir", str(base / "compiled"),
        "--semilandmarks-dir", *semi_dirs,
        "--image-dir", cfg["image_dir"],
        "--coco-json", cfg["coco_json"],
        "--output-dir", str(out_dir),
        "--vision-mode", "foreground",
        "--no-florence2",
        "--family", cfg.get("family", ""),
        "--k", str(cfg.get("k", 3)),
        "--model", cfg.get("model", "claude-sonnet-4-6"),
        "--group-labels", cfg["group_labels"],
        "--generate-key",
    ]
    if cfg.get("api_key_file"):
        cmd += ["--api-key-file", cfg["api_key_file"]]
    if cfg.get("pdf_dir"):
        cmd += ["--pdf-dir", cfg["pdf_dir"]]
    if cfg.get("user_prompts"):
        cmd += ["--user-prompts", cfg["user_prompts"]]
    if cfg.get("label_dir"):
        cmd += ["--label-dir", cfg["label_dir"]]

    return _run_step(cmd, "step10_confab_fix", log_dir, cfg["dry_run"])


def step_ontology(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 11: Auto-assign OBO Foundry ontology URIs to description terms."""
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    out_dir = desc_dir / "ontology_assignments"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_file = out_dir / "ontology_summary.json"
    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (ontology_summary.json exists), skipping")
        return True

    script = SCRIPT_DIR / "biorag_ontology_assigner.py"
    cmd = [
        python, str(script),
        "--descriptions_dir", str(desc_dir),
        "--output_dir", str(out_dir),
    ]
    return _run_step(cmd, "step11_ontology", log_dir, cfg["dry_run"])


def step_key_filter(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 12: Create human-measurable-only compiled data for key generation."""
    base = Path(cfg["output_base"])
    compiled_dir = base / "compiled"
    out_dir = base / "compiled_key"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_file = out_dir / "diaphorina_full_features.csv"
    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (filtered features CSV exists), skipping")
        return True

    script = SCRIPT_DIR / "biorag_key_feature_filter.py"
    cmd = [
        python, str(script),
        "--compiled_dir", str(compiled_dir),
        "--output_dir", str(out_dir),
    ]
    return _run_step(cmd, "step12_key_filter", log_dir, cfg["dry_run"])


def step_key_regen(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 13: Regenerate taxonomic key using filtered compiled directory.

    Deletes the cached key entry so BioRAG regenerates it, then runs
    BioRAG with --compiled-dir pointing to the filtered data. All
    species descriptions are cached from step 8/10, so only the key
    is regenerated.
    """
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    key_path = desc_dir / "taxonomic_key.txt"

    if key_path.exists() and not cfg["force"]:
        import json as _json
        try:
            with open(key_path) as f:
                text = f.read()
            comp_terms = ["cluster", "UMAP", "PCA", "GLCM", "LBP", "n_boundaries"]
            has_comp = any(t.lower() in text.lower() for t in comp_terms)
            if not has_comp:
                logger.info("  Key already regenerated (no computational features), skipping")
                return True
            logger.info("  Existing key has computational features, regenerating...")
        except Exception:
            pass

    # Delete the cached key so BioRAG regenerates it
    cache_dir = desc_dir / "biorag_cache" / "_all"
    cache_file = cache_dir / "_key_taxonomic.json"
    if cache_file.exists():
        import shutil
        backup = str(cache_file) + ".bak_pre_filtered"
        if not Path(backup).exists():
            shutil.copy2(str(cache_file), backup)
        cache_file.unlink()
        logger.info("  Deleted cached key entry for regeneration")

    # Back up existing key
    if key_path.exists():
        backup = str(key_path) + ".bak_pre_filtered"
        if not Path(backup).exists():
            import shutil
            shutil.copy2(str(key_path), backup)

    semi_dirs = [str(base / "semilandmarks")]
    filtered_compiled = base / "compiled_key"

    script = SCRIPT_DIR / "biosyslit_rag_retrieval.py"
    cmd = [
        python, str(script), "describe-biorag",
        "--compiled-dir", str(filtered_compiled),
        "--semilandmarks-dir", *semi_dirs,
        "--image-dir", cfg["image_dir"],
        "--coco-json", cfg["coco_json"],
        "--output-dir", str(desc_dir),
        "--vision-mode", "foreground",
        "--no-florence2",
        "--family", cfg.get("family", ""),
        "--k", str(cfg.get("k", 3)),
        "--model", cfg.get("model", "claude-sonnet-4-6"),
        "--group-labels", cfg["group_labels"],
        "--generate-key",
    ]
    if cfg.get("api_key_file"):
        cmd += ["--api-key-file", cfg["api_key_file"]]
    if cfg.get("pdf_dir"):
        cmd += ["--pdf-dir", cfg["pdf_dir"]]
    if cfg.get("user_prompts"):
        cmd += ["--user-prompts", cfg["user_prompts"]]
    if cfg.get("label_dir"):
        cmd += ["--label-dir", cfg["label_dir"]]

    return _run_step(cmd, "step13_key_regen", log_dir, cfg["dry_run"])


def step_key_num_check(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 14: Validate numeric thresholds in the taxonomic key."""
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    key_path = desc_dir / "taxonomic_key.txt"

    if not key_path.exists():
        logger.warning("  No taxonomic_key.txt found, skipping")
        return True

    out_dir = desc_dir / "key_check_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_file = out_dir / "key_check_report.json"
    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (key_check_report.json exists), skipping")
        return True

    script = SCRIPT_DIR / "biorag_key_checker.py"
    cmd = [
        python, str(script),
        "--key", str(key_path),
        "--compiled_dir", str(base / "compiled"),
        "--output_dir", str(out_dir),
    ]
    return _run_step(cmd, "step14_key_num_check", log_dir, cfg["dry_run"])


def step_key_correct(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 14b: Correct taxonomic key thresholds using numeric checker output.

    Reads key_check_report.json, feeds issues to VLM, writes corrected key,
    then re-runs the numeric checker to verify improvement.
    """
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    key_path = desc_dir / "taxonomic_key.txt"

    check_dir = desc_dir / "key_check_report"
    report_file = check_dir / "key_check_report.json"

    if not report_file.exists():
        logger.warning("  No key_check_report.json — run key_num_check first")
        return True

    # Skip if already corrected and no issues remain
    corrected_report = check_dir / "key_check_report_post_correction.json"
    if corrected_report.exists() and not cfg["force"]:
        logger.info("  Already corrected (post_correction report exists), skipping")
        return True

    # Check whether there are issues to fix
    import json as _json
    with open(report_file) as f:
        report = _json.load(f)
    n_issues = report.get("issues", 0)
    if n_issues == 0:
        logger.info("  No issues in checker report — key thresholds are correct")
        return True
    logger.info(f"  Found {n_issues} threshold issues — correcting via VLM")

    # Step 1: Correct the key
    script = SCRIPT_DIR / "biosyslit_rag_retrieval.py"
    cmd = [
        python, str(script), "correct-key",
        "--key-file", str(key_path),
        "--check-report", str(report_file),
        "--model", cfg.get("model", "claude-sonnet-4-6"),
    ]
    if cfg.get("api_key_file"):
        cmd += ["--api-key-file", cfg["api_key_file"]]

    ok = _run_step(cmd, "step14b_key_correct", log_dir, cfg["dry_run"])
    if not ok:
        return False

    # Step 2: Re-run numeric checker on corrected key
    logger.info("  Re-running numeric checker on corrected key...")
    check_script = SCRIPT_DIR / "biorag_key_checker.py"
    recheck_dir = check_dir
    # Move old report aside
    if report_file.exists():
        import shutil
        pre_corr = check_dir / "key_check_report_pre_correction.json"
        if not pre_corr.exists():
            shutil.copy2(str(report_file), str(pre_corr))
        report_file.unlink()

    compiled_dir = base / "compiled_key"
    if not compiled_dir.exists():
        compiled_dir = base / "compiled"

    cmd2 = [
        python, str(check_script),
        "--key", str(key_path),
        "--compiled_dir", str(compiled_dir),
        "--output_dir", str(recheck_dir),
    ]
    ok2 = _run_step(cmd2, "step14b_key_recheck", log_dir, cfg["dry_run"])
    if not ok2:
        return False

    # Rename new report for clarity
    if report_file.exists():
        import shutil
        shutil.copy2(str(report_file), str(corrected_report))
        # Log improvement
        with open(corrected_report) as f:
            new_report = _json.load(f)
        new_issues = new_report.get("issues", 0)
        logger.info(f"  Threshold correction: {n_issues} → {new_issues} issues "
                     f"({'%.0f' % (100 * (1 - new_issues / max(n_issues, 1)))}% reduction)")

    return True


def step_key_qual_check(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 15: Validate qualitative claims + compute E_Dicho efficiency."""
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    key_path = desc_dir / "taxonomic_key.txt"

    if not key_path.exists():
        logger.warning("  No taxonomic_key.txt found, skipping")
        return True

    out_dir = desc_dir / "key_qualitative_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_file = out_dir / "qualitative_check_report.json"
    if check_file.exists() and not cfg["force"]:
        logger.info("  Already complete (qualitative_check_report.json exists), skipping")
        return True

    script = SCRIPT_DIR / "biorag_key_qualitative_checker.py"
    cmd = [
        python, str(script),
        "--key", str(key_path),
        "--compiled_dir", str(base / "compiled"),
        "--output_dir", str(out_dir),
    ]
    return _run_step(cmd, "step15_key_qual_check", log_dir, cfg["dry_run"])


def step_confab_compare(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 16: Comparison figures/tables across confabulation checker rounds."""
    base = Path(cfg["output_base"])
    desc_dir = base / "biorag_descriptions"
    r1_dir = desc_dir / "confabulation_report"

    if not r1_dir.exists():
        logger.warning("  No confabulation_report found, skipping")
        return True

    out_dir = desc_dir / "confabulation_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_files = list(out_dir.glob("*.png"))
    if check_files and not cfg["force"]:
        logger.info("  Already complete (comparison figures exist), skipping")
        return True

    # Check if round 2 exists
    r2_dir = (base / "biorag_descriptions_r2" / "confabulation_report")
    r2_summary = r2_dir / "confabulation_summary.json"
    if not r2_summary.exists():
        # Run checker on r2 descriptions first
        r2_desc = base / "biorag_descriptions_r2"
        if r2_desc.exists():
            r2_dir = r2_desc / "confabulation_report"
            r2_dir.mkdir(parents=True, exist_ok=True)
            check_script = SCRIPT_DIR / "biorag_confabulation_checker.py"
            check_cmd = [
                python, str(check_script),
                "--descriptions_dir", str(r2_desc),
                "--compiled_dir", str(base / "compiled"),
                "--output_dir", str(r2_dir),
            ]
            logger.info("  Running confabulation checker on round 2 descriptions...")
            ok = _run_step(check_cmd, "step16_confab_check_r2", log_dir, cfg["dry_run"])
            if not ok:
                return False

    script = SCRIPT_DIR / "biorag_confabulation_compare.py"
    cmd = [
        python, str(script),
        "--round1", str(r1_dir),
        "--output_dir", str(out_dir),
    ]
    r2_summary = r2_dir / "confabulation_summary.json"
    if r2_summary.exists():
        cmd += ["--round2", str(r2_dir)]
        cmd += ["--labels", "Original", "Prompt-constrained"]
    else:
        cmd += ["--labels", "Round 1"]

    return _run_step(cmd, "step16_confab_compare", log_dir, cfg["dry_run"])


def step_key_compare(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 17: Key check comparison + qualitative check figures."""
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    out_dir = desc_dir / "key_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_files = list(out_dir.glob("*.png"))
    if check_files and not cfg["force"]:
        logger.info("  Already complete (key comparison figures exist), skipping")
        return True

    all_ok = True

    # Key qualitative figures + before/after comparison
    qual_report = desc_dir / "key_qualitative_check" / "qualitative_check_report.json"
    if qual_report.exists():
        fig_script = SCRIPT_DIR / "biorag_key_qualitative_figures.py"
        fig_cmd = [
            python, str(fig_script),
            "--report", str(qual_report),
            "--output_dir", str(out_dir),
        ]
        # Look for a "before" report from the original BioRAG run (step 8)
        r1_qual = base / "biorag_descriptions" / "key_qualitative_check" / "qualitative_check_report.json"
        if r1_qual.exists() and str(r1_qual) != str(qual_report):
            fig_cmd += ["--report_before", str(r1_qual)]
            logger.info("  Running qualitative figures with before/after comparison...")
        else:
            logger.info("  Running qualitative check figures...")
        ok = _run_step(fig_cmd, "step17_qual_figures", log_dir, cfg["dry_run"])
        if not ok:
            all_ok = False

    # Key numeric check comparison (needs at least 2 rounds)
    kc_r1 = desc_dir / "key_check_report"
    if kc_r1.exists():
        kc_script = SCRIPT_DIR / "biorag_key_compare.py"
        kc_cmd = [
            python, str(kc_script),
            "--round1", str(kc_r1),
            "--round2", str(kc_r1),
            "--labels", "Filtered key", "Filtered key",
            "--output_dir", str(out_dir),
        ]
        logger.info("  Running key check comparison...")
        ok = _run_step(kc_cmd, "step17_key_compare", log_dir, cfg["dry_run"])
        if not ok:
            all_ok = False

    return all_ok


def step_figure_plates(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 18: Generate per-species figure plates and inject figure references."""
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    plates_dir = desc_dir / "species_plates"

    check_files = list(plates_dir.glob("*/*_plate.png"))
    if check_files and not cfg["force"]:
        logger.info(f"  Already complete ({len(check_files)} plates exist), skipping")
        return True

    # Generate specimen plates
    plate_script = SCRIPT_DIR / "generate_species_plates.py"
    plate_cmd = [
        python, str(plate_script),
        "--coco_json", cfg["coco_json"],
        "--image_dir", cfg["image_dir"],
        "--group_labels", cfg["group_labels"],
        "--output_base", str(base),
        "--plates_dir", str(plates_dir),
        "--max_images_per_species", "8",
    ]
    logger.info("  Generating species plates...")
    ok = _run_step(plate_cmd, "step18_plates", log_dir, cfg["dry_run"])
    if not ok:
        return False

    if cfg.get("workflow", "v2") == "v2":
        return True          # v2: figures are referenced by build_species_treatment_docx_v2

    # Inject figure references into descriptions
    inject_script = SCRIPT_DIR / "inject_figure_refs.py"
    fig_out = desc_dir / "descriptions_with_figures"
    inject_cmd = [
        python, str(inject_script),
        "--descriptions_dir", str(desc_dir),
        "--plates_dir", str(plates_dir),
        "--output_dir", str(fig_out),
        "--start_figure", "11",
    ]
    logger.info("  Injecting figure references into descriptions...")
    _run_step(inject_cmd, "step18_inject_refs", log_dir, cfg["dry_run"])

    return True


def _active_descriptions_dir(base: Path) -> Path:
    """Return the most recent descriptions directory (r2 if exists, else r1)."""
    r2 = base / "biorag_descriptions_r2"
    if r2.exists() and (r2 / "biorag_cache").exists():
        return r2
    return base / "biorag_descriptions"


def step_figure_captions(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 19: Generate figure captions for all pipeline figures."""
    base = Path(cfg["output_base"])
    desc_dir = _active_descriptions_dir(base)
    plates_dir = desc_dir / "species_plates"
    captions_csv = base / "figure_captions.csv"

    if captions_csv.exists() and not cfg["force"]:
        logger.info(f"  Already complete ({captions_csv} exists), skipping")
        return True

    caption_script = SCRIPT_DIR / "generate_figure_captions.py"
    cmd = [
        python, str(caption_script),
        "--output_base", str(base),
        "--species_plates_dir", str(plates_dir),
        "--start_figure", "1",
        "--output_file", str(captions_csv),
    ]

    # Pass optional directories if they exist
    semi_dir = base.parent / "Diaphorina_semilandmarks"
    if not semi_dir.exists():
        semi_dir = base / "semilandmarks"
    if semi_dir.exists():
        cmd.extend(["--semilandmarks_dir", str(semi_dir)])

    colhom_dir = base.parent / "Diaphorina_color_homology"
    if not colhom_dir.exists():
        colhom_dir = base / "color_homology"
    if colhom_dir.exists():
        cmd.extend(["--color_homology_dir", str(colhom_dir)])

    texhom_dir = base.parent / "Diaphorina_texture_homology"
    if not texhom_dir.exists():
        texhom_dir = base / "texture_homology"
    if texhom_dir.exists():
        cmd.extend(["--texture_homology_dir", str(texhom_dir)])

    lmk_dir = base.parent / "Diaphorina_landmark_gpa_forewing"
    if not lmk_dir.exists():
        lmk_dir = base / "landmark_gpa"
    if lmk_dir.exists():
        cmd.extend(["--landmark_gpa_dir", str(lmk_dir)])

    logger.info("  Generating figure captions...")
    return _run_step(cmd, "step19_captions", log_dir, cfg["dry_run"])


# ═══════════════════════════════════════════════════════════════════════════════
# BioRAG v2 — EVIDENCE-TIERED KEY AND TREATMENTS (added 2026-09-17)
#   21 key_matrix      biorag_key_feature_filter_v2.py   Tier-1 data matrix
#   22 key_build       biorag_key_builder_v1.py          data-driven key (+SDD/JSON-LD)
#   23 describe_v2     biorag_description_refiner_v1.py  Diagnosis/Description/Remarks
#   24 treatments_docx build_species_treatment_docx_v2.py monograph DOCX
# Universal prompts: biorag_prompts/biorag_system_prompts_v2.txt
# Taxon specifics:   --taxon_profile (see biorag_prompts/taxon_profiles/TEMPLATE_taxon_profile.yaml)
# ═══════════════════════════════════════════════════════════════════════════════

def _v2_backend_args(cfg: Dict) -> List[str]:
    out = ["--llm-backend", cfg.get("llm_backend", "api"),
           "--taxon-profile", cfg["taxon_profile"]]
    if cfg.get("system_prompts"):
        out += ["--system-prompts", cfg["system_prompts"]]
    if cfg.get("cc_model"):
        out += ["--cc-model", cfg["cc_model"]]
    return out


def _require_profile(cfg: Dict) -> bool:
    if not cfg.get("taxon_profile"):
        logger.error("  --taxon_profile is required for the BioRAG v2 steps")
        return False
    return True


def step_annotation_screen(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 20.5: does every polygon sit on the specimen? Runs before the matrix is built, so a
    polygon drawn beside the structure never contributes a measurement or a colour. Nothing is
    deleted: the flags are merged with the manual exclusion list and handed to key_matrix."""
    base = Path(cfg["output_base"])
    out = base / "annotation_screen"
    if (out / "exclusion_list_merged.csv").exists() and not cfg["force"]:
        logger.info("  Already complete (exclusion_list_merged.csv exists), skipping")
        return True
    if not cfg.get("annotation_screen", True):
        logger.info("  annotation_screen is off in the config, skipping")
        return True
    cmd = [python, str(SCRIPT_DIR / "biorag_annotation_screen_v1.py"),
           "--coco", cfg["coco_json"], "--image_dir", cfg["image_dir"], "--out_dir", str(out),
           "--min_coverage", str(cfg.get("screen_min_coverage", 0.35))]
    if cfg.get("exclude_list"):
        cmd += ["--merge_with", cfg["exclude_list"]]
    if cfg.get("screen_parent"):
        cmd += ["--parent", cfg["screen_parent"]]
    ok = _run_step(cmd, "step20_5_annotation_screen", log_dir, cfg["dry_run"])
    summ = out / "annotation_screen_summary.json"
    if ok and summ.exists():
        r = json.loads(summ.read_text())
        logger.info(f"  {r.get('flagged')} of {r.get('annotations_tested')} annotations "
                    f"({r.get('flagged_percent')}%) do not sit on the specimen; "
                    f"{r.get('dimension_mismatches')} images have wrong dimensions in the COCO file")
    return ok


def step_key_matrix(cfg: Dict, python: str, log_dir: Path) -> bool:
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    out_dir = base / "compiled_key_tier"
    screened = base / "annotation_screen" / "exclusion_list_merged.csv"
    if screened.exists() and cfg.get("annotation_screen", True):
        # the manual list plus whatever the screen flagged (step annotation_screen)
        cfg = dict(cfg, exclude_list=str(screened))
        logger.info(f"  exclusion list: {screened} (manual entries + annotation screen)")
    if (out_dir / "specimen_matrix_long.csv").exists() and not cfg["force"]:
        logger.info("  Already complete (specimen_matrix_long.csv exists), skipping")
        return True
    cmd = [python, str(SCRIPT_DIR / "biorag_key_feature_filter_v2.py"),
           "--compiled_dir", cfg.get("compiled_dir") or str(base / "compiled"),
           "--output_dir", str(out_dir),
           "--taxon_profile", cfg["taxon_profile"]]
    if cfg.get("exclude_list"):
        cmd += ["--exclude_list", cfg["exclude_list"]]
    if cfg.get("exclude_flagged"):
        cmd += ["--exclude_flagged"]
    return _run_step(cmd, "step21_key_matrix", log_dir, cfg["dry_run"])


def step_key_build(cfg: Dict, python: str, log_dir: Path) -> bool:
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    out_dir = base / "key_v2"
    if (out_dir / "key_tree.json").exists() and not cfg["force"]:
        logger.info("  Already complete (key_tree.json exists), skipping")
        return True
    cmd = [python, str(SCRIPT_DIR / "biorag_key_builder_v1.py"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--output_dir", str(out_dir),
           "--model", cfg.get("model", "claude-sonnet-4-6")] + _v2_backend_args(cfg)
    if cfg.get("species"):
        cmd += ["--species"] + cfg["species"]
    return _run_step(cmd, "step22_key_build", log_dir, cfg["dry_run"])


def step_describe_v2(cfg: Dict, python: str, log_dir: Path) -> bool:
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    out_dir = base / "descriptions_v2"
    prior = cfg.get("prior_cache") or str(base / "biorag_descriptions" / "biorag_cache")
    cmd = [python, str(SCRIPT_DIR / "biorag_description_refiner_v1.py"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--compiled_dir", cfg.get("compiled_dir") or str(base / "compiled"),
           "--prior_cache", prior,
           "--key_tree", str(base / "key_v2" / "key_tree.json"),
           "--output_dir", str(out_dir),
           "--model", cfg.get("model", "claude-sonnet-4-6")] + _v2_backend_args(cfg)
    if cfg.get("localities"):
        cmd += ["--localities", cfg["localities"]]
    if cfg.get("species"):
        cmd += ["--species"] + cfg["species"]
    if cfg.get("repair_existing"):
        cmd += ["--repair_existing"]
    if cfg["force"]:
        cmd += ["--force"]
    return _run_step(cmd, "step23_describe_v2", log_dir, cfg["dry_run"])


def step_char_figures(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23.4: the tables and figures of the character investigation — repeatability against the
    two acceptance lines, congruence with the measured counterparts, the gating ablation if it was
    run, and each character set side by side. These are what the Methods and Results cite."""
    base = Path(cfg["output_base"])
    out = base / "descriptive_states"
    rel = out / "character_reliability.tsv"
    if not rel.exists() and not cfg["dry_run"]:
        logger.warning("  no character_reliability.tsv yet (run char_reliability) — skipping")
        return True
    cmd = [python, str(SCRIPT_DIR / "biorag_reliability_figures_v1.py"),
           "--reliability", str(rel), "--out_dir", str(out / "figures")]
    for flag, path in (("--repeatability", out / "repeatability.json"),
                       ("--scoring_report", out / "scoring_report.json"),
                       ("--ablation_dir", out / "gating_ablation")):
        if Path(path).exists():
            cmd += [flag, str(path)]
    final = cfg.get("novelty_dir") or (base / "novelty_gated")
    if (Path(final) / "calibration.json").exists():
        cmd += ["--final_run", str(final)]
    return _run_step(cmd, "step23_4_char_figures", log_dir, cfg["dry_run"])


def step_subjective_check(cfg: Dict, python: str, log_dir: Path) -> bool:
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    cmd = [python, str(SCRIPT_DIR / "biorag_subjective_checks_v1.py"),
           "--descriptions_dir", str(base / "descriptions_v2"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--prior_cache", cfg.get("prior_cache") or str(base / "biorag_descriptions" / "biorag_cache"),
           "--taxon-profile", cfg["taxon_profile"]]
    states = base / "descriptive_states" / "descriptive_states_by_specimen.tsv"
    if states.exists():
        cmd += ["--descriptive_matrix", str(states)]
    return _run_step(cmd, "step23b_subjective_check", log_dir, cfg["dry_run"])


def step_descriptive_states(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23.2: score the descriptive characters per specimen from the isolated-structure
    images — one state per character from a fixed vocabulary, no free text. A second pass over
    a sample (rescore_fraction) gives the repeatability the next step needs."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    roots = cfg.get("descriptive_image_roots") or cfg.get("color_output_dirs") or []
    if isinstance(roots, str):
        roots = [roots]
    if not roots:
        logger.warning("  no descriptive_image_roots in the config — skipping the per-specimen scoring")
        return True
    out = base / "descriptive_states"
    cmd = [python, str(SCRIPT_DIR / "biorag_descriptive_scoring_v1.py"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--taxon_profile", cfg["taxon_profile"],
           "--image_roots", *[str(r) for r in roots],
           "--out_dir", str(out)]
    if cfg.get("descriptive_model"):
        cmd += ["--model", cfg["descriptive_model"]]
    ok = _run_step(cmd, "step23_2_descriptive_states", log_dir, cfg["dry_run"])
    rep_json = out / "scoring_report.json"
    if ok and rep_json.exists() and not cfg["dry_run"]:
        r = json.loads(rep_json.read_text())
        logger.info(f"  scored {r.get('states', '?')} states over "
                    f"{r.get('specimens', '?')} specimens")
    return ok


def step_char_reliability(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23.3: how repeatable is each descriptive character, and does it agree with the
    measured metrics? Characters that do not repeat are flagged so the wording and the
    delimitation both know not to lean on them — which characters survive is taxon-specific."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    out = base / "descriptive_states"
    scoring = out / "descriptive_states_by_specimen.tsv"
    if not scoring.exists() and not cfg["dry_run"]:
        logger.warning("  no per-specimen states yet (run descriptive_states) — skipping")
        return True
    cmd = [python, str(SCRIPT_DIR / "biorag_character_reliability_v1.py"),
           "--scoring", str(scoring),
           "--taxon_profile", cfg["taxon_profile"],
           "--out_dir", str(out)]
    retest = cfg.get("descriptive_retest") or (out / "retest" / "descriptive_states_by_specimen.tsv")
    if Path(retest).exists():
        cmd += ["--rescoring", str(retest)]
    if cfg.get("compiled_dir"):
        cmd += ["--computed_dir", str(cfg["compiled_dir"])]
    elif (base / "compiled").exists():
        cmd += ["--computed_dir", str(base / "compiled")]
    if cfg.get("reliability_min_fine"):
        cmd += ["--min_fine", str(cfg["reliability_min_fine"])]
    if cfg.get("reliability_min_coarse"):
        cmd += ["--min_coarse", str(cfg["reliability_min_coarse"])]
    ok = _run_step(cmd, "step23_3_char_reliability", log_dir, cfg["dry_run"])
    rel = out / "character_reliability.json"
    if ok and rel.exists() and not cfg["dry_run"]:
        r = json.loads(rel.read_text())
        logger.info(f"  characters usable as scored: {len(r.get('use', []))}; "
                    f"usable coarsened: {len(r.get('use_coarse', []))}; "
                    f"flagged as not repeatable: {len(r.get('flag', []))}")
        if r.get("flag"):
            logger.info(f"  flagged for this taxon: {', '.join(r['flag'])} "
                        f"(kept out of the key and scored only as wording)")
    return ok


def step_calibrate(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 22.5: measure the cutoffs on this reference set rather than inheriting them.

    Leave one specimen out tells you whether there are enough specimens; leave one
    species out tells you which novelty rule works on this material and at what cost in
    false alarms. Both are properties of the taxon and its imaging, so they are measured
    here and written out as the numbers to use. Runs before any treatment is written, so
    a reference set too thin to support a key is discovered before the expensive steps."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    key_tree = base / "key" / "key_tree.json"
    # key_build writes to key_v2/; earlier runs wrote to key/. Take whichever exists (key_v2 first),
    # otherwise a fresh output folder skips its own calibration without saying why.
    for cand in (base / "key_v2" / "key_tree.json", base / "key" / "key_tree.json"):
        if cand.exists():
            key_tree = cand
            break
    else:
        if cfg["dry_run"]:
            key_tree = base / "key_v2" / "key_tree.json"
    matrix = base / "compiled_key_tier"
    if not key_tree.exists() and not cfg["dry_run"]:
        logger.warning("  no key yet (run key_build) — skipping calibration")
        return True
    out = base / "calibration"
    cmd = [python, str(SCRIPT_DIR / "biorag_calibrate_v1.py"),
           "--matrix_dir", str(matrix),
           "--taxon_profile", cfg["taxon_profile"],
           "--key_tree", str(key_tree),
           "--out_dir", str(out),
           "--python", python]
    rel = base / "descriptive_states" / "character_reliability.json"
    if rel.exists():
        cmd += ["--reliability", str(rel)]
    ok = _run_step(cmd, "step22_5_calibrate", log_dir, cfg["dry_run"])
    rep = out / "calibration.json"
    if ok and rep.exists() and not cfg["dry_run"]:
        r = json.loads(rep.read_text())
        rs = r.get("reference_set", {})
        logger.info(f"  reference set: {rs.get('species')} species, {rs.get('specimens')} specimens, "
                    f"median {rs.get('median_series')} per species")
        lo = r.get("leave_one_specimen_out", {})
        if lo.get("overall_correct") is not None:
            logger.info(f"  key identifies {100 * lo['overall_correct']:.0f}% of its own specimens "
                        f"when their record is withheld")
        logger.info(f"  novelty rule for this taxon: {r.get('recommended_rule')} "
                    f"(catches {r.get('recommended_rule_detection')}, "
                    f"false alarms {r.get('recommended_rule_false_alarm')})")
        for w in r.get("warnings", []):
            logger.warning(f"  {w}")
    return ok


def step_confab_check_v2(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23c: independent audit of the treatments against the data matrix."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    cmd = [python, str(SCRIPT_DIR / "biorag_confabulation_checker_v2.py"),
           "--descriptions_dir", str(base / "descriptions_v2"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--taxon_profile", cfg["taxon_profile"],
           "--key_tree", str(base / "key_v2" / "key_tree.json"),
           "--output_dir", str(base / "descriptions_v2" / "confabulation_report_v2")]
    if cfg.get("localities"):
        cmd += ["--localities", cfg["localities"]]
    ok = _run_step(cmd, "step23c_confab_check_v2", log_dir, cfg["dry_run"])
    rep = base / "descriptions_v2" / "confabulation_report_v2" / "confabulation_summary.json"
    if ok and rep.exists() and not cfg["dry_run"]:
        s = json.loads(rep.read_text())
        logger.info(f"  audit: {s['errors']} errors in {s['total_traits_checked']} claims "
                    f"({s['confabulation_rate']:.2f}%), {s['warnings']} warnings")
        if s["errors"] or s["warnings"]:
            logger.info("  repair them with: describe_v2 --repair_existing "
                        "(pipeline: --only_steps describe_v2 with repair_existing: true)")
    return ok


def step_type_material(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23.6b: build the Code-compliant type statements from the designations the collectors
    returned. Skipped silently when no designation sheet is configured, so the monograph keeps its
    "not yet designated" placeholder rather than inventing a holotype."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    types = cfg.get("type_designations")
    if not types or not Path(types).exists():
        logger.info("  no type_designations sheet configured — treatments keep the placeholder")
        return True
    loc = cfg.get("localities") or str(base / "localities" / "localities_verified.tsv")
    cmd = [python, str(SCRIPT_DIR / "biorag_type_material_v1.py"),
           "--types", str(types), "--localities", str(loc),
           "--taxon_profile", cfg["taxon_profile"],
           "--out_dir", str(base / "localities"), "--write_localities"]
    if cfg.get("types_fixed_by"):
        cmd += ["--fixed_by", cfg["types_fixed_by"]]
    if cfg.get("zoobank_lsid"):
        cmd += ["--zoobank", cfg["zoobank_lsid"]]
    ok = _run_step(cmd, "step23_6b_type_material", log_dir, cfg["dry_run"])
    rep_f = base / "localities" / "type_material.json"
    if ok and rep_f.exists() and not cfg["dry_run"]:
        r = json.loads(rep_f.read_text())
        logger.info(f"  type material: {r.get('ready', 0)} of {r.get('to_be_described', 0)} new "
                    f"species have a valid designation; {len(r.get('problems', []))} problems")
        for p_ in (r.get("problems") or [])[:5]:
            logger.warning(f"    {p_}")
        if r.get("not_ready"):
            logger.warning("  species without a valid type designation keep the placeholder — their "
                           "names cannot be published until this is resolved (ICZN Art. 16.4)")
    return ok


def step_char_gate(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23.65: reliability gate on the wording. A character whose state a second reading of the
    same image does not reproduce is not necessarily wrong — the structure may be just as described —
    but the reading cannot be offered as an observation a reader could confirm, so those statements
    come out of the diagnoses and descriptions and into the report. Which characters those are is
    measured per taxon in step 23.3, so this runs for one species or five hundred alike."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    rel = base / "descriptive_states" / "character_reliability.json"
    if not rel.exists() and not cfg["dry_run"]:
        logger.warning("  no character_reliability.json (run descriptive_states + char_reliability) "
                       "— skipping; nothing can be gated without the retest")
        return True
    desc = base / "descriptions_v2"
    # "python" (the default) makes the deletions by rule: deterministic, needs no account, and removes
    # only what it was told to. "llm" is available and reads better, but measured on the Diaphorina
    # treatments it also deleted statements about characters that had passed, in 47% of its edits —
    # invisible to both the subsequence check and the numeric audit. See BIORAG_V2_README.md §9d.
    engine = cfg.get("gate_engine", "python")
    if engine == "llm" and cfg.get("llm_backend") == "none":
        logger.info("  no LLM backend configured — editing by rule instead (--engine python)")
        engine = "python"
    cmd = [python, str(SCRIPT_DIR / "biorag_character_gate_v1.py"),
           "--descriptions_dir", str(desc),
           "--reliability", str(rel),
           "--taxon_profile", cfg["taxon_profile"],
           "--engine", engine,
           "--out_dir", str(desc / "character_gate")]
    if engine == "llm":
        cmd += _v2_backend_args(cfg)
    key = base / "key_v2" / "taxonomic_key.txt"
    if key.exists():
        cmd += ["--key", str(key)]
    # the deletions are deterministic; --rewrite is only a fallback for prose the rules cannot
    # edit safely, and is off unless asked for
    if cfg.get("gate_rewrite") and cfg.get("llm_backend") != "none":
        cmd += ["--rewrite"] + _v2_backend_args(cfg)
    if cfg.get("gate_coarsen"):
        cmd += ["--coarsen"]
    ok = _run_step(cmd, "step23_65_char_gate", log_dir, cfg["dry_run"])
    rep = desc / "character_gate" / "character_gate_report.json"
    if ok and rep.exists() and not cfg["dry_run"]:
        r = json.loads(rep.read_text())
        logger.info(f"  struck {len(r['characters']['flag'])} unreliable characters "
                    f"({', '.join(r['characters']['flag']) or 'none'}) from "
                    f"{r['species_edited']} treatments: "
                    + ", ".join(f"{k} {v}" for k, v in r.get("by_action", {}).items()))
        if r.get("key_lines_with_struck_words"):
            logger.warning(f"  {len(r['key_lines_with_struck_words'])} key lines still use a struck "
                           f"word — check the couplet wording by hand")
    # editing prose can strand a measurement from the label that says what it measures, so the
    # independent audit runs again over the edited text and the result is compared with the
    # audit before the gate. Anything new here is damage done by this step.
    if ok and not cfg["dry_run"] and r_changed(desc):
        after = desc / "confabulation_report_v2_after_gate"
        v = [python, str(SCRIPT_DIR / "biorag_confabulation_checker_v2.py"),
             "--descriptions_dir", str(desc), "--matrix_dir", str(base / "compiled_key_tier"),
             "--taxon_profile", cfg["taxon_profile"],
             "--key_tree", str(base / "key_v2" / "key_tree.json"),
             "--output_dir", str(after)]
        if cfg.get("localities"):
            v += ["--localities", cfg["localities"]]
        _run_step(v, "step23_65_char_gate_verify", log_dir, False)
        f0 = desc / "confabulation_report_v2" / "confabulation_summary.json"
        f1 = after / "confabulation_summary.json"
        if f1.exists():
            a1 = json.loads(f1.read_text())
            a0 = json.loads(f0.read_text()) if f0.exists() else {"errors": 0, "needs_review": 0}
            d_err = a1.get("errors", 0) - a0.get("errors", 0)
            d_rev = a1.get("needs_review", 0) - a0.get("needs_review", 0)
            if d_err > 0 or d_rev > 0:
                logger.error(f"  the gate introduced {d_err} errors and {d_rev} review cases — "
                             f"restore from the *_before_gate_* files and report it")
                return False
            logger.info(f"  re-audited after editing: {a1.get('errors', 0)} errors, "
                        f"{a1.get('needs_review', 0)} to review in "
                        f"{a1.get('total_traits_checked', 0)} claims (unchanged by the gate)")
    return ok


def r_changed(desc: Path) -> bool:
    """Did the gate actually edit anything? (a backup file dated today means yes)"""
    return any(desc.glob("*/*_treatment_before_gate_*.json"))


def step_ontology_v2(cfg: Dict, python: str, log_dir: Path) -> bool:
    """Step 23d: entity-quality ontology annotation from OBO releases."""
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    cmd = [python, str(SCRIPT_DIR / "biorag_ontology_annotator_v2.py"),
           "--descriptions_dir", str(base / "descriptions_v2"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--taxon_profile", cfg["taxon_profile"], "--download"]
    return _run_step(cmd, "step23d_ontology_v2", log_dir, cfg["dry_run"])


def step_treatments_docx(cfg: Dict, python: str, log_dir: Path) -> bool:
    if not _require_profile(cfg):
        return False
    base = Path(cfg["output_base"])
    if not any((base / "descriptions_v2").glob("*/*_treatment.json")):
        # descriptions are written by a vision-language model; without one (--llm_backend none)
        # the run ends at the data matrix, key and evidence sheets, and there is nothing to typeset
        logger.warning("  no species treatments to build: descriptions need a vision-language model "
                       "(--llm_backend claude-code or api). The data matrix, key and per-species "
                       "evidence sheets (descriptions_v2/<species>/) are complete.")
        return True
    out = base / "treatments" / "monograph_treatments.docx"
    cmd = [python, str(SCRIPT_DIR / "build_species_treatment_docx_v2.py"),
           "--descriptions-dir", str(base / "descriptions_v2"),
           "--key-dir", str(base / "key_v2"),
           "--matrix-dir", str(base / "compiled_key_tier"),
           "--taxon-profile", cfg["taxon_profile"],
           "--output", str(out)]
    if cfg.get("localities"):
        cmd += ["--localities", cfg["localities"]]
    else:
        cmd += ["--emit-locality-template", str(base / "treatments" / "localities_TEMPLATE.tsv")]
    plates = cfg.get("plates_dir") or str(_active_descriptions_dir(base) / "species_plates")
    cmd += ["--plates-dir", plates]
    if cfg.get("title"):
        cmd += ["--title", cfg["title"]]
    if cfg.get("authors"):
        cmd += ["--authors", cfg["authors"]]
    if not _run_step(cmd, "step24_treatments_docx", log_dir, cfg["dry_run"]):
        return False
    exp = [python, str(SCRIPT_DIR / "build_monograph_exports_v1.py"),
           "--descriptions_dir", str(base / "descriptions_v2"),
           "--key_dir", str(base / "key_v2"),
           "--matrix_dir", str(base / "compiled_key_tier"),
           "--taxon-profile", cfg["taxon_profile"],
           "--output_dir", str(base / "treatments" / "machine_readable"),
           "--plates_dir", plates]
    if cfg.get("localities"):
        exp += ["--localities", cfg["localities"]]
    if cfg.get("title"):
        exp += ["--title", cfg["title"]]
    if cfg.get("authors"):
        exp += ["--authors", cfg["authors"]]
    return _run_step(exp, "step24b_machine_readable_exports", log_dir, cfg["dry_run"])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_STEPS = {
    "coco_clean":      (0, step_coco_clean),
    "measurements":    (1, step_measurements),
    "semilandmarks":   (2, step_semilandmarks),
    "color":           (3, step_color_extraction),
    "color_homology":  (4, step_color_homology),
    "texture":         (5, step_texture_homology),
    "landmark_gpa":    (6, step_landmark_gpa),
    "compile":         (7, step_compile),
    "biorag":          (8, step_biorag),
    "confab_check":    (9, step_confab_check),
    "confab_fix":      (10, step_confab_fix),
    "ontology":        (11, step_ontology),
    "key_filter":      (12, step_key_filter),
    "key_regen":       (13, step_key_regen),
    "key_num_check":   (14, step_key_num_check),
    "key_correct":     (15, step_key_correct),
    "key_qual_check":  (16, step_key_qual_check),
    "confab_compare":  (17, step_confab_compare),
    "key_compare":     (18, step_key_compare),
    "figure_plates":   (19, step_figure_plates),
    "figure_captions": (20, step_figure_captions),
    "annotation_screen": (20.5, step_annotation_screen),
    "key_matrix":      (21, step_key_matrix),
    "key_build":       (22, step_key_build),
    "calibrate":       (22.5, step_calibrate),
    "describe_v2":     (23, step_describe_v2),
    "descriptive_states": (23.2, step_descriptive_states),
    "char_reliability": (23.3, step_char_reliability),
    "char_figures": (23.4, step_char_figures),
    "subjective_check": (23.5, step_subjective_check),
    "confab_check_v2":  (23.6, step_confab_check_v2),
    "type_material":    (23.62, step_type_material),
    "char_gate":        (23.65, step_char_gate),
    "ontology_v2":      (23.7, step_ontology_v2),
    "treatments_docx": (24, step_treatments_docx),
}
V2_STEPS = ["annotation_screen", "key_matrix", "key_build", "calibrate", "describe_v2", "descriptive_states", "char_reliability",
            "char_figures", "subjective_check", "confab_check_v2", "type_material", "char_gate",
            "ontology_v2", "treatments_docx"]
V2_WORKFLOW = ["coco_clean", "measurements", "semilandmarks", "color", "color_homology", "texture",
               "landmark_gpa", "compile", "annotation_screen", "key_matrix", "biorag", "key_build", "calibrate", "describe_v2",
               "descriptive_states", "char_reliability", "char_figures", "subjective_check",
               "confab_check_v2", "type_material", "char_gate",
               "ontology_v2", "figure_plates", "treatments_docx"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Descriptron → BioRAG Full Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--coco_json", required=True,
                    help="Master COCO JSON with segmentation annotations")
    p.add_argument("--image_dir", required=True,
                    help="Directory containing specimen images")
    p.add_argument("--group_labels", required=True,
                    help="CSV mapping image filenames to species groups")
    p.add_argument("--output_base", required=True,
                    help="Base output directory (subdirs created automatically)")
    p.add_argument("--family", default="",
                    help="Taxonomic family name (for BioRAG)")
    p.add_argument("--api_key_file", default=None,
                    help="Claude API key file (required for BioRAG step)")

    p.add_argument("--keypoints_json", default=None,
                    help="Separate COCO keypoints JSON (for landmark GPA)")
    p.add_argument("--extra_keypoints_json", nargs="*", default=[],
                   help="Further landmark files with their own landmark scheme (e.g. head landmarks beside wing "
                        "landmarks); each gets its own GPA in landmark_gpa_<file stem> and is compiled with the rest")
    p.add_argument("--pdf_dir", default=None,
                    help="Literature PDFs directory (for BioRAG RAG)")
    p.add_argument("--rag_index", default=None,
                    help="Prebuilt literature index JSON (BioSysLit and/or PDFs), from "
                         "'biosyslit_rag_retrieval_v2 index --taxon <taxon> [--pdf-dir <dir>] --output <file>'; "
                         "used instead of --pdf_dir")
    p.add_argument("--user_prompts", default=None,
                    help="User prompt questions file (.docx/.csv)")
    p.add_argument("--exclude_list", default=None,
                    help="Outlier exclusion CSV (for compile step)")
    p.add_argument("--grouping_file", default=None,
                    help="Grouping file for measurement UMAP (often same as group_labels)")
    p.add_argument("--label_dir", default=None,
                    help="Directory with label images for Materials Examined (Florence-2 OCR)")

    p.add_argument("--num_landmarks", type=int, default=100,
                    help="Number of semilandmarks (default 100)")
    p.add_argument("--slide_method", choices=["none", "procd", "bending"], default="none",
                    help="Semilandmark sliding as in geomorph (V42): none = fixed (default), procd = minimum "
                         "Procrustes distance, bending = minimum bending energy")
    p.add_argument("--alpha", type=float, default=0.05,
                    help="Significance threshold for diagnostic features (default 0.05)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                    help="VLM model for BioRAG (default claude-sonnet-4-6)")
    p.add_argument("--k", type=int, default=3,
                    help="Number of RAG literature retrievals (default 3)")
    p.add_argument("--species", nargs="*", default=None,
                    help="Specific species to describe (default all)")
    p.add_argument("--bg_normalize", action="store_true", default=True,
                    help="Enable background chromaticity correction (default on)")
    p.add_argument("--no_bg_normalize", action="store_false", dest="bg_normalize")

    p.add_argument("--ratio_config", default=None,
                    help="TSV defining taxonomic ratios for compile step")
    p.add_argument("--species_pattern", default=None,
                    help="Regex for specimen ID extraction in ratio computation")
    p.add_argument("--skip_steps", nargs="*", default=[],
                    choices=STEP_NAMES, help="Steps to skip")
    p.add_argument("--only_steps", nargs="*", default=None,
                    choices=STEP_NAMES, help="Run only these steps")
    p.add_argument("--force", action="store_true",
                    help="Force re-run even if outputs exist")
    p.add_argument("--conda_env", default="measure_env",
                    help="Conda environment name (default measure_env)")
    # --- BioRAG v2 (evidence-tiered) options ---
    p.add_argument("--workflow", default="v2", choices=["v2", "v1"],
                   help="v2 (default): evidence-tiered, data-driven key; v1: original LLM-key workflow")
    p.add_argument("--keep_flagged", action="store_true",
                   help="v2: keep values flagged by the outlier/label-swap screen (default: exclude them)")
    p.add_argument("--v2_only", action="store_true",
                   help="Run only the BioRAG v2 steps (key_matrix, key_build, describe_v2, treatments_docx)")
    p.add_argument("--taxon_profile", default=None,
                   help="Taxon profile YAML (names, structures, ratios, questions) — required for v2 steps")
    p.add_argument("--system_prompts", default=None,
                   help="Universal system-prompt file (default biorag_prompts/biorag_system_prompts_v2.txt)")
    p.add_argument("--llm_backend", default="api", choices=["api", "claude-code", "none"],
                   help="LLM backend for v2 steps: Anthropic API or Claude Code subscription")
    p.add_argument("--cc_model", default=None, help="Model alias for the claude-code backend")
    p.add_argument("--compiled_dir", default=None,
                   help="Existing compiled data dir (default <output_base>/compiled)")
    p.add_argument("--prior_cache", default=None,
                   help="Earlier BioRAG/TOWLEY cache with image observations")
    p.add_argument("--localities", default=None, help="Verified localities TSV")
    p.add_argument("--plates_dir", default=None, help="Consolidated plates dir with plate_index.tsv")
    p.add_argument("--exclude_flagged", action="store_true",
                   help="Drop values flagged by the outlier / label-swap screen")
    p.add_argument("--no_annotation_screen", action="store_true",
                   help="Skip step 20.5 (does every polygon sit on the specimen?). The screen assumes "
                        "transmitted light, i.e. a specimen darker than its background")
    p.add_argument("--screen_min_coverage", type=float, default=0.35,
                   help="An annotation with less than this share of its area on the specimen is "
                        "left out of the matrix")
    p.add_argument("--screen_parent", default=None,
                   help="Category every other annotation on the same image should lie inside "
                        "(e.g. whole_wing); optional")
    p.add_argument("--repair_existing", action="store_true",
                   help="v2 describe step: audit existing treatments and repair only the flagged problems "
                        "(instead of rewriting them)")
    p.add_argument("--title", default=None)
    p.add_argument("--authors", default=None)
    # --- character reliability and the wording gate (steps 23.2-23.65) ---
    p.add_argument("--descriptive_image_roots", nargs="*", default=None,
                    help="Folders of isolated-structure images (the semilandmark output) that the "
                         "per-specimen character scoring reads. Without these, steps 23.2-23.65 are skipped")
    p.add_argument("--descriptive_model", default=None,
                    help="Model for the per-specimen character scoring (default: --model)")
    p.add_argument("--descriptive_retest", default=None,
                    help="A second scoring pass to compare against (default: <out>/descriptive_states/retest, "
                         "which the scoring step writes automatically)")
    p.add_argument("--reliability_min_fine", type=float, default=None,
                    help="Keep a character as scored when this share of states repeats exactly (default 0.85)")
    p.add_argument("--reliability_min_coarse", type=float, default=None,
                    help="Keep a character at band level when this share of bands repeats (default 0.80)")
    p.add_argument("--gate_engine", default=None, choices=["python", "llm"],
                    help="How the unreproducible statements are removed from the treatments: 'python' "
                         "(default) edits by rule and removes only what it matched; 'llm' asks a model, "
                         "accepting only replies that are the original with words deleted")
    p.add_argument("--gate_coarsen", action="store_true",
                    help="Also replace a band-level character's word with its band name. Off by default: "
                         "the bands are a scoring device and their names do not always carry the same "
                         "sense in prose")
    p.add_argument("--type_designations", default=None,
                    help="The completed type-designation sheet from the collaborator workbook. "
                         "Without it the treatments keep the 'not yet designated' placeholder")
    p.add_argument("--types_fixed_by", default=None,
                    help="The publication making the designations, e.g. 'Serbina & Van Dam, 2026'")
    p.add_argument("--zoobank_lsid", default=None, help="ZooBank LSID of that work, if registered")
    p.add_argument("--novelty_dir", default=None,
                    help="A novelty run to summarise in the character figures (default: <out>/novelty_gated)")
    p.add_argument("--dry_run", action="store_true",
                    help="Print commands without executing")

    return p.parse_args()


_PATH_ARGS = ("coco_json", "image_dir", "group_labels", "output_base", "api_key_file", "keypoints_json",
              "pdf_dir", "rag_index", "user_prompts", "exclude_list", "grouping_file", "label_dir", "ratio_config",
              "taxon_profile", "system_prompts", "compiled_dir", "prior_cache", "localities", "plates_dir",
              "descriptive_retest", "novelty_dir", "type_designations")


def main():
    args = parse_args()
    # steps run with cwd = gui/, so make every user-supplied path absolute first
    for a in _PATH_ARGS:
        v = getattr(args, a, None)
        if isinstance(v, str) and v:
            setattr(args, a, str(Path(v).expanduser().resolve()))

    base = Path(args.output_base)
    base.mkdir(parents=True, exist_ok=True)
    log_dir = base / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = base / "pipeline_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

    # v2.0.4: the API key comes from the environment, the credentials file, or (first use, interactive
    # only) a prompt; asked here once so every step inherits it. Never logged.
    if args.llm_backend == "api":
        try:
            import descriptron_credentials as _creds
            _creds.load_into_environment()
            if not _creds.ensure_key("ANTHROPIC_API_KEY"):
                logger.warning("  --llm_backend api but no ANTHROPIC_API_KEY (environment, %s, or prompt): "
                               "the description steps will fail; use --llm_backend none for the data-only "
                               "outputs", _creds.credentials_path())
        except ImportError:
            pass

    logger.info("=" * 70)
    logger.info("Descriptron → BioRAG Full Pipeline")
    logger.info("=" * 70)
    logger.info(f"  COCO JSON:    {args.coco_json}")
    logger.info(f"  Image dir:    {args.image_dir}")
    logger.info(f"  Group labels: {args.group_labels}")
    logger.info(f"  Output base:  {args.output_base}")
    logger.info(f"  Family:       {args.family}")
    logger.info(f"  Dry run:      {args.dry_run}")
    logger.info(f"  Force:        {args.force}")

    for path_attr in ["coco_json", "image_dir", "group_labels"]:
        p = getattr(args, path_attr)
        if not Path(p).exists():
            logger.error(f"Path does not exist: {p}")
            sys.exit(1)

    cfg = vars(args)

    config_path = base / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Config saved: {config_path}")

    python = _python_bin(args.conda_env)
    logger.info(f"  Python:       {python}")

    categories = _discover_categories(args.coco_json)
    logger.info(f"  Categories:   {', '.join(categories)}")

    order = V2_WORKFLOW if args.workflow == "v2" else [n for n in STEP_NAMES if n not in V2_STEPS]
    if args.workflow == "v2":
        cfg["exclude_flagged"] = bool(args.exclude_flagged or not args.keep_flagged)
        cfg["annotation_screen"] = not args.no_annotation_screen
        cfg["screen_min_coverage"] = args.screen_min_coverage
        cfg["screen_parent"] = args.screen_parent
        if not args.taxon_profile:
            draft = base / "taxon_profile_DRAFT.yaml"
            if not draft.exists():
                cmd = [python, str(SCRIPT_DIR / "biorag_make_taxon_profile_v1.py"),
                       "--coco_json", args.coco_json, "--group_labels", args.group_labels,
                       "--family", args.family or "", "--output", str(draft)]
                if args.keypoints_json:
                    cmd += ["--keypoints_json", args.keypoints_json]
                if args.user_prompts:
                    cmd += ["--user_prompts", args.user_prompts]
                _run_step(cmd, "step00_draft_taxon_profile", log_dir, False)
            logger.warning(f"  No --taxon_profile given: using the auto-generated DRAFT {draft}. "
                           f"Edit names, status, sex-specific structures and ratios, then re-run with "
                           f"--taxon_profile {draft}")
            cfg["taxon_profile"] = args.taxon_profile = str(draft)
        # record the effective v2 settings (defaults applied above) for reproducibility
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if args.v2_only:
        steps_to_run = set(V2_STEPS)
    elif args.only_steps:
        steps_to_run = set(args.only_steps)
    else:
        steps_to_run = set(STEP_NAMES) - set(args.skip_steps)

    if not args.taxon_profile and steps_to_run & set(V2_STEPS):
        logger.info("  No --taxon_profile given: BioRAG v2 steps are skipped")
        steps_to_run -= set(V2_STEPS)

    results = {}
    for step_name in order:
        step_num, step_fn = ALL_STEPS[step_name]
        if step_name not in steps_to_run:
            logger.info(f"\n[Step {step_num}] {step_name} — SKIPPED (not selected)")
            continue

        logger.info(f"\n[Step {step_num}] {step_name}")
        logger.info("-" * 50)

        t0 = time.time()
        ok = step_fn(cfg, python, log_dir)
        elapsed = time.time() - t0

        results[step_name] = {"ok": ok, "elapsed": elapsed}

        if not ok:
            logger.error(f"\n  Step {step_name} FAILED after {elapsed:.0f}s")
            logger.error(f"  Pipeline halted. Fix the error and re-run.")
            logger.error(f"  Completed steps will be skipped on re-run (resume).")
            sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)
    for name, r in results.items():
        status = "OK" if r["ok"] else "FAILED"
        logger.info(f"  {name:20s} {status:6s} ({r['elapsed']:.0f}s)")
    logger.info(f"\n  Output: {args.output_base}")


if __name__ == "__main__":
    main()
