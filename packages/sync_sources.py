#!/usr/bin/env python3
"""
sync_sources.py — copy the working scripts into the package trees
=================================================================

`segment-anything-2/gui/` stays the single source of truth. The three
distributions are built *from* it rather than being a second copy that has to be
kept in step by hand, which is the usual way a packaged fork of a working system
starts to drift.

Which script goes where is not decided here: it is read from the script
inventory TSV that the supplementary figure generator produces, which derives
the assignment from what each script imports. So the packaging, the figure and
the code cannot disagree.

    python sync_sources.py --gui_dir ../segment-anything-2/gui \
        --inventory ~/Desktop/Towley_paper/figures_supplementary/FigS1_script_inventory.tsv

Run it again after editing anything in gui/ and rebuild the wheels.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

# where each distribution's programs land
TARGETS = {
    "descriptron-core":   ("descriptron-core", "descriptron_core"),
    "descriptron-vision": ("descriptron-vision", "descriptron_vision"),
    "descriptron-gui":    ("descriptron-gui", "descriptron_gui"),
}

# things the inventory does not list but the packages need
EXTRA_VISION = ["torchvision_det",                   # whole directory
                "dinov3_landmark_transfer_v52.py",    # v52 (orientation search) is not in the inventory yet
                "descriptron_video_track.py"]         # v2.1.0: video - individuals + pose (SAM2 video)
# modules a core program imports that the inventory files elsewhere or does not list:
# biosyslit_rag_retrieval_v2 (core) wraps biosyslit_rag_retrieval (listed under vision
# because it CAN use Florence-2, which it loads lazily), and that imports
# descriptron_rosetta. Without these the literature RAG dies on import in core.
EXTRA_CORE = ["measure/biosyslit_rag_retrieval.py", "measure/descriptron_rosetta.py",
              "measure/biorag_provenance_v1.py",         # v2.1.3: provenance stamps written by run_full_pipeline_v2
              "measure/descriptron_phylo.py",            # v2.2.0: phylogenetic signal, PGLS, ancestral states, phylomorphospace
              "measure/descriptron_trait_stats.py",      # v2.3.0: PERMANOVA, pairwise separation, identification, allometry
              "measure/validation/validate_trait_stats.py",   # v2.3.0: vs vegan/class/prop.test/binom.test/chisq.test
              "measure/validation/validate_phylo_real_tree.py",          # v2.2.0: vs geomorph/ape/phytools (plethspecies)
              "measure/validation/validate_phylo_traits.py",             # v2.2.0: vs geomorph/ape/phytools, any trait sets
              "measure/validation/validate_phylo_published_waldron2025.py",   # v2.2.0: published K, 57 Plethodon
              "measure/descriptron_credentials.py",      # v2.0.4: API keys / tokens (core + DINOLand + GUI)
              # v2.1.0: converters, COCO housekeeping, centre lines, joints, metadata + shape statistics
              "measure/descriptron_convert.py", "measure/descriptron_coco_tools.py",
              "measure/descriptron_centerline.py", "measure/descriptron_joints.py",
              "measure/descriptron_metadata.py", "measure/descriptron_shape_stats.py",
              "measure/validation/validate_shape_stats_vs_rrpp.py",
              "coco_combiner_V13.py", "coco_converter_v24.py",   # run by descriptron_coco_tools prepare-d2
              "measure/landmark_gpa_V2.py",      # V2: mirror-image specimens reflected before GPA
              "measure/semi_landmark_and_kpts_procrustesV42_GPA.py",   # V42: same for outlines
              "measure/validation/validate_shape_stats_vs_geomorph.py",
              "measure/validation/validate_semilandmarks_vs_geomorph.py",
              "measure/validation/validate_pipeline_gm_vs_geomorph.py",
              "measure/validation/validate_centerline_lengths.py",   # v2.1.2
              "measure/validation/validate_texture_glcm.py",         # v2.1.2
              "measure/descriptron_reexamine_v1.py",                 # v2.6.0: specimens to re-examine + map (step 22.6)
              "measure/descriptron_character_signal_v1.py",          # v2.6.0: per-character table + phylogenetic signal (step 21.1)
              "measure/descriptron_phylo_figure_v1.py",              # v2.6.0: phylogenetic summary figure (step 7.2)
              "measure/validation/dna_vs_morphology_v1.py",          # v2.6.0: barcode gap, DNA groups vs morphospecies
              "measure/validation/coi_species_tree_v1.py",           # v2.6.0: species tree from a barcode gene tree
              "measure/descriptron_reexamine_v2.py",                 # v2.7.0: interactive re-examination map v2
              "measure/generate_species_plates.py",                  # v2.7.0: species plates (step 18; never shipped before)
              "measure/descriptron_labelled_plates_v1.py",           # v2.7.0: labelled plates in the specimens' own colours
              "measure/descriptron_check_cross_image_copies_v1.py",  # v2.7.0: finds annotations copied onto other images
              "measure/biorag_coded_states_from_coco_v1.py",         # v2.7.5: descriptive characters recorded in the GUI -> matrix
              "descriptron_descriptive_characters.json",             # v2.7.5: its vocabulary (types and labels)
              "measure/descriptron_community_v1.py",                 # v2.7.6: phylogeny vs treatment; community phylogenetics
              "measure/validation/validate_community_vs_r.py",       # v2.7.6: its check against vegan / ape
              "measure/validation/validate_community_run_vs_r.py"]   # v2.7.7: one run checked against vegan / ape / phytools, number by number
EXTRA_GUI = ["marmot.jpg", "icons"]
EXTRA_GUI_TOOLS = ["descriptron-v2-v74.py", "descriptron-v2-v75.py", "descriptron-v2-v76.py",
                   "descriptron-v2-v77.py", "descriptron-v2-v78.py",   # v2.5.0: SAM 3 dialog (v77); drag and drop + SAM 3 prompts (v78)
                   "descriptron-v2-v79.py",                            # v2.5.1: finds descriptron-sam3 / DESCRIPTRON_SAM3_PYTHON
                   "descriptron-v2-v80.py",                            # v2.5.2: finds/downloads the SAM2 checkpoint; any SAM2 install
                   "descriptron-v2-v81.py",                            # v2.7.0: annotations load only onto their own image; folder masks kept
                   "descriptron-v2-v82.py",                            # v2.7.5: descriptive characters per structure; rename/delete categories
                   "descriptron_descriptive_characters.py", "descriptron_descriptive_characters.json",
                   "descriptron_category_edit.py",
                   "descriptron-v2-v83.py",                            # v2.7.6: Community & phylogeny button
                   "descriptron_sam3_instances.py"]                    # v2.5.0: run by the GUI in the separate sam3 env
# v2.5.1: SAM 3 as its own distribution (Python 3.12, sam3 from PyPI). The text vocabulary goes beside the wrapper
# in BOTH places it runs from: the sam3 wheel leaves it out.
SAM3_FILES = ["descriptron_sam3_instances.py"]
SAM3_DIRS = ["sam3_assets"]   # v2.1.2: v75 = v74 + Keypoint R-CNN train/predict; cli.py runs the newest descriptron-v2-*.py
DATA_FOR_CORE = ["measure/biorag_prompts"]


def find(gui_dir: Path, name: str) -> Path | None:
    for candidate in (gui_dir / "measure" / name, gui_dir / name):
        if candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gui_dir", required=True)
    ap.add_argument("--inventory", required=True,
                    help="FigS1_script_inventory.tsv — supplies the script -> package map")
    ap.add_argument("--clean", action="store_true",
                    help="empty each tools/ directory first, so a removed script really goes")
    a = ap.parse_args()

    gui = Path(a.gui_dir).resolve()
    rows = list(csv.DictReader(open(a.inventory), delimiter="\t"))
    if not rows or "pip_package" not in rows[0]:
        raise SystemExit("inventory has no pip_package column — regenerate it with "
                         "generate_script_inventory_figure_v3.py")

    counts = {}
    for dist, (pkg_dir, module) in TARGETS.items():
        tools = HERE / pkg_dir / "src" / module / "tools"
        if a.clean and tools.exists():
            shutil.rmtree(tools)
        tools.mkdir(parents=True, exist_ok=True)
        (HERE / pkg_dir / "src" / module / "__init__.py").touch(exist_ok=True)

        n = 0
        for row in rows:
            if row.get("pip_package") != dist:
                continue
            src = find(gui, row["script"])
            if src is None:
                print(f"  ! missing: {row['script']}")
                continue
            shutil.copy2(src, tools / src.name)
            n += 1
        counts[dist] = n

    # the torchvision detectors are a directory, not a single script
    vis_tools = HERE / "descriptron-vision" / "src" / "descriptron_vision" / "tools"
    for extra in EXTRA_VISION:
        src = gui / extra
        if src.is_dir():
            dst = vis_tools / extra
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.bak*", "tests"))
            n = len(list(dst.rglob("*.py")))
            counts["descriptron-vision"] += n
            print(f"  + {extra}/ ({n} modules) -> descriptron-vision")
        elif src.is_file():
            shutil.copy2(src, vis_tools / extra)
            counts["descriptron-vision"] += 1
            print(f"  + {extra} -> descriptron-vision")

    core_tools = HERE / "descriptron-core" / "src" / "descriptron_core" / "tools"
    for extra in EXTRA_CORE:
        src = gui / extra
        if src.is_file():
            shutil.copy2(src, core_tools / src.name)
            print(f"  + {src.name} -> descriptron-core (imported by core programs)")
        else:
            print(f"  ! missing: {extra}")

    # Prompts, taxon profiles, ontology releases and schemas.
    #
    # They go in TWO places, and the second one is not redundant. Fifteen programs
    # resolve their defaults as `Path(__file__).parent / "biorag_prompts" / ...`,
    # i.e. beside themselves — which after packaging means inside tools/. Shipping
    # them only as data/ leaves `biorag-key` dying with
    #   FileNotFoundError: .../tools/biorag_prompts/biorag_system_prompts_v2.txt
    # A real copy, not a symlink: a symlink does not survive a wheel.
    core_pkg = HERE / "descriptron-core" / "src" / "descriptron_core"
    for rel in DATA_FOR_CORE:
        src = gui / rel
        if not src.is_dir():
            continue
        for where in ("data", "tools"):
            dst = core_pkg / where / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.bak*"))
        size = sum(f.stat().st_size for f in (core_pkg / "data" / src.name).rglob("*")
                   if f.is_file())
        print(f"  + {rel} -> core data/ and tools/ ({size/1e6:.1f} MB each)")

    gui_tools = HERE / "descriptron-gui" / "src" / "descriptron_gui" / "tools"
    for extra in EXTRA_GUI_TOOLS:
        src = gui / extra
        if src.is_file():
            shutil.copy2(src, gui_tools / src.name)
            counts["descriptron-gui"] += 1
            print(f"  + {extra} -> descriptron-gui")
        else:
            print(f"  ! missing: {extra}")

    for extra in EXTRA_GUI:
        src = gui / extra
        dst = HERE / "descriptron-gui" / "src" / "descriptron_gui" / "data"
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                target = dst / src.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, dst / src.name)
            print(f"  + {extra} -> descriptron-gui")

    sam3_tools = HERE / "descriptron-sam3" / "src" / "descriptron_sam3" / "tools"
    if a.clean and sam3_tools.exists():
        shutil.rmtree(sam3_tools)
    sam3_tools.mkdir(parents=True, exist_ok=True)
    counts["descriptron-sam3"] = 0
    for f in SAM3_FILES:
        shutil.copy2(gui / f, sam3_tools / f)
        counts["descriptron-sam3"] += 1
        print(f"  + {f} -> descriptron-sam3")
    for d in SAM3_DIRS:
        for dst_tools in (sam3_tools, gui_tools):
            dst = dst_tools / d
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(gui / d, dst, ignore=shutil.ignore_patterns("*.bak*"))
        print(f"  + {d}/ -> descriptron-sam3 and descriptron-gui")

    print()
    for dist, n in counts.items():
        print(f"{dist:20} {n:>3} programs")


if __name__ == "__main__":
    main()
