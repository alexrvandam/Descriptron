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
              "measure/validation/validate_semilandmarks_vs_geomorph.py"]
EXTRA_GUI = ["marmot.jpg", "icons"]
EXTRA_GUI_TOOLS = ["descriptron-v2-v74.py"]   # v2.1.0 GUI (binder tabs); cli.py runs the newest descriptron-v2-*.py
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

    print()
    for dist, n in counts.items():
        print(f"{dist:20} {n:>3} programs")


if __name__ == "__main__":
    main()
