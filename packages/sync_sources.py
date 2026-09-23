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
EXTRA_VISION = ["torchvision_det"]                    # whole directory
EXTRA_GUI = ["marmot.jpg", "icons"]
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
