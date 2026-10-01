"""descriptron_sam3.cli - run descriptron_sam3_instances.py (shipped in tools/, with SAM 3's text vocabulary)."""
from __future__ import annotations

import runpy
import sys
from importlib.resources import files
from pathlib import Path


def main() -> None:
    path = Path(str(files("descriptron_sam3") / "tools" / "descriptron_sam3_instances.py"))
    if not path.is_file():
        raise SystemExit(f"{path} is missing - was sync_sources.py run before building?")
    sys.argv = [str(path), *sys.argv[1:]]
    runpy.run_path(str(path), run_name="__main__")
