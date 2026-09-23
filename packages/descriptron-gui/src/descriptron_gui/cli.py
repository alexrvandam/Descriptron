"""
descriptron_gui.cli — launch the annotation GUI
===============================================

The GUI drives the analysis programs directly, so this package depends on core
*and* vision; it is not a thin front end over them.

tkinter is in the standard library but is not always packaged with Python on
Linux, so a missing import is caught here and explained rather than presented as
a traceback.
"""
from __future__ import annotations

import os
import runpy
import sys
from importlib.resources import files
from pathlib import Path


def main() -> None:
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "tkinter is missing. It ships with Python on Windows and macOS; on Linux:\n"
            "  Debian/Ubuntu:  sudo apt install python3-tk\n"
            "  Fedora/RHEL:    sudo dnf install python3-tkinter\n"
            f"(original error: {exc})") from exc

    tools = Path(str(files("descriptron_gui") / "tools"))
    candidates = sorted(tools.glob("descriptron-v2-*.py"), reverse=True)
    if not candidates:
        raise SystemExit(f"no GUI program found in {tools} — "
                         "was sync_sources.py run before building?")
    path = candidates[0]
    sys.path.insert(0, str(tools))
    os.environ.setdefault("DESCRIPTRON_GUI_DATA",
                          str(Path(str(files("descriptron_gui") / "data")))) 
    sys.argv = [str(path), *sys.argv[1:]]
    runpy.run_path(str(path), run_name="__main__")
