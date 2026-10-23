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


def _gui_number(p: Path) -> int:
    """'descriptron-v2-v96.py' -> 96; a name without a number sorts first (-1)."""
    tail = p.stem.rsplit("-v", 1)[-1]
    return int(tail) if tail.isdigit() else -1


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
    # 2.8.1: newest by the NUMBER after '-v' (a string sort put v100 before v99); DESCRIPTRON_GUI_VERSION=v94 picks an older one
    candidates = sorted(tools.glob("descriptron-v2-*.py"), key=_gui_number, reverse=True)
    if not candidates:
        raise SystemExit(f"no GUI program found in {tools} — "
                         "was sync_sources.py run before building?")
    path = candidates[0]
    want = os.environ.get("DESCRIPTRON_GUI_VERSION", "").strip()
    if want:
        chosen = [p for p in candidates if _gui_number(p) == _gui_number(Path(f"descriptron-v2-v{want.lstrip('vV')}.py"))]
        if not chosen:
            raise SystemExit(f"DESCRIPTRON_GUI_VERSION={want}: no such GUI. Installed: "
                             + ", ".join(p.stem.split("-")[-1] for p in candidates))
        path = chosen[0]
    if sys.argv[1:2] == ["--which"]:
        from importlib.metadata import version
        print(f"{path}\ndescriptron-gui {version('descriptron-gui')}")
        return
    sys.path.insert(0, str(tools))
    os.environ.setdefault("DESCRIPTRON_GUI_DATA",
                          str(Path(str(files("descriptron_gui") / "data")))) 
    sys.argv = [str(path), *sys.argv[1:]]
    runpy.run_path(str(path), run_name="__main__")
