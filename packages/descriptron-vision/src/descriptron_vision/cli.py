"""
descriptron_vision.cli — detectors, propagation and landmark transfer
=====================================================================

Same dispatch as `descriptron_core.cli`. The two dependencies that are not on
PyPI — SAM2 and Detectron2 — are not declared anywhere: PyPI rejects a git
dependency in package metadata. They are imported lazily by the programs that
need them, and `require()` below turns the resulting ImportError into the exact
command to run.
"""
from __future__ import annotations

import sys

from descriptron_core.cli import run as _run_core   # noqa: F401  (shared behaviour)
from descriptron_core.cli import data_path          # noqa: F401

import os
import runpy
from importlib.resources import files
from pathlib import Path

INSTALL_HINTS = {
    "sam2": ("SAM2 is not on PyPI, so it cannot be a declared dependency.\n"
             "  pip install git+https://github.com/facebookresearch/segment-anything-2"),
    "detectron2": ("Detectron2 is not on PyPI and must be compiled:\n"
                   "  pip install git+https://github.com/facebookresearch/detectron2\n"
                   "It needs a C++/CUDA toolchain and has no macOS wheels. If you would\n"
                   "rather not build it, the Docker image carries it ready to run, and the\n"
                   "torchvision backend needs none of this."),
}


def require(module: str):
    """Import an optional heavyweight dependency, or explain how to get it."""
    try:
        return __import__(module)
    except ImportError as exc:
        hint = INSTALL_HINTS.get(module, f"  pip install {module}")
        raise SystemExit(f"{module} is required for this step but is not installed.\n"
                         f"{hint}\n\n(original error: {exc})") from exc


def tools_dir() -> Path:
    return Path(str(files("descriptron_vision") / "tools"))


def _run(script: str, argv: list[str], subdir: str | None = None) -> None:
    base = tools_dir() / subdir if subdir else tools_dir()
    path = base / script
    if not path.exists():
        raise SystemExit(f"{path} is missing — was sync_sources.py run before building?")
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(tools_dir()))
    os.environ.setdefault("DESCRIPTRON_DATA", str(data_path()))
    sys.argv = [str(path), *argv]
    runpy.run_path(str(path), run_name="__main__")


def train():      _run("tv_train_v1.py", sys.argv[1:], "torchvision_det")
def predict():    _run("tv_predict_v1.py", sys.argv[1:], "torchvision_det")
def sweep():      _run("tv_sweep_v1.py", sys.argv[1:], "torchvision_det")
def sam2_pal():   _run("sam2_pal_batch_v21.py", sys.argv[1:])
def dinoland():   _run("dinov3_landmark_transfer_v51.py", sys.argv[1:])
def measure():
    _run("measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py", sys.argv[1:])
