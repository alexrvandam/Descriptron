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
def predict():    _run("tv_predict_v2.py", sys.argv[1:], "torchvision_det")   # v2.1.2: v2 adds --image_folder (unannotated images)
def sweep():      _run("tv_sweep_v1.py", sys.argv[1:], "torchvision_det")
def sam2_pal():   _run("sam2_pal_batch_v21.py", sys.argv[1:])
def dinoland():   _run("dinov3_landmark_transfer_v52.py", sys.argv[1:])   # v52 = v51 + --orientation_search
def measure():
    _run("measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py", sys.argv[1:])


DINOMASK_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"   # 2.8.1: every published DINOMask result used the ViT-L


def _hf_token_from_store() -> None:
    """2.8.1: HF_TOKEN not set -> take it from Descriptron's key store (~/.config/descriptron/credentials, as DINOLand
    does). Only reads; never prompts. The environment always wins."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from descriptron_core.cli import tools_dir as _core_tools
        sys.path.append(str(_core_tools()))
        import descriptron_credentials
        descriptron_credentials.load_into_environment()
    except Exception:
        pass


def _dinomask_argv() -> list[str]:
    """2.8.1: ViT-L unless the user passes --model (argparse keeps the last value, so theirs wins)."""
    _hf_token_from_store()
    return ["--model", DINOMASK_MODEL, *sys.argv[1:]]


# 2.8.0: DINOMask (frozen DINOv3 + per-patch classifier) and BioRAG-Probe (literature labels x DINOv3 tokens)
def dinomask():       _run("dinomask_train_predict_v4.py", _dinomask_argv(), "dinomask")
def dinomask_rot4():  _run("dinomask_predict_rot4_v1.py", _dinomask_argv(), "dinomask")
def dinomask_v5():    _run("dinomask_train_predict_v5.py", _dinomask_argv(), "dinomask")   # 2.8.1: v5 (token cache, --bg_keep, --grid)
def probe():          _run("biorag_probe_v1.py", sys.argv[1:])


def _available() -> list[str]:
    d = tools_dir()
    if not d.exists():
        return []
    names = [p.stem for p in d.iterdir() if p.suffix == ".py" and not p.name.startswith("_")]
    for sub in ("dinomask", "torchvision_det"):
        if (d / sub).is_dir():
            names += [f"{sub}/{p.stem}" for p in (d / sub).iterdir()
                      if p.suffix == ".py" and not p.name.startswith("_")]
    return sorted(names)


def main() -> None:
    """2.8.0: `descriptron-vision <program> [args]` runs any program of this package (as `descriptron` does for core);
    `descriptron-vision --list` shows them. Programs in a sub-folder are named `dinomask/<file>`."""
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(main.__doc__.strip())
        print(f"\n{len(_available())} programs available; --list shows them.")
        return
    if args[0] in ("--list", "list"):
        print("\n".join(_available()))
        return
    name = args[0][:-3] if args[0].endswith(".py") else args[0]
    matches = [n for n in _available() if n == name] or [n for n in _available() if n.startswith(name)]
    if len(matches) != 1:
        raise SystemExit(f"'{args[0]}' is " + ("ambiguous: " + ", ".join(matches) if matches else
                                                "not a program of descriptron-vision; --list shows them"))
    sub, _, stem = matches[0].rpartition("/")
    _run(stem + ".py", args[1:], sub or None)
