"""Florence-2 scale-bar reading in measurement_script_to_try_after_kpts_prediction_measure_kpts_V35 (2.8.6).

From 2.8.0 to 2.8.5 Florence-2 did not load in pip installs or the Docker image: V35 loaded it only through
Microsoft's remote code, which breaks with transformers >= 4.56 (needed for DINOv3). V35 then fell back silently and
found no scale. Nothing tested it, because no test read a real scale bar. This one does: the fixture is the scale bar
of a Kenyan Diaphorina image (bar + its number, cropped without resizing), and the expected px/mm is the published
value for that image (combined_scales.csv).

The test fails (does not skip) when the model cannot be loaded although the weights are reachable. It is skipped only
when torch/transformers are missing or the weights are neither cached nor downloadable (no network).

V35 is found, in this order: $DESCRIPTRON_V35, the copy beside tests/ (repository), the copy installed with
descriptron-vision. Its functions are taken out with ast, as in test_skeleton_length_v1 (V35 imports umap, seaborn,
... at the top)."""
import ast
import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

V35_NAME = "measurement_script_to_try_after_kpts_prediction_measure_kpts_V35.py"
DATA = Path(__file__).resolve().parent / "data"
FIXTURE = DATA / "scalebar_100um_ref.json"
NAMES = ("_load_florence2", "detect_image_scale_florence2", "_bar_in_window", "find_scale_bar_length_in_pixels",
         "parse_scale_text", "calculate_pixels_per_mm")
GLOBALS = ("_FLORENCE2_MODEL", "_FLORENCE2_PROCESSOR", "_SCALE_WINDOWS")
NATIVE_REPO = "florence-community/Florence-2-base"
REMOTE_REPO = "microsoft/Florence-2-base"


def _find_v35():
    env = os.environ.get("DESCRIPTRON_V35")
    if env:
        return Path(env)
    beside = Path(__file__).resolve().parents[1] / V35_NAME
    if beside.exists():
        return beside
    try:
        import descriptron_vision
        installed = Path(descriptron_vision.__file__).resolve().parent / "tools" / V35_NAME
        if installed.exists():
            return installed
    except ImportError:
        pass
    pytest.skip("V35 not found (set DESCRIPTRON_V35)")


def _tf_version():
    return tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])


def _weights_reachable(repo):
    """True if the weights are cached or the Hugging Face Hub answers; False only when neither (no network)."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(repo, timeout=20)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def v35():
    path = _find_v35()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in GLOBALS for t in n.targets))]
    found = {n.name for n in keep if isinstance(n, ast.FunctionDef)}
    assert found == set(NAMES), f"V35 lost {set(NAMES) - found}"
    from PIL import Image
    ns = {"cv2": cv2, "np": np, "re": re, "Image": Image, "logging": logging}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    ns["__v35_path__"] = str(path)
    return ns


@pytest.fixture(scope="module")
def reference():
    ref = json.loads(FIXTURE.read_text(encoding="utf-8"))
    image = cv2.imread(str(DATA / ref["image"]))
    assert image is not None, "fixture image missing"
    return ref, image


def test_florence2_loads(v35, caplog):
    """The model must load wherever the weights can be had; with transformers >= 4.56 through the native class."""
    native = _tf_version() >= (4, 56)
    if not _weights_reachable(NATIVE_REPO if native else REMOTE_REPO):
        pytest.skip("Florence-2 weights neither cached nor downloadable (no network)")
    with caplog.at_level(logging.INFO):
        model, processor = v35["_load_florence2"]()
    assert model is not None and processor is not None, (
        f"Florence-2 did not load with transformers {transformers.__version__} ({v35['__v35_path__']}): "
        + " | ".join(r.getMessage() for r in caplog.records if "Florence" in r.getMessage()))
    print(f"V35: {v35['__v35_path__']}; transformers {transformers.__version__}; {type(model).__module__}.{type(model).__name__}")


def test_reads_real_scale_bar(v35, reference):
    """Florence-2 reads '100 um' and V35 finds the 116-px bar: 1160 px/mm, the published value, within 2%."""
    ref, image = reference
    if not _weights_reachable(NATIVE_REPO if _tf_version() >= (4, 56) else REMOTE_REPO):
        pytest.skip("Florence-2 weights neither cached nor downloadable (no network)")
    model, _ = v35["_load_florence2"]()
    assert model is not None, f"Florence-2 did not load with transformers {transformers.__version__}"
    args = SimpleNamespace(min_width=10, min_aspect_ratio=5.0, max_aspect_ratio=200.0)
    px_per_mm, text, contour, text_bbox = v35["detect_image_scale_florence2"](image, args)
    assert px_per_mm is not None, "no scale bar found in the fixture"
    expected = ref["expected_px_per_mm"]
    assert abs(px_per_mm - expected) / expected <= ref["tolerance_fraction"], \
        f"{px_per_mm:.2f} px/mm ({text}) vs published {expected} px/mm"
    print(f"read '{text}' -> {px_per_mm:.2f} px/mm (published {expected})")
