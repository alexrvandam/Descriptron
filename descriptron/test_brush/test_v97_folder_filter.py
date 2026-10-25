"""
Headless tests for the v97 Load Folder mask filter (_v97_folder_images, 25 Oct 2026).

Load Folder (v82-v96) skipped only names ending in _mask.png/.jpg/.tif, so a folder of photographs with thousands
of mask cutouts written by other tools loaded them all as specimens. v97 skips (1) files with 'mask' as a word in
the name and (2) one-channel PNGs named <another listed image's stem>_...

The function is taken VERBATIM from the GUI source (parsed with ast, no copy-paste) and run on real files in a
temporary folder.

Run:  <python of the GUI env> test_brush/test_v97_folder_filter.py
"""
import ast, os, sys, tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.dirname(HERE)
V97 = os.path.join(GUI_DIR, "descriptron-v2-v97.py")


def _load():
    src = open(V97).read()
    tree = ast.parse(src)
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and any(getattr(t, "id", None) == "IMAGE_EXTS" for t in n.targets)]
    code = "\n\n".join([ast.get_source_segment(src, consts[0])] +
                       [ast.get_source_segment(src, defs[n]) for n in ("_is_descriptron_output", "_v97_folder_images")])
    ns = {"os": os, "Image": Image}
    exec(code, ns)
    return ns["_v97_folder_images"]


folder_images = _load()


def _rgb(path):
    Image.fromarray(np.full((8, 8, 3), 120, np.uint8)).save(path)


def _grey(path):
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(path)


def _binary(path):
    Image.fromarray(np.zeros((8, 8), bool)).save(path)


T = []
def test(f): T.append(f); return f


def _run(files):
    """files: {name: writer}; returns (kept, skipped)."""
    with tempfile.TemporaryDirectory() as d:
        for name, w in files.items():
            w(os.path.join(d, name))
        return folder_images(d, os.listdir(d))


@test
def rgb_photos_kept():
    kept, skipped = _run({"sp1.jpg": _rgb, "sp2.png": _rgb, "sp3.tif": _rgb})
    assert kept == ["sp1.jpg", "sp2.png", "sp3.tif"] and skipped == [], (kept, skipped)


@test
def mask_words_skipped():
    names = ["sp1_eye_0_mask.png", "sp1_eye_0_mask_cutout.png", "sp1_mask_image.png", "mask0.png",
             "warped_mask_9_x.png"]
    kept, skipped = _run(dict({"sp1.jpg": _rgb}, **{n: _rgb for n in names}))
    assert kept == ["sp1.jpg"], kept
    assert sorted(skipped) == sorted(names), skipped


@test
def binary_png_named_after_an_image_skipped():
    kept, skipped = _run({"Tri-1.tif": _rgb, "Tri-1_leg_0.png": _binary})
    assert kept == ["Tri-1.tif"] and skipped == ["Tri-1_leg_0.png"], (kept, skipped)


@test
def rgb_png_named_after_an_image_kept():
    kept, skipped = _run({"Tri-1.tif": _rgb, "Tri-1_detail.png": _rgb})
    assert kept == ["Tri-1.tif", "Tri-1_detail.png"] and skipped == [], (kept, skipped)


@test
def grey_png_not_named_after_another_image_kept():
    kept, skipped = _run({"ct_slice_0001.png": _grey, "ct_slice_0002.png": _grey, "sp1.jpg": _rgb})
    assert kept == ["ct_slice_0001.png", "ct_slice_0002.png", "sp1.jpg"] and skipped == [], (kept, skipped)


@test
def no_prefix_confusion_between_tri1_and_tri10():
    kept, skipped = _run({"Tri-1.tif": _rgb, "Tri-10.jpg": _rgb, "Tri-10.png": _grey})
    assert kept == ["Tri-1.tif", "Tri-10.jpg", "Tri-10.png"] and skipped == [], (kept, skipped)


@test
def words_containing_mask_kept():
    kept, skipped = _run({"damask_moth.jpg": _rgb, "masked_bee.jpg": _rgb})
    assert kept == ["damask_moth.jpg", "masked_bee.jpg"] and skipped == [], (kept, skipped)


@test
def hidden_and_non_image_files_ignored():
    kept, skipped = _run({"._sp1.jpg": _rgb, "sp1.jpg": _rgb, "notes.txt": lambda p: open(p, "w").write("x")})
    assert kept == ["sp1.jpg"] and skipped == [], (kept, skipped)


if __name__ == '__main__':
    bad = 0
    for f in T:
        try:
            f(); print('ok  ', f.__name__)
        except Exception as e:
            bad += 1; print('FAIL', f.__name__, repr(e))
    print(f'{len(T) - bad}/{len(T)} passed'); sys.exit(1 if bad else 0)
