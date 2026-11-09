"""descriptron_mask_tools_v1: combine / trim_end / tile on synthetic, non-beetle inputs (a flower: petal, sepal).
The Platydracus reproductions (byte-identical soma_v5.json, soma_v5b.json, parts_v2.json) need the study's data and
are run outside the test suite."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mu = pytest.importorskip("pycocotools.mask")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_mask_tools_v1 as mt          # noqa: E402


def _rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return r


def _flower(h=120, w=100):
    """a 'flower': a petal disc (top) and a sepal block (bottom) with a 2 px pale gap between them in the specimen
    mask; the petal prediction has a hole and a stray speck."""
    yy, xx = np.mgrid[:h, :w]
    petal = (yy - 40) ** 2 + (xx - 50) ** 2 < 30 ** 2
    sepal = (yy >= 72) & (yy < 110) & (xx >= 30) & (xx < 70)
    fg = petal | sepal
    fg[69:72] = False                                          # the pale gap
    pred_petal = petal.copy(); pred_petal[38:42, 48:52] = False; pred_petal[2:5, 2:5] = True
    return fg, petal, sepal, pred_petal


def _write_inputs(tmp_path):
    fg, petal, sepal, pp = _flower()
    d = tmp_path / "img"; d.mkdir()
    cv2.imwrite(str(d / "f1_mask.png"), fg.astype(np.uint8) * 255)
    img = np.full(fg.shape + (3,), 255, np.uint8); img[petal] = (60, 60, 200); img[sepal] = (40, 140, 40)
    cv2.imwrite(str(d / "f1.png"), img)
    pred = {"images": [{"id": 1, "file_name": "f1.png", "height": fg.shape[0], "width": fg.shape[1]}],
            "categories": [{"id": 1, "name": "petal"}, {"id": 2, "name": "sepal"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": _rle(pp)},
                            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": _rle(sepal)}]}
    (tmp_path / "pred.json").write_text(json.dumps(pred))
    (tmp_path / "table.csv").write_text("file,taxon,side\nf1.png,Rosa canina,front\n")
    return fg, petal, sepal


PROFILE = """
mask_tools:
  combine:
    categories: [corolla, flower, calyx]
    image_table: {file: file, attrs: [{name: species, column: taxon}, {name: view, column: side}]}
    image_filter: {side: [front]}
    outputs:
      - {name: corolla, ops: [{union: [pred:petal]}, {foreground: 0}, {open: 2}, {largest: true}]}
      - {name: flower, ops: [{union: [out:corolla, pred:sepal]}, {foreground: 3}, {largest: true}]}
      - {name: calyx, ops: [{union: [out:flower]}, {subtract: [out:corolla]}, {open: 2}, {largest: true}]}
  tile: {container: flower, parts: [petal, sepal], order_check: [petal, sepal], axis: y}
"""


def _load(path):
    D = json.load(open(path))
    cat = {c["id"]: c["name"] for c in D["categories"]}
    im = D["images"][0]
    return D, {cat[a["category_id"]]: mt.decode(a["segmentation"], im["height"], im["width"]) for a in D["annotations"]}


def test_combine_by_profile_recipe_flower(tmp_path):
    fg, petal, sepal = _write_inputs(tmp_path)
    (tmp_path / "p.yaml").write_text(PROFILE)
    mt.main(["combine", "--profile", str(tmp_path / "p.yaml"), "--predictions", str(tmp_path / "pred.json"),
             "--image_dir", str(tmp_path / "img"), "--image_table", str(tmp_path / "table.csv"), "--out", str(tmp_path / "o.json")])
    D, M = _load(tmp_path / "o.json")
    assert [c["name"] for c in D["categories"]] == ["corolla", "flower", "calyx"]
    assert D["images"][0]["species"] == "Rosa canina" and D["images"][0]["view"] == "front"
    iou = lambda a, b: (a & b).sum() / (a | b).sum()
    assert iou(M["corolla"], petal) > 0.97                      # hole filled, speck removed (largest piece)
    assert not M["corolla"][2:5, 2:5].any() and M["corolla"][40, 50]
    assert iou(M["flower"], petal | sepal) > 0.95 and M["flower"][70, 50]   # the 3-row gap bridged
    assert (M["calyx"] & M["corolla"]).sum() == 0 and iou(M["calyx"], sepal) > 0.85


def test_combine_one_output_on_the_command_line(tmp_path):
    fg, petal, sepal = _write_inputs(tmp_path)
    mt.main(["combine", "--predictions", str(tmp_path / "pred.json"), "--image_dir", str(tmp_path / "img"),
             "--output", "petals", "--from", "petal", "--bridge_px", "0", "--open_px", "2", "--largest",
             "--out", str(tmp_path / "o.json")])
    D, M = _load(tmp_path / "o.json")
    assert list(M) == ["petals"] and (M["petals"] & petal).sum() / petal.sum() > 0.97


def test_trim_end_narrow_lobe_and_pale_band_any_axis():
    h, w = 100, 60
    m = np.zeros((h, w), bool); m[20:95, 10:50] = True; m[5:20, 25:35] = True        # a narrow lobe at the top
    L = np.full((h, w), 40.0); L[28:31] = 90.0                                        # a pale band at rows 28-30
    R = mt.trim_end_mask(m, L, 0.8, 0.5, 0.3, 15.0, 0.6)                              # band within the first 30%
    assert R["flat_row"] == 20 and R["membrane_row"] == 31 and R["new_top"] == 31
    assert R["mask"][:31].sum() == 0 and R["mask"][31:95, 10:50].all()
    R0 = mt.trim_end_mask(m, None, 0.8, 0.5, 0.15, 15.0, 0.6)                          # no image: lobe only
    assert R0["new_top"] == 20 and R0["mask"][20:95, 10:50].all()
    # the same mask lying on its side with the lobe at the right end: axis x, end
    Rx = mt.trim_end_mask(m.T[:, ::-1], None, 0.8, 0.5, 0.15, 15.0, 0.6, axis="x", end="end")
    np.testing.assert_array_equal(Rx["mask"], R0["mask"].T[:, ::-1])


def test_tile_splits_container_into_named_parts_with_rules(tmp_path):
    fg, petal, sepal = _write_inputs(tmp_path)
    (tmp_path / "p.yaml").write_text(PROFILE)
    mt.main(["combine", "--profile", str(tmp_path / "p.yaml"), "--predictions", str(tmp_path / "pred.json"),
             "--image_dir", str(tmp_path / "img"), "--image_table", str(tmp_path / "table.csv"), "--out", str(tmp_path / "c.json")])
    mt.main(["tile", "--profile", str(tmp_path / "p.yaml"), "--predictions", str(tmp_path / "pred.json"),
             "--coco", str(tmp_path / "c.json"), "--extra", "outline=flower", "--out", str(tmp_path / "t.json"),
             "--report", str(tmp_path / "r.csv")])
    D, M = _load(tmp_path / "t.json")
    _, C = _load(tmp_path / "c.json")
    assert [c["name"] for c in D["categories"]] == ["petal", "sepal", "outline"]
    assert ((M["petal"] | M["sepal"]) == C["flower"]).all() and not (M["petal"] & M["sepal"]).any()   # an exact tiling
    assert "ok" in (tmp_path / "r.csv").read_text()
    # rules: a narrow 'neck' of B in front of its widest line goes to A; A's wedge in front of its shoulder to B
    cont = np.zeros((60, 40), bool); cont[5:55, 5:35] = True
    lab0 = np.zeros((60, 40), np.int8); lab0[5:20] = 1; lab0[20:24, 15:25] = 2; lab0[24:40] = 2; lab0[40:55] = 3
    lab0[20:24, 5:15] = 1; lab0[20:24, 25:35] = 1
    lab = mt.tile_container(cont, lab0, ["a", "b", "c"], ["a:b:0.6"], [], "y")
    assert (lab[20:24, 15:25] == 1).all() and (lab[24:40, 5:35] == 2).all()
    lab0 = np.zeros((60, 40), np.int8); lab0[5:30] = 1; lab0[30:55] = 2; lab0[22:30, 18:22] = 2     # a wedge of c
    lab = mt.tile_container(cont, lab0, ["b", "c"], [], ["c:b:0.12:0.6"], "y")
    # c spans rows 22-54 (33 rows), its shoulder is row 30: it may reach int(0.12 x 33) = 3 rows in front of it
    assert (lab[22:27, 18:22] == 1).all() and (lab[27:30, 18:22] == 2).all()
