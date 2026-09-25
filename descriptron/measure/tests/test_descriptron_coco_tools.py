"""Tests for descriptron_coco_tools: specimen-grouped split, annotation check, category rename/merge, folders."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_coco_tools as t  # noqa: E402


def coco(n_spec=5, views=3):
    ims = [{"id": i, "file_name": f"spec{i // views}_view{i % views}.png", "width": 100, "height": 100}
           for i in range(n_spec * views)]
    anns = [{"id": i, "image_id": i, "category_id": 1, "segmentation": [[10, 10, 50, 10, 50, 50]]} for i in range(len(ims))]
    return {"images": ims, "annotations": anns, "categories": [{"id": 1, "name": "head"}]}


def test_split_keeps_specimens_together():
    tr, va, man = t.split_coco(coco(), 0.4, r"^(spec\d+)_", seed=1)
    spec = lambda d: {im["file_name"].split("_")[0] for im in d["images"]}
    assert not spec(tr) & spec(va)
    assert man["val"]["groups"] == 2 and len(va["images"]) == 6
    assert len(tr["annotations"]) + len(va["annotations"]) == 15


def test_split_without_pattern_is_per_image_and_never_empties_train():
    tr, va, man = t.split_coco(coco(1, 2), 0.9)
    assert man["groups"] == 2 and len(tr["images"]) == 1 and len(va["images"]) == 1


def test_check_finds_planted_problems(tmp_path):
    c = coco(2, 1)
    c["categories"] += [{"id": 2, "name": "whole wing "}, {"id": 3, "name": "whole_wing"}]
    c["annotations"].append({"id": 9, "image_id": 0, "category_id": 1, "segmentation": [300, 300, 400, 300, 400, 400]})
    c["annotations"].append({"id": 10, "image_id": 1, "category_id": 1, "keypoints": [1, 1, 2]})
    c["annotations"].append({"id": 11, "image_id": 0, "category_id": 1, "keypoints": [1, 1, 2, 2, 2, 2]})
    (tmp_path / "spec0_view0.jpg").write_bytes(b"x")
    issues = t.check_coco(c, str(tmp_path))
    keys = " ".join(issues)
    for expected in ("spaces/underscores", "leading/trailing", "outside the image", "flat", "different lengths", "missing"):
        assert expected in keys, expected
    assert any("extension differs" in x for x in issues["image file missing"])


def test_rename_merges_categories():
    c = coco(1, 2)
    c["categories"].append({"id": 2, "name": "occipital margin"})
    c["annotations"].append({"id": 5, "image_id": 0, "category_id": 2, "segmentation": [[1, 1, 5, 1, 5, 5]]})
    out = t.rename_categories(c, {"occipital margin": "head"})
    assert [x["name"] for x in out["categories"]] == ["head"]
    assert {a["category_id"] for a in out["annotations"]} == {1}


def test_folders_preview_then_apply(tmp_path, capsys):
    import numpy as np
    from PIL import Image
    (tmp_path / "._a.png").write_bytes(b"x")
    (tmp_path / "a_mask.png").write_bytes(b"x")
    Image.fromarray((np.arange(64, dtype=np.uint16).reshape(8, 8) * 900)).save(tmp_path / "b.tif")
    args = ["folders", str(tmp_path), "--remove-dot-underscore", "--move-masks", "--tiff8"]
    t.main(args)
    assert (tmp_path / "._a.png").exists() and (tmp_path / "a_mask.png").exists()          # preview changes nothing
    t.main(args + ["--apply"])
    assert not (tmp_path / "._a.png").exists()
    assert (tmp_path / "_masks_moved_out" / "a_mask.png").exists()
    png = np.asarray(Image.open(tmp_path / "_png8" / "b.png"))
    assert png.dtype == np.uint8 and png.max() == 255 and (tmp_path / "b.tif").exists()      # original kept
