"""Tests for descriptron_convert: tpsDig, MorphoJ, StereoMorph, landmark tables and VIA 2 -> Descriptron COCO."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_convert as dc  # noqa: E402

W, H = 200, 100


@pytest.fixture
def imgs(tmp_path):
    from PIL import Image
    d = tmp_path / "img"; d.mkdir()
    for n in ("a.tif", "b.jpg"):
        Image.new("RGB", (W, H)).save(d / n)
    return d


def run(*argv):
    dc.main([str(a) for a in argv])


def kps(path):
    c = json.load(open(path))
    ims = {i["id"]: i for i in c["images"]}
    return c, {ims[a["image_id"]]["file_name"]: a for a in c["annotations"] if "keypoints" in a}


def test_tps_bottom_origin_is_flipped_and_scale_is_mm_per_pixel(tmp_path, imgs):
    t = tmp_path / "x.tps"
    t.write_text("LM=3\n10 90\n20 80\n-1 -1\nIMAGE=C:\\photos\\a.tif\nID=7\nSCALE=0.01\n\n")
    run("tps2coco", t, "--images", imgs, "--out", tmp_path / "o.json")
    c, k = kps(tmp_path / "o.json")
    a = k["a.tif"]
    assert a["keypoints"] == [10, 10, 2, 20, 20, 2, 0, 0, 0]          # y flipped with H=100; -1 -1 missing
    assert c["images"][0]["scale_px_per_mm"] == 100.0                  # 0.01 mm/pixel -> 100 px/mm
    assert a["point_order"] == [1, 2, 3] and a["num_keypoints"] == 2


def test_tps_headers_before_coordinates_and_px_per_mm_scale(tmp_path, imgs):
    t = tmp_path / "old.tps"                                           # older Descriptron layout
    t.write_text("LM=2\nIMAGE=a.tif\nSCALE=530\n5 6\n7 8\n\n")
    run("tps2coco", t, "--images", imgs, "--y-origin", "top", "--out", tmp_path / "o.json")
    c, k = kps(tmp_path / "o.json")
    assert k["a.tif"]["keypoints"] == [5, 6, 2, 7, 8, 2]
    assert c["images"][0]["scale_px_per_mm"] == 530.0


def test_tps_curves_become_semilandmark_sets(tmp_path, imgs):
    t = tmp_path / "c.tps"
    t.write_text("LM=1\n1 1\nCURVES=1\nPOINTS=3\n1 2\n3 4\n5 6\nIMAGE=a.tif\n")
    run("tps2coco", t, "--images", imgs, "--y-origin", "top", "--out", tmp_path / "o.json")
    c = json.load(open(tmp_path / "o.json"))
    semi = [a for a in c["annotations"] if a.get("semilandmarks")]
    assert len(semi) == 1 and semi[0]["keypoints"][::3] == [1, 3, 5]


def test_tps_round_trip_is_exact(tmp_path, imgs):
    t = tmp_path / "x.tps"
    t.write_text("LM=2\n10.5 90.25\n-1 -1\nIMAGE=a.tif\nSCALE=0.02\n\nLM=2\n1 2\n3 4\nIMAGE=b.jpg\n")
    run("tps2coco", t, "--images", imgs, "--out", tmp_path / "o.json")
    run("coco2tps", tmp_path / "o.json", "--images", imgs, "--out", tmp_path / "back.tps")
    back = dc.parse_tps(str(tmp_path / "back.tps"))
    assert [s["lm"] for s in back] == [[(10.5, 90.25), (-1, -1)], [(1, 2), (3, 4)]]
    assert back[0]["scale"] == pytest.approx(0.02)


def test_morphoj_with_header_and_na(tmp_path, imgs):
    m = tmp_path / "m.txt"
    m.write_text("ID\tx1\ty1\tx2\ty2\na\t10\t90\tNA\tNA\n")
    run("morphoj2coco", m, "--images", imgs, "--out", tmp_path / "o.json")
    _, k = kps(tmp_path / "o.json")
    assert k["a.tif"]["keypoints"] == [10, 10, 2, 0, 0, 0]


def test_morphoj_round_trip(tmp_path, imgs):
    m = tmp_path / "m.txt"
    m.write_text("a,10,90,30,40\n")                                    # no header, comma separated
    run("morphoj2coco", m, "--images", imgs, "--out", tmp_path / "o.json")
    run("coco2morphoj", tmp_path / "o.json", "--images", imgs, "--out", tmp_path / "b.txt")
    assert (tmp_path / "b.txt").read_text().splitlines()[1].split("\t") == ["a", "10.00", "90.00", "30.00", "40.00"]


def test_table_orders_numeric_landmarks(tmp_path, imgs):
    t = tmp_path / "t.csv"
    t.write_text("image,landmark,x,y\na.tif,10,1,2\na.tif,2,3,4\n")
    run("table2coco", t, "--images", imgs, "--out", tmp_path / "o.json")
    c, k = kps(tmp_path / "o.json")
    assert c["categories"][0]["keypoints"] == ["2", "10"] and k["a.tif"]["keypoints"] == [3, 4, 2, 1, 2, 2]


def test_stereomorph_landmarks_and_curve(tmp_path, imgs):
    d = tmp_path / "shapes"; d.mkdir()
    (d / "a.txt").write_text("<image.filename>a.tif</image.filename>\n<landmarks.pixel>\nLM1\t5\t6\nLM2\tNA\tNA\n"
                             "</landmarks.pixel>\n<curves.pixel>\nedge\t1\t1\nedge\t9\t9\n</curves.pixel>\n")
    run("stereomorph2coco", d, "--images", imgs, "--out", tmp_path / "o.json")
    c, k = kps(tmp_path / "o.json")
    assert k["a.tif"]["keypoints"] == [5, 6, 2, 0, 0, 0]
    line = [a for a in c["annotations"] if a.get("is_line")][0]
    assert line["line_points"] == [[1, 1], [9, 9]] and line["segmentation"]


def test_via_project_all_shapes(tmp_path, imgs):
    reg = lambda sh, name: {"shape_attributes": sh, "region_attributes": {"name": name}}
    proj = {"_via_img_metadata": {"b.jpg123": {"filename": "b.JPG", "regions": [
        reg({"name": "point", "cx": 5, "cy": 6}, "2"), reg({"name": "point", "cx": 1, "cy": 2}, "1"),
        reg({"name": "polygon", "all_points_x": [0, 10, 10], "all_points_y": [0, 0, 10]}, "whole wing "),
        reg({"name": "rect", "x": 1, "y": 1, "width": 4, "height": 3}, "box"),
        reg({"name": "circle", "cx": 50, "cy": 50, "r": 5}, "eye"),
        reg({"name": "polyline", "all_points_x": [0, 20, 40], "all_points_y": [5, 5, 9]}, "seta")]}}}
    f = tmp_path / "via.json"; f.write_text(json.dumps(proj))
    run("via2coco", f, "--images", imgs, "--out", tmp_path / "o.json")
    c, k = kps(tmp_path / "o.json")
    assert k["b.jpg"]["keypoints"] == [1, 2, 2, 5, 6, 2]                 # ordered by landmark number
    names = [x["name"] for x in c["categories"]]
    assert "whole_wing" in names and {"box", "eye", "seta"} <= set(names)
    assert c["images"][0]["width"] == W                                  # matched case-insensitively
    assert any(a.get("is_line") for a in c["annotations"])


def test_via_coco_export_quirks(tmp_path, imgs):
    exp = {"images": [{"id": 0, "file_name": "a.jpg", "width": 0, "height": 0}],
           "categories": [{"id": 1, "name": "genal processes ", "supercategory": "Head "}],
           "annotations": [{"id": 1, "image_id": "0", "category_id": 1, "segmentation": [0, 0, 10, 0, 10, 10]},
                           {"id": 2, "image_id": "0", "segmentation": [0, 0, 5, 0, 5, 5]}]}
    f = tmp_path / "e.json"; f.write_text(json.dumps(exp))
    run("via2coco", f, "--images", imgs, "--out", tmp_path / "o.json")
    c = json.load(open(tmp_path / "o.json"))
    assert c["images"][0] == {"id": 1, "file_name": "a.tif", "width": W, "height": H}
    assert [x["name"] for x in c["categories"]] == ["genal_processes", "unlabeled"]
    assert all(isinstance(a["segmentation"][0], list) and a["bbox"] for a in c["annotations"])


def test_per_image_files_for_dinoland(tmp_path, imgs):
    t = tmp_path / "x.tps"
    t.write_text("LM=1\n1 1\nIMAGE=a.tif\n\nLM=1\n2 2\nIMAGE=b.jpg\n")
    run("tps2coco", t, "--images", imgs, "--out", tmp_path / "o.json", "--per-image", tmp_path / "refs")
    one = json.load(open(tmp_path / "refs" / "a_keypoints.json"))
    assert len(one["images"]) == 1 and one["annotations"][0]["keypoints"] == [1, 99, 2]
