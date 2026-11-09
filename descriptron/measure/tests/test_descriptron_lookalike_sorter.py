"""descriptron_lookalike_sorter_v1: the drag board from a COCO file with any category (here 'petal'), and the
groups read back from saved positions (single linkage, gap report, reading-order numbering, named merges)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_lookalike_sorter_v1 as ls          # noqa: E402


def test_build_board_any_category_with_tree_and_thumbnails(tmp_path):
    from PIL import Image
    d = tmp_path / "img"; d.mkdir()
    imgs, anns = [], []
    for i, sp in enumerate(["Rosa canina", "Rosa canina", "Malva sylvestris", "Viola odorata"]):
        a = np.full((80, 60, 3), 255, np.uint8); a[10:50, 15:45] = (200, 40 + 40 * i, 90)
        Image.fromarray(a).save(d / f"f{i}.png")
        imgs.append({"id": i + 1, "file_name": f"f{i}.png", "width": 60, "height": 80, "species": sp, "view": "front"})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "segmentation": [[15, 10, 45, 10, 45, 50, 15, 50]]})
    (tmp_path / "c.json").write_text(json.dumps({"images": imgs, "annotations": anns,
                                                 "categories": [{"id": 1, "name": "petal"}]}))
    (tmp_path / "t.nwk").write_text("((Rosa_canina:1,Malva_sylvestris:1):1,Geranium_sp:2);")
    out = tmp_path / "board.html"
    ls.main(["build", "--coco", str(tmp_path / "c.json"), "--image_dir", str(d), "--category", "petal",
             "--mask_background", "--image_filter", "view=front", "--tree", str(tmp_path / "t.nwk"),
             "--title", "Flower look-alikes", "--save_name", "flowers_positions.json", "--thumb_height", "40",
             "--items_out", str(tmp_path / "items.json"), "--thumbs_dir", str(tmp_path / "th"), "--out", str(out)])
    items = json.load(open(tmp_path / "items.json"))
    assert [i["sp"] for i in items] == ["Rosa_canina", "Malva_sylvestris", "Viola_odorata"]   # one per species
    assert [i["tree"] for i in items] == [True, True, False] and items[0]["name"] == "R. canina"
    html = out.read_text()
    assert "Flower look-alikes" in html and "flowers_positions.json" in html and "data:image/jpeg;base64," in html
    rows = list(csv.DictReader(open(tmp_path / "th" / "tip_images.csv")))
    th = Image.open(rows[0]["image"])
    assert th.height == 40 and abs(th.width - 40 * 32 / 42) <= 1          # cropped to the petal (+3% margin)


def test_read_groups_single_linkage_gap_report_numbering_and_merges(tmp_path, capsys):
    rng = np.random.default_rng(2)
    piles = {(100, 100): 4, (600, 120): 3, (150, 700): 5, (900, 650): 2}     # reading order: top row left to right
    P = []
    for (cx, cy), k in piles.items():
        for j in range(k):
            P.append({"species": f"s{cx}_{j}", "on_tree": j == 0, "x": int(cx + rng.integers(-20, 20)),
                      "y": int(cy + rng.integers(-20, 20))})
    rng.shuffle(P)
    (tmp_path / "pos.json").write_text(json.dumps({"species": P}))
    ls.main(["read", "--positions", str(tmp_path / "pos.json"), "--row_height", "400", "--merge", "big=G1,G3",
             "--merge_column", "kind", "--note", "G4=a pair", "--out", str(tmp_path / "g.csv")])
    rows = list(csv.DictReader(open(tmp_path / "g.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["group"], set()).add(r["species"].split("_")[0])
    assert by == {"G1": {"s100"}, "G2": {"s600"}, "G3": {"s150"}, "G4": {"s900"}}
    assert {r["kind"] for r in rows if r["group"] in ("G1", "G3")} == {"big"} and all(
        r["note"] == "a pair" for r in rows if r["group"] == "G4")
    assert list(rows[0]) == ["group", "kind", "species", "on_tree", "x", "y", "note"]
    out = capsys.readouterr().out
    assert "4 groups" in out and "gap report" in out and "G2, G4" in out           # unnamed groups listed
    R = ls.single_linkage(np.array([[p["x"], p["y"]] for p in P], float), None)
    assert R["within_max"] < R["cut"] < R["between_min"]
