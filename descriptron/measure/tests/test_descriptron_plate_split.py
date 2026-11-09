"""descriptron_plate_split_v1: captions of a real revision parse letter by letter (including its typographic slips
and a caption run on into the next figure's); scale bars measure to the pixel; on synthetic plates the specimens,
letters (read on the plate, in column or row layouts), detached appendages, touching specimens, own and shared
px/mm and the warnings come out right; files that are not plates are skipped, not cut."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import descriptron_plate_split_v1 as ps        # noqa: E402

CAP = ("A and B) Platydracus martini; C and D) P. basicornis; E and F) P. curticornis. A, C, and E) Dorsal view; "
       "B, D, and F, ventral view. Scale bar 2 mm.")


# ------------------------------------------------------------------ captions (real variants from one revision)
def test_caption_standard_and_slips():
    c = ps.parse_caption(CAP)
    assert c["letters"] == list("ABCDEF") and not c["warnings"]
    assert c["species"]["D"] == "Platydracus basicornis" and c["views"]["F"] == "ventral"
    assert [(b["cls"], b["mm"], b["letters"]) for b in c["bars"]] == [(None, 2.0, None)]
    # "A and B), Platydracus" and "C and D, P. gladius" (comma for the bracket)
    c = ps.parse_caption("A and B), Platydracus tigris; C and D, P. gladius; E and F) P. acutatus. A, C, and E) "
                         "Dorsal view; B, D, F, ventral view. Scale bar 2 mm.")
    assert c["species"]["B"] == "Platydracus tigris" and c["species"]["C"] == "Platydracus gladius"
    assert c["views"]["D"] == "ventral" and not c["warnings"]


def test_caption_ranges_combined_plate_and_missing_species():
    c = ps.parse_caption("A to D) Platydracus purpuronotus; E and F) P. brachelytrus. A and E) Dorsal view; B and "
                         "F) ventral view; C and G) paramere view; D and H) lateral view. Habitus scale bar 2 mm, "
                         "aedeagus scale bar 0.")
    assert c["letters"] == list("ABCDEFGH")
    assert c["species"]["C"] == "Platydracus purpuronotus" and c["species"]["G"] is None
    assert c["views"]["G"] == "paramere" and c["views"]["D"] == "lateral"
    assert any("no species named for G, H" in w for w in c["warnings"])
    assert {(b["cls"], b["mm"]) for b in c["bars"]} == {("habitus", 2.0), ("genitalia", None)}
    assert any("not readable" in w for w in c["warnings"])


def test_caption_cut_at_next_figure_and_conflicting_views():
    c = ps.parse_caption("A to D) Platydracus felis. A) Dorsal view; B) ventral view; C) paramere view; D) lateral "
                         "view. Habitus scale bar 2 mm, aedeagus scale bar 0.5 mm. Fig. 32. Distribution maps of P. "
                         "ruficollis, P. cooteri in China.")
    assert c["letters"] == list("ABCD") and "Distribution" not in c["text"]
    assert any("Fig. 32." in w for w in c["warnings"])
    assert {(b["cls"], b["mm"]) for b in c["bars"]} == {("habitus", 2.0), ("genitalia", 0.5)}
    c = ps.parse_caption("A to D) Platydracus mortuorum; E and F) P. decipiens. A, C, and E) Dorsal view; A, C, and "
                         "E) paramere view; B, D, and F) lateral view. Scale bar 0.")
    assert c["views"]["A"] is None and c["views"]["B"] == "lateral"
    assert any("A is given 2 views" in w for w in c["warnings"])
    assert c["bars"][0]["mm"] is None


def test_caption_letter_scoped_bars_units_and_genus():
    c = ps.parse_caption("A-C) T. alpha; D) T. beta. Dorsal views. Scale bars: 1 mm (A-C), 500 µm (D).",
                         genus="Tetramorium")
    assert c["species"]["A"] == "Tetramorium alpha" and c["views"]["D"] == "dorsal"
    assert ps.bar_mm_for("B", "habitus", c["bars"], {})[0] == 1.0
    assert ps.bar_mm_for("D", "habitus", c["bars"], {})[0] == 0.5
    c = ps.parse_caption("A) P. alpha; B) P. beta. Dorsal view.")
    assert any("not spelled out" in w for w in c["warnings"])
    assert ps.parse_caption("Distribution map of the species.")["letters"] == []


# ------------------------------------------------------------------ scale bars
def test_bar_length_counts_soft_ends():
    g = np.full((200, 60), 255.0, np.float32)
    g[50:130, 20:26] = 0                      # 80 rows of ink
    g[49, 20:26] = 127.5; g[130, 20:26] = 127.5   # half-dark end rows -> +0.5 each
    sat = np.zeros_like(g); fg = g < 235; dark = g < 140
    bars = ps.find_bars(g, sat, fg, dark, longer=1000, orient="both")     # as on a 1000-px plate
    assert len(bars) == 1 and bars[0]["orient"] == "vertical"
    assert abs(bars[0]["px"] - 81.0) < 0.05


# ------------------------------------------------------------------ synthetic plates
def _font(size):
    for n in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default(size)


def make_plate(path, layout="column", letters=True, touching=False, bars=(60, 70, 80)):
    """3 x 2 brown 'beetles' with antennae and legs; letters bottom-right of each cell; black bars beside the
    letters of the dorsal panels (A, C, E in column layout). One detached antenna tip (red) on specimen A."""
    W, H = 960, 1160
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = _font(44)
    order = {"column": "ACEBDF", "row": "ABCDEF"}[layout]       # top row then bottom row, left to right
    cells = {}
    for k, L in enumerate(order):
        col, row = k % 3, k // 3
        cells[L] = (160 + 320 * col, 300 + 560 * row)
    for i, L in enumerate("ABCDEF"):
        cx, cy = cells[L]
        body = (110 + 10 * i, 70, 40)
        d.ellipse([cx - 55, cy - 180, cx + 55, cy + 180], fill=body)
        d.ellipse([cx - 35, cy - 240, cx + 35, cy - 170], fill=body)              # head
        for s in (-1, 1):
            d.line([cx + 20 * s, cy - 230, cx + 95 * s, cy - 265], fill=body, width=4)   # antennae
            for yy in (-90, 0, 90):
                d.line([cx + 50 * s, cy + yy, cx + 110 * s, cy + yy + 60], fill=body, width=5)  # legs
        if L == "A":
            d.ellipse([cx + 98, cy - 274, cx + 108, cy - 262], fill=(200, 40, 40))  # detached antenna tip
    if touching:                                   # C's right antenna runs on into E's left antenna
        (cx, cy), (ex, ey) = cells["C"], cells["E"]
        d.line([cx + 95, cy - 265, ex - 95, ey - 265], fill=(130, 70, 40), width=4)
    dorsal = {"column": "ACE", "row": "ACE"}[layout]
    for L in "ABCDEF":
        cx, cy = cells[L]
        lx, ly = cx + 100, cy + 200
        if letters:
            d.text((lx, ly), L, fill=(0, 0, 0), font=f)
        if L in dorsal:
            bl = bars["ACE".index(L)]
            bot = ly + 44
            d.rectangle([lx - 22, bot - bl, lx - 17, bot - 1], fill=(0, 0, 0))   # 6 px wide, bl px tall
    im.save(path)
    return cells


def run(tmp_path, plate, caption=CAP, extra=()):
    out = tmp_path / "out"
    ps.main(["--plates", str(plate), "--caption", caption, "--out_dir", str(out), *extra])
    rows = list(csv.DictReader(open(out / "plate_split_panels.csv")))
    prow = list(csv.DictReader(open(out / "plate_split_plates.csv")))
    return out, rows, prow


def _at(rows, cells):
    """letter whose specimen bbox holds each cell centre"""
    got = {}
    for L, (cx, cy) in cells.items():
        for r in rows:
            x, y, w, h = (int(r[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            if x <= cx < x + w and y <= cy < y + h:
                got[L] = r["letter"]
    return got


def test_column_plate_letters_scale_crops(tmp_path):
    cells = make_plate(tmp_path / "fig001_p001.png")
    out, rows, prow = run(tmp_path, tmp_path / "fig001_p001.png", extra=["--mask"])
    assert len(rows) == 6 and prow[0]["status"] == "ok" and prow[0]["layout"] == "column"
    assert _at(rows, cells) == {L: L for L in "ABCDEF"}
    assert all(r["letter_method"] == "read" for r in rows)
    by = {r["letter"]: r for r in rows}
    for L, ppm in zip("ACE", (30, 35, 40)):            # 60, 70, 80 px bars of 2 mm
        assert by[L]["px_per_mm_source"] == "own bar" and abs(float(by[L]["px_per_mm"]) - ppm) < 0.05
    for L, D in zip("BDF", "ACE"):
        assert by[L]["px_per_mm_source"].startswith(f"shared from {D}")
        assert by[L]["px_per_mm"] == by[D]["px_per_mm"] and by[L]["scale_bar_px"] == ""
    assert by["C"]["crop_file"] == "fig001_p001_C_Platydracus_basicornis_dorsal.png"
    for r in rows:
        crop = np.asarray(Image.open(out / r["crop_file"]).convert("RGB")).astype(int)
        assert crop.min() >= 25, "a letter or scale bar (pure black) leaked into a crop"
        assert crop[:20].min() == 255 and crop[:, :20].min() == 255          # white padding
        assert (out / r["mask_file"]).exists()
    a_crop = np.asarray(Image.open(out / by["A"]["crop_file"]).convert("RGB")).astype(int)
    assert ((a_crop[..., 0] == 200) & (a_crop[..., 1] == 40)).sum() > 40     # detached antenna tip kept
    assert (out / "qc" / "fig001_p001_qc.png").exists()


def test_row_layout_is_read_not_assumed(tmp_path):
    cells = make_plate(tmp_path / "p.png", layout="row")
    out, rows, prow = run(tmp_path, tmp_path / "p.png")
    assert _at(rows, cells) == {L: L for L in "ABCDEF"} and prow[0]["layout"] == "row"
    # forcing column order on a row plate assigns by position but says it disagrees with the letters read
    out, rows, prow = run(tmp_path, tmp_path / "p.png", extra=["--letter_order", "column"])
    assert "disagrees" in prow[0]["warnings"]


def test_touching_specimens_are_split(tmp_path):
    cells = make_plate(tmp_path / "p.png", touching=True)
    out, rows, prow = run(tmp_path, tmp_path / "p.png")
    assert len(rows) == 6 and _at(rows, cells) == {L: L for L in "ABCDEF"}
    assert "touching specimens" in prow[0]["warnings"]


def test_no_letters_falls_back_to_position_with_warning(tmp_path):
    cells = make_plate(tmp_path / "p.png", letters=False)
    out, rows, prow = run(tmp_path, tmp_path / "p.png")
    assert _at(rows, cells) == {L: L for L in "ABCDEF"}
    assert all(r["letter_method"] == "column" for r in rows)
    assert "no letters could be read" in prow[0]["warnings"]


def test_count_mismatch_and_unreadable_bar_are_warned(tmp_path):
    make_plate(tmp_path / "p.png")
    cap = CAP.replace("E and F) P. curticornis", "E to H) P. curticornis").replace("Scale bar 2 mm", "Scale bar 0.")
    out, rows, prow = run(tmp_path, tmp_path / "p.png", caption=cap)
    assert "6 specimens found but the caption has 8 letters" in prow[0]["warnings"]
    assert "G, H of the caption not found" in prow[0]["warnings"]
    assert all(r["px_per_mm"] == "" for r in rows)
    assert any("not readable" in r["warnings"] for r in rows)
    out, rows, prow = run(tmp_path, tmp_path / "p.png", caption=cap, extra=["--bar_mm", "2", "--no_share_scale"])
    by = {r["letter"]: r for r in rows}
    assert abs(float(by["A"]["px_per_mm"]) - 30) < 0.05 and by["B"]["px_per_mm"] == ""


def test_not_a_plate_is_skipped(tmp_path):
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 200, (400, 600, 3), dtype=np.uint8)).save(tmp_path / "map.png")
    out, rows, prow = run(tmp_path, tmp_path / "map.png")
    assert rows == [] and prow[0]["status"] == "skipped" and "not white" in prow[0]["warnings"]


def test_caption_table_matching_and_cli(tmp_path):
    pl = tmp_path / "plates"; pl.mkdir()
    make_plate(pl / "fig007_p012-000.png")
    make_plate(pl / "fig009_p015-000.png")
    with open(tmp_path / "m.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fig", "habitus", "caption"])
        w.writerow(["7", "True", CAP]); w.writerow(["7", "False", "Distribution maps."])
        w.writerow(["11", "True", CAP])
    r = subprocess.run([sys.executable, str(HERE / "descriptron_plate_split_v1.py"), "--plates", str(pl),
                        "--captions", str(tmp_path / "m.csv"), "--where", "habitus=True", "--key_col", "fig",
                        "--out_dir", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    prow = {p["plate"]: p for p in csv.DictReader(open(tmp_path / "o" / "plate_split_plates.csv"))}
    assert prow["fig007_p012-000.png"]["status"] == "ok" and prow["fig009_p015-000.png"]["status"] == "skipped"
    assert "fig=11" in r.stdout
    # a caption table keyed by file name
    with open(tmp_path / "f.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "caption"]); w.writerow(["fig009_p015-000.png", CAP])
    caps, notes = ps.match_captions(sorted(pl.iterdir()), tmp_path / "f.csv", None, "caption", r"fig0*(\d+)", [])
    assert [p.name for p in caps] == ["fig009_p015-000.png"]
