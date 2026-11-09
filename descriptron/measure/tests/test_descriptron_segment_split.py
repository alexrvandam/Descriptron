"""Tests for descriptron_segment_split_v1: coarse part masks -> fine homologous parts (no torch needed).

Synthetic specimens with known answers: a segmented abdomen whose intersegmental lines are boundary ridges (plus a decoy
colour band of two full-width ridges inside one segment, and an optional narrow apex), two elytra with a suture ridge off
the midline (and without one: midline fallback), a three-segment leg with joint ridges, six such legs named by the serial
homology rule; foreground windows kept and pinholes filled; the watershed moving a boundary onto a ridge; polygon
round trip; the CLI with --tokens off (image gradient only) writing new files and leaving the input unchanged.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_segment_split_v1 as ss  # noqa: E402

H, W = 900, 300
CUTS = [450, 530, 610, 690, 770]               # intersegmental lines (image y) of a 370..850 abdomen: 6 segments of 80
ABD = (90, 210, 370, 850)                      # x0, x1, y0, y1


def opts(*extra):
    return ss.build_parser().parse_args(["in.json", "--out", "out.json", "--view", "dorsal"] + list(extra))


def ellipse(cx, cy, rx, ry):
    yy, xx = np.mgrid[:H, :W]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def rect(x0, x1, y0, y1):
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
    return m


def specimen(apex=False):
    """Head up, dorsal: head, pronotum, elytra, abdomen (+ narrow apex). -> {name: mask}"""
    m = {"head": ellipse(150, 80, 50, 40), "pronotum": rect(95, 205, 125, 215), "elytra": rect(80, 220, 215, 370),
         "abdomen": rect(*ABD)}
    if apex:
        m["abdomen"] = m["abdomen"] | rect(130, 170, 850, 890)          # narrow, abrupt: tergite 9
    return m


def labels_and_body(masks):
    fg = np.zeros((H, W), bool)
    for v in masks.values():
        fg |= v
    o = opts()
    L = ss.build_labels(masks, fg, ss.role_matcher(o), o)
    body, warn = ss.body_of(L, o)
    assert body is not None, warn
    return L, body, o


def abdomen_maps(decoy=True, apex=False):
    """Line strength across the axis: ridges at the cuts over the full width; notches at the margins; a decoy colour
    band (two full-width ridges 20 px apart) inside the third segment."""
    b = np.zeros((H, W), np.float32)
    for y in CUTS + ([850] if apex else []):
        b[y - 1:y + 2, :] = 1.0
    if decoy:
        for y in (628, 648):
            b[y - 1:y + 2, :] = 1.0
    b += 0.05 * np.random.default_rng(0).random((H, W)).astype(np.float32)
    return ss.Maps.from_boundary({"L18line": b, "L18": b}, normal={"L18line": (0.0, 1.0), "L18": (0.0, 1.0)})


def cut_ys(parts):
    """Boundaries between consecutive numbered segments (median y of the shared edge)."""
    segs = sorted([p for p in parts if isinstance(p.get("segment"), int)], key=lambda p: p["segment"])
    out = []
    for a, b in zip(segs, segs[1:]):
        edge = a["mask"] & ss.dilate(b["mask"], 1.5)
        out.append(float(np.median(np.nonzero(edge)[0])))
    return out


# ------------------------------------------------------------------ abdomen
def test_abdomen_cuts_numbered_from_anterior_and_decoy_band_rejected():
    L, body, o = labels_and_body(specimen())
    parts, qc, cands = ss.split_abdomen(L.mask("abdomen"), abdomen_maps(), body, "dorsal", o)
    names = [p["name"] for p in sorted(parts, key=lambda p: np.nonzero(p["mask"])[0].mean())]
    assert names == [f"tergite {n}" for n in range(3, 9)]
    ys = cut_ys(parts)
    assert len(ys) == 5 and all(abs(a - b) <= 3 for a, b in zip(ys, CUTS)), ys
    rejected = [c for c in cands if not c["accepted"]]
    assert any(abs(c["y"] - 628) <= 4 for c in rejected) or any(abs(c["y"] - 648) <= 4 for c in rejected)
    assert all(p["apex_present"] == "no" for p in parts)


def test_abdomen_apex_does_not_inflate_the_regular_count():
    L, body, o = labels_and_body(specimen(apex=True))
    parts, qc, cands = ss.split_abdomen(L.mask("abdomen"), abdomen_maps(apex=True), body, "dorsal", o)
    names = sorted(p["name"] for p in parts)
    assert "tergite 9" in names and "tergite 8" in names and "tergite 3" in names
    assert sum(isinstance(p.get("segment"), int) and p["segment"] <= 8 for p in parts) == 6
    assert all(p["apex_present"] == "yes" for p in parts)
    t9 = next(p for p in parts if p["name"] == "tergite 9")
    assert np.nonzero(t9["mask"])[0].min() >= 845


def test_abdomen_options_numbering_and_no_apex():
    L, body, o = labels_and_body(specimen(apex=True))
    o2 = opts("--first_segment", "1", "--last_regular_segment", "7", "--apex", "none",
              "--segment_name", "abdominal segment {n}")
    parts, _, _ = ss.split_abdomen(L.mask("abdomen"), abdomen_maps(apex=True), body, "dorsal", o2)
    names = {p["name"] for p in parts}
    assert "abdominal segment 1" in names and "tergite 9" not in names
    assert all(p["apex_present"] == "no" for p in parts)


def test_ventral_segment_names():
    L, body, o = labels_and_body(specimen())
    parts, _, _ = ss.split_abdomen(L.mask("abdomen"), abdomen_maps(), body, "ventral", o)
    assert {p["name"] for p in parts} >= {"sternite 3", "sternite 8"}


# ------------------------------------------------------------------ elytra
def test_elytra_split_follows_the_suture_ridge():
    L, body, o = labels_and_body(specimen())
    b = np.zeros((H, W), np.float32)
    ys = np.arange(215, 370)
    xs = np.round(156 + 3 * np.sin(ys / 30.0)).astype(int)            # off the midline (150) by ~6 px, curved
    for y, x in zip(ys, xs):
        b[y, x - 1:x + 2] = 1.0
    mp = ss.Maps.from_boundary({"L18line": b}, normal={"L18line": (1.0, 0.0)})
    parts, qc = ss.split_elytra(L.mask("elytra"), mp, body, "dorsal", o)
    by = {p["name"]: p for p in parts}
    assert set(by) == {"left_elytron", "right_elytron"}
    assert by["left_elytron"]["method"].startswith("L18 ridge")
    # head up, dorsal: the specimen's left is on the image's left; the boundary sits on the ridge
    left = by["left_elytron"]["mask"]
    err = [abs(np.nonzero(left[y])[0].max() + 0.5 - x) for y, x in zip(ys[5:-5], xs[5:-5])]
    assert np.median(err) <= 1.5 and max(err) <= 3


def test_elytra_without_ridge_fall_back_to_the_midline_and_ventral_mirrors():
    L, body, o = labels_and_body(specimen())
    mp = ss.Maps.from_boundary({"L18line": np.zeros((H, W), np.float32)})
    parts, _ = ss.split_elytra(L.mask("elytra"), mp, body, "ventral", o)
    by = {p["name"]: p for p in parts}
    assert by["left_elytron"]["method"].startswith("rule fallback")
    # ventral: the specimen's left is on the image's RIGHT
    assert np.nonzero(by["left_elytron"]["mask"])[1].mean() > 150 > np.nonzero(by["right_elytron"]["mask"])[1].mean()


# ------------------------------------------------------------------ legs
def strip(p0, p1, w0, w1=None):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-d[1], d[0]])
    w1 = w0 if w1 is None else w1
    return [tuple(p0 + n * w0 / 2), tuple(p1 + n * w1 / 2), tuple(p1 - n * w1 / 2), tuple(p0 - n * w0 / 2)]


def draw(polys, shape=(H, W)):
    img = Image.new("L", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(img)
    for p in polys:
        dr.polygon(p, fill=1, outline=1)
    return np.asarray(img, bool)


def leg_masks(base, direction, lengths=(70, 70, 50), widths=(14, 9, 6), bend=(0, 25, 10)):
    """Femur, tibia, tarsus from `base` along `direction` (deg, image coords), bent at the joints. -> masks, joints."""
    pts, ang, p = [], direction, np.asarray(base, float)
    out, joints = [], []
    for L_, w_, b_ in zip(lengths, widths, bend):
        ang += b_
        q = p + L_ * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        out.append(draw([strip(p, q, w_, w_ * 0.9)]) | ellipse(p[0], p[1], w_ / 2, w_ / 2))
        joints.append(q); p = q
    return out, joints[:-1]


def test_three_segment_leg_cut_at_its_joint_ridges():
    trunk = rect(100, 200, 200, 600)
    segs, joints = leg_masks((200, 300), 60)
    m = np.zeros((H, W), bool)
    for s_ in segs:
        m |= s_
    m &= ~trunk
    b = np.zeros((H, W), np.float32)
    for j in joints:
        b = np.maximum(b, ellipse(j[0], j[1], 2.5, 2.5).astype(np.float32))
    mp = ss.Maps.from_boundary({"L18line": b})
    parts, qc = ss.split_leg(m, trunk, mp, "dorsal", opts())
    names = [p["generic"] for p in parts]
    assert names == ["femur", "tibia", "tarsus"]
    cuts = [(x, y) for x, y, *_ in qc["cuts"]]
    assert len(cuts) == 2
    for (cx, cy), j in zip(cuts, joints):                             # joints placed on the ridges, in order
        assert math.hypot(cx - j[0], cy - j[1]) <= 4, ((cx, cy), j)
    assert parts[0]["mask"][305, 203]                                 # the piece at the body is the femur
    assert all(p["method"] == "L18 joint ridge" for p in parts)


def test_six_legs_named_by_serial_homology_prefix():
    masks = specimen()
    trunk = np.zeros((H, W), bool)
    for v in masks.values():
        trunk |= v
    legs = np.zeros((H, W), bool)
    b = np.zeros((H, W), np.float32)
    for y, ang in ((180, -30), (290, 10), (400, 40)):
        for side, sx in (("right", 1), ("left", -1)):
            x0 = 150 + sx * 60
            a = ang if sx > 0 else 180 - ang
            segs, joints = leg_masks((x0, y), a, lengths=(60, 60, 40), widths=(12, 8, 5),
                                     bend=(0, 25 * sx, 10 * sx))
            for s_ in segs:
                legs |= s_
            for j in joints:
                b = np.maximum(b, ellipse(j[0], j[1], 2.5, 2.5).astype(np.float32))
    legs &= ~trunk
    masks["leg"] = legs
    L, body, o = labels_and_body(masks)
    mp = ss.Maps.from_boundary({"L18line": b})
    trunk_l = (L.lab > 0) & ~L.mask("leg")
    cl, n = ss.label8(L.mask("leg"))
    parts = []
    for i in range(1, n + 1):
        ps, _ = ss.split_leg(cl == i, trunk_l, mp, "dorsal", o)
        parts += ps
    ss.name_legs(parts, L, {"file_name": "synthetic_dorsal.png"}, "dorsal", o)
    names = sorted(p["name"] for p in parts)
    expected = sorted(f"{side}_{seg}{part}" for side in ("left", "right") for seg in ("pro", "meso", "meta")
                      for part in ("femur", "tibia", "tarsus"))
    assert names == expected
    lpf = next(p for p in parts if p["name"] == "left_profemur")
    assert np.nonzero(lpf["mask"])[1].mean() < 150                   # dorsal: specimen's left = image left


# ------------------------------------------------------------------ foreground, watershed, polygons, names
def test_derived_foreground_keeps_windows_and_fills_pinholes():
    img = np.full((200, 200, 3), 255, np.uint8)
    img[60:140, 60:140] = (40, 30, 20)                               # body
    img[100:103, 100:103] = 255                                      # pinhole (glare speck)
    img[20:60, 60:66] = (40, 30, 20); img[20:26, 60:140] = (40, 30, 20); img[20:60, 134:140] = (40, 30, 20)  # loop
    o = opts()
    fg = ss.derive_fg(img, o)
    fg2, npin, _ = ss.fill_pinholes(fg, ss.pinhole_px(fg.shape, o))
    assert fg2[101, 101] and npin >= 1
    assert not fg2[40, 100]                                          # window enclosed by the loop stays background


def test_coarse_window_removed_by_the_foreground():
    masks = specimen()
    fg = np.zeros((H, W), bool)
    for v in masks.values():
        fg |= v
    masks["leg"] = rect(220, 260, 300, 340)                           # covers background only
    o = opts("--tokens", "off")
    mp = ss.Maps.from_boundary({"img": np.zeros((H, W), np.float32)})
    res = ss.split_image(np.zeros((H, W, 3), np.uint8), masks, fg, mp, "dorsal", {"file_name": "x"}, o)
    assert res["summary"]["windows_removed"] == 1
    assert not any((p["mask"] & ~fg).any() for p in res["parts"])


def test_watershed_moves_the_boundary_onto_the_ridge():
    lab = np.zeros((60, 100), np.int16); lab[:, 10:50] = 1; lab[:, 50:90] = 2
    cost = np.zeros((60, 100)); cost[:, 56] = 5.0
    out = ss.sharpen(lab, cost, lab > 0, band=10)
    first2 = [int(np.nonzero(r == 2)[0].min()) for r in out[5:-5]]
    assert all(55 <= x <= 57 for x in first2)
    assert (out == 1).sum() > 0 and (out == 2).sum() > 0


def test_polygon_round_trip():
    m = ellipse(150, 400, 60, 120) | rect(140, 160, 200, 300)
    polys = ss.mask_polygons(m, eps=1.0)
    r = ss.sh.raster([np.asarray(p).reshape(-1, 2) for p in polys], W, H)
    assert ss.iou(r, m) > 0.97


def test_names_and_rename():
    assert ss.canon("profemur (left)") == ss.canon("left_profemur") == "left_profemur"
    assert ss.canon("antenna (right)") == "right_antenna" and ss.canon("elytra") == "elytron"
    assert ss.parse_rename("mandibles=mouthparts, A=b") == {"mandibles": "mouthparts", "a": "b"}
    role = ss.role_matcher(opts())
    assert role("abdomen") == "abdomen" and role("body (SAM2)") == "body" and role("mouthparts") == "mouthparts"


# ------------------------------------------------------------------ CLI
def test_cli_tokens_off_writes_new_files(tmp_path):
    masks = specimen()
    rgb = np.full((H, W, 3), 255, np.uint8)
    fg = np.zeros((H, W), bool)
    for v in masks.values():
        fg |= v
    rgb[fg] = (60, 40, 30)
    for y in CUTS:                                                   # pale intersegmental lines on the photograph
        rgb[y - 1:y + 2, ABD[0]:ABD[1]] = (200, 180, 150)
    Image.fromarray(rgb).save(tmp_path / "spec_dorsal.png")
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(masks)]
    anns = []
    for i, (n, m) in enumerate(masks.items()):
        anns.append({"id": i + 1, "image_id": 1, "category_id": i + 1, "segmentation": ss.mask_polygons(m),
                     "area": float(m.sum()), "bbox": [0, 0, 1, 1], "iscrowd": 0})
    coco = {"images": [{"id": 1, "file_name": "spec_dorsal.png", "width": W, "height": H}], "categories": cats,
            "annotations": anns}
    inp = tmp_path / "coarse.json"
    inp.write_text(json.dumps(coco))
    before = inp.read_text()
    out = tmp_path / "fine.json"
    rc = ss.main([str(inp), "--out", str(out), "--images", str(tmp_path), "--view_regex", "_(dorsal|ventral)",
                  "--tokens", "off", "--qc_dir", str(tmp_path / "qc")])
    assert rc == 0 and inp.read_text() == before
    d = json.loads(out.read_text())
    names = {c["name"] for c in d["categories"]}
    assert {"left_elytron", "right_elytron", "head", "pronotum"} <= names
    assert any(n.startswith("tergite") for n in names)
    assert (tmp_path / "fine_parts.csv").exists() and (tmp_path / "fine_images.csv").exists()
    assert (tmp_path / "qc" / "spec_dorsal_segsplit_qc.png").exists()
