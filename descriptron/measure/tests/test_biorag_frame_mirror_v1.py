#!/usr/bin/env python3
"""
Frame orientation, part two: the anatomy next door, and mirrored frames.

`test_biorag_frames_v1.py` fixes the properties of the width-profile cue. These tests fix
the two cues added on 2026-09-21, on synthetic data where the answer is known by
construction:

  (i)   a structure photographed beside a neighbour that marks one of its ends. A known
        subset of the frames is built half a turn out; the adjacency cue must return
        exactly that subset and nothing else, with no threshold tuned to make it so.
  (ii)  a known subset of the images is MIRRORED. The per-image vote must recover exactly
        those images from the structures that can read them, carry the verdict to a
        structure that cannot, and the read-time transform must put a marked point back
        where it belongs — including when the frame's long axis is oblique, where the
        axis-aligned (x, 1 - y) and (1 - x, y) are both wrong.
  (iii) an orientation table WITHOUT the new columns must be read exactly as before:
        `read_time_transform` resolves to the half turn the heat-map script has always
        applied, and the heat-map script must not take the new code path at all.

Run:  python measure/tests/test_biorag_frame_mirror_v1.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import biorag_frame_orientation_v1 as fo                                  # noqa: E402
import biorag_vlm_heatmap_figures_v1 as hm                                # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# synthetic anatomy
# ─────────────────────────────────────────────────────────────────────────────
def club(n=64, knob=0.55):
    """A shaft 10 long and 1 wide with a knob at x = 0 — a tibia, roughly. Open ring."""
    x = np.linspace(0, 10, n // 2)
    half = 0.5 + knob * np.exp(-(x / 1.2) ** 2)
    return np.r_[np.c_[x, half], np.c_[x[::-1], -half[::-1]]]


def blob(centre, r=0.8, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[centre[0] + r * np.cos(a), centre[1] + r * 1.6 * np.sin(a)]


def sim(pts, deg, scale, shift):
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.asarray(pts, float) @ R.T * scale + np.asarray(shift, float)


def build_case(tmp: Path, turned: set, n_spec=12, canvas=512.0, seed=3):
    """A frames directory, a GPA directory and a COCO, exactly as the real pipeline lays
    them out. The structure 'bar' is a club; the neighbour 'nub' always sits at the club's
    KNOB end in the photograph, whatever the frame did with it."""
    rng = np.random.default_rng(seed)
    shape = club()
    cons = shape - shape.mean(0)
    cons = cons / np.abs(cons).max() * (canvas * 0.42) + canvas / 2.0
    sdir = tmp / "frames" / "bar"
    sdir.mkdir(parents=True)
    np.save(sdir / "consensus_px.npy", cons)

    gpa_imgs, gpa_anns, coco_imgs, coco_anns = [], [], [], []
    for i in range(n_spec):
        sid = f"sp_{i}"
        fn = f"img_{i}.tif"
        # the photograph: the club and, at its knob end, the nub
        img = sim(shape, rng.uniform(0, 360), rng.uniform(20, 40),
                  rng.uniform(200, 800, 2))
        knob_img = sim(np.array([[-1.6, 0.0]]), 0, 1, [0, 0])          # placeholder
        # the knob end of THIS club in image coordinates: landmark 0 is at x = 0
        end = img[0] + (img[0] - img[len(img) // 4])                   # just past the knob
        nub = blob([0, 0], 1.0) * (np.linalg.norm(img[1] - img[0]) * 2.0) + end
        # the frame: the similarity that puts the club on the consensus, half turned for
        # the specimens in `turned`
        tgt = cons if sid not in turned else (canvas - cons)
        R, sc, t, _res = fo.recover_similarity(img, tgt)
        lm_frame = (img @ R) * sc + t
        np.save(sdir / f"{sid}_landmarks.npy", lm_frame)

        gpa_imgs.append({"id": i, "file_name": f"{fn}_{i}"})
        gpa_anns.append({"id": i, "image_id": i, "category_id": 1,
                         "segmentation": [img.ravel().tolist()]})
        coco_imgs.append({"id": i, "file_name": fn})
        coco_anns.append({"id": 2 * i, "image_id": i, "category_id": 1,
                          "segmentation": [img.ravel().tolist()]})
        coco_anns.append({"id": 2 * i + 1, "image_id": i, "category_id": 2,
                          "segmentation": [nub.ravel().tolist()]})
        _ = knob_img
    gdir = tmp / "gpa" / "bar"
    gdir.mkdir(parents=True)
    (gdir / "back_transformed_coco.json").write_text(json.dumps(
        {"images": gpa_imgs, "annotations": gpa_anns,
         "categories": [{"id": 1, "name": "back_transformed"}]}))
    coco = tmp / "coco.json"
    coco.write_text(json.dumps({"images": coco_imgs, "annotations": coco_anns,
                                "categories": [{"id": 1, "name": "bar"},
                                               {"id": 2, "name": "nub"}]}))
    return sdir, tmp / "gpa", coco, cons


# ─────────────────────────────────────────────────────────────────────────────
def test_the_adjacency_cue_finds_exactly_the_frames_that_are_turned():
    turned = {"sp_2", "sp_5", "sp_9", "sp_11"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sdir, gpa, coco, cons = build_case(tmp, turned)
        polys = fo.load_coco_polygons(coco)
        u, _v = fo.principal_axis(fo.ring(cons))
        got = fo.orient_by_neighbour(sdir, "bar", gpa, polys, {"bar": ["nub"]}, {},
                                     fo.ring(cons), u,
                                     lambda fn: Path(fn).stem.replace("img_", "sp_"))
        assert len(got) == 12, f"only {len(got)} frames read"
        found = {s for s, v in got.items() if v[0]}
        assert found == turned, (sorted(found), sorted(turned))
        assert min(v[1] for v in got.values()) > 0.5, "a contact point near the middle"
        assert all(v[2] == "neighbour:nub" for v in got.values())


def test_the_adjacency_cue_needs_no_threshold_to_get_it_right():
    """The same case with every threshold at its most permissive and at a demanding value:
    the answer does not move, which is the whole claim made for this cue."""
    turned = {"sp_1", "sp_7"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sdir, gpa, coco, cons = build_case(tmp, turned, seed=11)
        polys = fo.load_coco_polygons(coco)
        u, _v = fo.principal_axis(fo.ring(cons))
        for margin in (0.0, 0.3, 0.8):
            got = fo.orient_by_neighbour(sdir, "bar", gpa, polys, {"bar": ["nub"]}, {},
                                         fo.ring(cons), u,
                                         lambda fn: Path(fn).stem.replace("img_", "sp_"),
                                         min_end_margin=margin)
            assert {s for s, v in got.items() if v[0]} == turned, margin


def test_a_frame_whose_contour_does_not_match_is_dropped_not_guessed():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sdir, gpa, coco, cons = build_case(tmp, set(), seed=5)
        np.save(sdir / "sp_3_landmarks.npy", np.load(sdir / "sp_3_landmarks.npy") * 1.0
                + np.random.default_rng(0).normal(0, 9.0, (64, 2)))    # a different shape
        polys = fo.load_coco_polygons(coco)
        u, _v = fo.principal_axis(fo.ring(cons))
        got = fo.orient_by_neighbour(sdir, "bar", gpa, polys, {"bar": ["nub"]}, {},
                                     fo.ring(cons), u,
                                     lambda fn: Path(fn).stem.replace("img_", "sp_"))
        assert "sp_3" not in got and len(got) == 11


# ─────────────────────────────────────────────────────────────────────────────
def test_the_per_image_vote_recovers_the_mirrored_images_and_carries_them():
    """Two structures can read the handedness, a third cannot. The vote must decide the
    image from the two that can, and hand the answer to the third."""
    mirrored_images = {"img_1", "img_4", "img_6"}
    reg, image_of = {}, {}
    for i in range(8):
        img = f"img_{i}"
        m = img in mirrored_images
        for st, readable in (("readsA", True), ("readsB", True), ("symmetric", False)):
            sid = f"sp_{i}"
            # the near-symmetric structure reads noise, and says so (ambiguous)
            reg[(st, sid)] = ((m if readable else bool(i % 3 == 0)), (not readable), 0.5)
            image_of[(st, sid)] = img
    verdict, per_image = fo.mirror_vote(reg, image_of, None, 1)
    assert {i for i, v in per_image.items() if v[0]} == mirrored_images, per_image
    for i in range(8):
        for st in ("readsA", "readsB", "symmetric"):
            assert verdict[(st, f"sp_{i}")] == int(f"img_{i}" in mirrored_images)


def test_the_vote_ignores_a_structure_that_is_not_on_the_reliable_list():
    reg, image_of = {}, {}
    for i in range(8):
        reg[("good", f"sp_{i}")] = (i in (1, 4), False, 0.9)
        reg[("liar", f"sp_{i}")] = (i in (0, 2, 3, 5, 6, 7), False, 0.9)
        image_of[("good", f"sp_{i}")] = image_of[("liar", f"sp_{i}")] = f"img_{i}"
    v, _pi = fo.mirror_vote(reg, image_of, ["good"], 1)
    assert {i for i in range(8) if v[("liar", f"sp_{i}")]} == {1, 4}


def test_the_read_time_transform_puts_a_marked_point_back_on_an_oblique_axis():
    """The frames' long axis is wherever the Procrustes consensus happened to leave it —
    on this data set 3 to 180 degrees — so the correction is a reflection about THAT line,
    and the axis-aligned (x, 1 - y) / (1 - x, y) are the wrong answer except by luck."""
    rng = np.random.default_rng(7)
    for axis in (0.0, 37.0, 90.0, 131.0):
        pts = 0.5 + rng.uniform(-0.3, 0.3, (6, 2))
        mirrored = fo.apply_frame_transform(pts, "mirror", axis)
        assert np.allclose(fo.apply_frame_transform(mirrored, "mirror", axis), pts)
        # a reflection is not a rotation: no half turn can undo it unless the points happen
        # to be symmetric, which random points are not
        assert not np.allclose(fo.apply_frame_transform(mirrored, "halfturn"), pts, atol=1e-2)
        assert not np.allclose(fo.apply_frame_transform(mirrored, "identity"), pts, atol=1e-2)
        # mirror then half turn is the reflection about the perpendicular, and nothing else
        both = fo.apply_frame_transform(fo.apply_frame_transform(pts, "mirror", axis),
                                        "halfturn")
        assert np.allclose(both, fo.apply_frame_transform(pts, "mirror_halfturn", axis))
        if axis not in (0.0, 90.0):
            assert not np.allclose(fo.apply_frame_transform(mirrored, "flipy"), pts, atol=1e-2)
            assert not np.allclose(fo.apply_frame_transform(mirrored, "flipx"), pts, atol=1e-2)


def test_the_outline_test_chooses_the_transform_that_restores_a_chiral_configuration():
    """`best_read_time_transform` on a deliberately chiral outline: the mirrored copy must
    be given a reflection, and the copy as it stands must be left alone."""
    n, canvas = 64, 512.0
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + 0.35 * np.sin(a) + 0.22 * np.sin(3 * a + 0.7) + 0.15 * np.cos(2 * a - 1.1)
    cons = np.c_[r * np.cos(a), r * np.sin(a)] * 150 + canvas / 2.0
    axis = fo.consensus_axis_degrees(cons)
    same, _d, gap = fo.best_read_time_transform(cons, cons, canvas, axis)
    assert same == "identity" and gap > 0.0
    mirrored = fo.apply_frame_transform(cons / canvas, "mirror", axis) * canvas
    name, _d2, gap2 = fo.best_read_time_transform(mirrored, cons, canvas, axis)
    assert name == "mirror", name
    assert gap2 > 0.0


# ─────────────────────────────────────────────────────────────────────────────
def _write_table(path: Path, rows, new_columns: bool):
    head = ["structure", "specimen_id", "flipped", "method"]
    if new_columns:
        head += ["mirrored", "mirror_axis_deg", "read_time_transform"]
    lines = ["\t".join(head)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in head))
    path.write_text("\n".join(lines) + "\n")


def test_a_table_without_the_new_columns_behaves_exactly_as_before():
    rows = [dict(structure="bar", specimen_id=f"sp_{i}", flipped=int(i % 3 == 0),
                 method="profile") for i in range(9)]
    with tempfile.TemporaryDirectory() as td:
        old = Path(td) / "old.tsv"
        _write_table(old, rows, new_columns=False)
        tr, _counts = hm.load_transforms(old)
        assert len(tr) == 9
        for i in range(9):
            name, ang = tr[("bar", f"sp_{i}")]
            assert name == ("halfturn" if i % 3 == 0 else "identity")
            assert ang == 0.0
        assert not any(v[0] in ("mirror", "mirror_halfturn", "flipx", "flipy")
                       for v in tr.values())
        # and the half turn it resolves to is the map the script has always applied
        p = np.array([[0.2, 0.7], [0.9, 0.1]])
        assert np.allclose(hm.transform_unit(p, "halfturn"), 1.0 - p)
        assert np.allclose(hm.transform_unit(p, "identity"), p)
        im = np.arange(16, dtype=np.float32).reshape(4, 4)
        assert np.array_equal(hm.transform_image(im, "halfturn"), im[::-1, ::-1])
        assert np.array_equal(hm.transform_image(im, "identity"), im)


def test_a_table_with_the_new_columns_is_read_and_agrees_with_the_orientation_module():
    rows = [dict(structure="bar", specimen_id=f"sp_{i}", flipped=int(i % 2),
                 method="neighbour", mirrored=int(i % 3 == 0), mirror_axis_deg=37.0,
                 read_time_transform=("mirror_halfturn" if (i % 3 == 0 and i % 2)
                                      else "mirror" if i % 3 == 0
                                      else "halfturn" if i % 2 else "identity"))
            for i in range(12)]
    with tempfile.TemporaryDirectory() as td:
        new = Path(td) / "new.tsv"
        _write_table(new, rows, new_columns=True)
        tr, _counts = hm.load_transforms(new)
        assert any(v[0] in ("mirror", "mirror_halfturn") for v in tr.values())
        rng = np.random.default_rng(1)
        p = 0.5 + rng.uniform(-0.3, 0.3, (5, 2))
        for (st, sid), (name, ang) in tr.items():
            # the two modules must agree point for point, or the table and the figures
            # would disagree about what a correction means
            assert np.allclose(hm.transform_unit(p, name, ang),
                               fo.apply_frame_transform(p, name, ang)), name


def test_the_image_transform_matches_the_point_transform():
    """The background frame and the reports must move together; a reflection applied to
    one and not the other is worse than no correction at all."""
    n = 96
    im = np.zeros((n, n), np.float32)
    im[20:30, 60:70] = 255.0                                   # a mark, off both axes
    for name, ang in (("halfturn", 0.0), ("flipx", 0.0), ("flipy", 0.0),
                      ("mirror", 0.0), ("mirror", 44.0), ("mirror_halfturn", 44.0)):
        moved = hm.transform_image(im, name, ang)
        src = np.array([[64.5 / (n - 1), 24.5 / (n - 1)]])     # the mark, in unit coords
        want = hm.transform_unit(src, name, ang)[0] * (n - 1)
        ys, xs = np.nonzero(moved > 0.5 * moved.max())
        w = moved[ys, xs]
        got = np.array([float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())])
        assert np.linalg.norm(got - want) < 2.0, (name, ang, got, want)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print(f"  ok   {f.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL {f.__name__}: {e}")
        except Exception as e:                                             # noqa: BLE001
            bad += 1
            print(f"  ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
