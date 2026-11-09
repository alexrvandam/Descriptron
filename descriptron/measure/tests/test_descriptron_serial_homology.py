"""Tests for descriptron_serial_homology_v1: generic leg parts -> 'profemur (left)' etc. by rule.

Synthetic insect (body ellipse, head, six legs of femur + tibia + tarsus, optional coxae and landmarks) with known
answers in dorsal and ventral view, rotated, with a leg missing; global (Hungarian) linking; name matching; the CLI
writes a new file and keeps the input; and, when a hand-labelled Oxytelus file is present, the round trip
(names stripped to generic -> rule -> compared with the hand labels), mirrored and rotated.
"""
import copy
import csv
import json
import math
import re
import sys
from pathlib import Path

import os

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import descriptron_serial_homology_v1 as sh  # noqa: E402

# hand-labelled Oxytelus COCO for the real-data checks; point DESCRIPTRON_TEST_OXYTELUS_COCO at it, otherwise those tests skip
OXY = Path(os.environ.get("DESCRIPTRON_TEST_OXYTELUS_COCO", "/nonexistent/oxytelus_corrected_coco_sutures.json"))
GENERIC = {"coxa": 101, "femur": 102, "tibia": 103, "tarsus": 104, "trochanter": 105}


# ------------------------------------------------------------------ synthetic insect
def ellipse(cx, cy, rx, ry, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])


def strip(p0, p1, w0, w1=None):
    """Tapered quadrilateral from p0 (width w0) to p1 (width w1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-d[1], d[0]])
    w1 = w0 if w1 is None else w1
    return np.array([p0 + n * w0 / 2, p1 + n * w1 / 2, p1 - n * w1 / 2, p0 - n * w0 / 2])


def unit(deg):
    """Direction for a RIGHT-side leg (image right, head up): 0 = lateral, negative = forwards, positive = backwards."""
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


# femur angle, knee bend (tibia relative to femur), tarsus bend; right-side legs, mirrored for the left
POSE = {"pro": (-40, -35, -10), "meso": (5, 45, 15), "meta": (35, 40, 10)}
ATTACH_Y = {"pro": 290, "meso": 370, "meta": 450}


def insect(coxae=False, landmarks=False, pose=POSE, lengths=(70, 80, 45), cx=300.0, trochanters=False):
    """-> (polys {name: [array]}, truth {name: true full name in dorsal view}) for a head-up insect."""
    polys = {"body": [ellipse(cx, 420, 70, 230)], "head": [ellipse(cx, 165, 45, 40)]}
    if landmarks:
        polys["pronotum"] = [ellipse(cx, 265, 60, 50)]
        polys["abdomen"] = [ellipse(cx, 545, 62, 110)]
    truth = {}
    for seg, y in ATTACH_Y.items():
        fa, kb, tb = pose[seg]
        for side, sx in (("right", 1), ("left", -1)):
            hw = 70 * math.sqrt(max(0.0, 1 - ((y - 420) / 230) ** 2))
            base = np.array([cx + sx * (hw - 6), y])
            d1 = unit(fa) * [sx, 1]
            if trochanters:                                    # coxa -> trochanter -> femur
                tr0 = base; base = base + d1 * 14
                polys[f"{seg}trochanter ({side})"] = [strip(tr0, base, 15, 14)]
                if coxae:
                    polys[f"{seg}coxa ({side})"] = [strip(np.array([cx + sx * (hw - 34), y]), tr0, 22, 18)]
            elif coxae:
                c0 = np.array([cx + sx * (hw - 34), y]); polys[f"{seg}coxa ({side})"] = [strip(c0, base, 22, 18)]
            knee = base + d1 * lengths[0]
            d2 = unit(fa + kb) * [sx, 1]
            ank = knee + d2 * lengths[1]
            d3 = unit(fa + kb + tb) * [sx, 1]
            tip = ank + d3 * lengths[2]
            polys[f"{seg}femur ({side})"] = [strip(base, knee, 16, 13)]
            polys[f"{seg}tibia ({side})"] = [strip(knee, ank, 10, 8)]
            polys[f"{seg}tarsus ({side})"] = [strip(ank, tip, 6, 4)]
    truth = {k: k for k in polys if sh.FULL_NAME_RE.match(k)}
    return polys, truth


def transform(polys, f):
    return {k: [np.array([f(*p) for p in P]) for P in v] for k, v in polys.items()}


def rotate(polys, deg, about=(300, 420)):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    out = transform(polys, lambda x, y: (about[0] + c * (x - about[0]) - s * (y - about[1]),
                                         about[1] + s * (x - about[0]) + c * (y - about[1])))
    allp = np.vstack([P for v in out.values() for P in v]); shift = 20 - allp.min(0)
    return transform(out, lambda x, y: (x + shift[0], y + shift[1]))


def prefix(n):
    """'profemur (left)' -> 'left_profemur' (DINOMask style)."""
    side, seg, part = sh.parse_name(n)
    return f"{side}_{seg}{part}"


def to_coco(images, model_names=None):
    """images: [(file_name, polys)] -> COCO with leg parts as GENERIC categories, or named by model_names
    {true name: category name} (a 'model' prediction); returns (coco, truth by ann id)."""
    cats = [{"id": v, "name": k, "supercategory": ""} for k, v in GENERIC.items()]
    other = {}
    ims, anns, truth = [], [], {}
    for i, (fn, polys) in enumerate(images, 1):
        allp = np.vstack([P for v in polys.values() for P in v])
        ims.append({"id": i, "file_name": fn, "width": int(allp[:, 0].max()) + 20, "height": int(allp[:, 1].max()) + 20})
        for name, P in polys.items():
            m = sh.FULL_NAME_RE.match(name)
            if m and model_names is not None and name in model_names:
                mn = model_names[name]
                if mn not in other:
                    other[mn] = 200 + len(other); cats.append({"id": other[mn], "name": mn, "supercategory": ""})
                cid = other[mn]
            elif m:
                cid = GENERIC[m.group(2)]
            else:
                if name not in other:
                    other[name] = 1 + len(other); cats.append({"id": other[name], "name": name, "supercategory": ""})
                cid = other[name]
            aid = 1000 * i + len(anns)
            anns.append({"id": aid, "image_id": i, "category_id": cid, "iscrowd": 0,
                         "segmentation": [[round(float(v), 2) for xy in p for v in xy] for p in P]})
            if m:
                truth[aid] = name
    return {"images": ims, "annotations": anns, "categories": cats}, truth


def run(coco, *args):
    opts = sh.build_parser().parse_args(["in.json", "--out", "out.json", *args])
    return sh.run(coco, opts)


def swap_sides(n):
    return n.replace("(left)", "(L)").replace("(right)", "(left)").replace("(L)", "(right)")


def accuracy(rows, truth, expect=lambda n: n):
    got = {r["annotation_id"]: r["final_name"] for r in rows}
    return sum(got.get(a) == expect(n) for a, n in truth.items()), len(truth)


# ------------------------------------------------------------------ synthetic tests
def test_dorsal_names_all_legs():
    polys, _ = insect()
    coco, truth = to_coco([("ins.png", polys)])
    out, rows, _ = run(coco, "--view", "dorsal")
    assert accuracy(rows, truth) == (18, 18)
    assert all(r["rank_method"] == "rank" for r in rows)
    assert all(r["confidence"] == "high" for r in rows)


def test_ventral_view_mirrors_sides_and_coxae_anchor_the_legs():
    polys, _ = insect(coxae=True)
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "ventral")                # same picture, called ventral: sides swap
    assert accuracy(rows, truth, swap_sides) == (24, 24)
    assert {r["anchor"].split()[0] for r in rows} == {"coxa"}


def test_horizontal_flip_dorsal_swaps_ventral_keeps():
    polys, _ = insect()
    flipped = transform(polys, lambda x, y: (600 - x, y))
    coco, truth = to_coco([("ins.png", flipped)])
    assert accuracy(run(coco, "--view", "dorsal")[1], truth, swap_sides) == (18, 18)
    assert accuracy(run(coco, "--view", "ventral")[1], truth) == (18, 18)


@pytest.mark.parametrize("deg", [37, 90, 180, 270, -60])
def test_rotation_does_not_change_names(deg):
    polys, _ = insect(coxae=True)
    coco, truth = to_coco([("ins.png", rotate(polys, deg))])
    assert accuracy(run(coco, "--view", "dorsal")[1], truth) == (24, 24)


def test_anterior_option_without_head():
    polys, _ = insect()
    polys = rotate(polys, 90)                                  # head now points to the image's right
    del polys["head"]
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "dorsal")
    assert accuracy(rows, truth)[0] == 0 and "anterior end unknown" in rows[0]["warnings"]
    assert accuracy(run(coco, "--view", "dorsal", "--anterior", "right")[1], truth) == (18, 18)
    assert accuracy(run(coco, "--view", "dorsal", "--anterior", "left")[1], truth)[0] == 0


def test_one_leg_missing_uses_the_other_side():
    polys, _ = insect()
    for p in ("femur", "tibia", "tarsus"):
        del polys[f"meso{p} (left)"]
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "dorsal")
    assert accuracy(rows, truth) == (15, 15)
    assert {r["rank_method"] for r in rows if "(left)" in r["final_name"]} == {"contralateral"}


def test_hidden_femur_tibia_anchors_the_leg():
    # a femur hidden under the body (dorsal view): the tibia emerges at the body outline and anchors the leg
    polys, truth = insect()
    for seg, side in (("pro", "left"), ("meta", "right")):
        fem = polys.pop(f"{seg}femur ({side})")[0]
        shift = fem[[0, 3]].mean(0) - fem[[1, 2]].mean(0)       # knee -> base: move tibia and tarsus to the body edge
        for p in ("tibia", "tarsus"):
            polys[f"{seg}{p} ({side})"] = [P + shift for P in polys[f"{seg}{p} ({side})"]]
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "dorsal")
    assert accuracy(rows, truth) == (16, 16)
    assert sum("anchored on a tibia" in r["warnings"] for r in rows) == 4


def test_orphan_tibia_away_from_the_body_is_a_fragment():
    # the model missed a femur that sticks out: its tibia starts a femur-length from the body -> not ranked, not guessed
    polys, _ = insect()
    del polys["mesofemur (left)"]
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "dorsal")
    got = {truth[r["annotation_id"]]: r for r in rows}
    assert got["mesotibia (left)"]["final_name"] == "" and "fragment" in got["mesotibia (left)"]["warnings"]
    assert got["mesotarsus (left)"]["final_name"] == ""
    assert accuracy(rows, truth) == (15, 17)                    # the other legs: contralateral, all correct


def test_fewer_legs_not_forced_without_landmarks_but_named_with_them():
    for lm in (False, True):
        polys, _ = insect(landmarks=lm)
        for side in ("left", "right"):
            for p in ("femur", "tibia", "tarsus"):
                del polys[f"meso{p} ({side})"]
        coco, truth = to_coco([("ins.png", polys)])
        _, rows, _ = run(coco, "--view", "dorsal")
        if not lm:
            assert all(r["final_name"] == "" and "not forced" in r["warnings"] for r in rows)
        else:
            assert accuracy(rows, truth) == (12, 12)
            assert {r["rank_method"] for r in rows} == {"landmarks"}


def test_no_view_means_no_side():
    polys, _ = insect()
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco)
    assert accuracy(rows, truth)[0] == 0 and all("view unknown" in r["warnings"] for r in rows)


def test_view_per_image_from_file_name_and_field():
    polys, _ = insect()
    coco, truth = to_coco([("sp1_dorsal.png", polys), ("sp1_ventral.png", polys), ("sp2.png", polys)])
    coco["images"][2]["view"] = "ventral"
    _, rows, _ = run(coco, "--view_regex", r"_(dorsal|ventral)\b", "--view", "dorsal")
    by_im = {}
    for r in rows:
        by_im.setdefault(r["image_id"], []).append(r["final_name"] == (truth[r["annotation_id"]] if r["image_id"] == 1
                                                                          else swap_sides(truth[r["annotation_id"]])))
    assert all(all(v) for v in by_im.values()) and len(by_im) == 3


def test_hungarian_two_tibiae_cannot_share_a_femur():
    # femur A's distal end is nearest to BOTH tibiae; a greedy nearest-femur rule would give A two tibiae
    pts = lambda P: np.array([(x, y) for x in np.arange(P[0], P[1]) for y in np.arange(P[2], P[3])], float)
    fa = sh.Part({"id": "A"}, "femur", pts((0, 60, 0, 10)))
    fb = sh.Part({"id": "B"}, "femur", pts((0, 60, 30, 40)))
    t1 = sh.Part({"id": "t1"}, "tibia", pts((62, 120, 2, 8)))
    t2 = sh.Part({"id": "t2"}, "tibia", pts((64, 120, 14, 20)))
    for f in (fa, fb):
        f.prox = int(np.argmin(f.ends[:, 0]))                  # proximal = left end
    fr = sh.Frame()
    links = sh.link([fa, fb], [t1, t2], sh.end_cost(fr), 1e9)
    nearest = [int(np.argmin([min(np.linalg.norm(f.end("dist") - e) for e in t.ends) for f in (fa, fb)])) for t in (t1, t2)]
    assert nearest == [0, 0]                                   # greedy would collide
    assert sorted(v[0] for v in links.values()) == [0, 1]   # Hungarian: one tibia per femur
    assert links[0][0] == 0 and links[1][0] == 1


def test_generic_name_matching():
    m = sh.part_matcher({"coxa": "coxa", "femur": "femur,femora", "tibia": "tibia", "tarsus": "tarsus"})
    assert [m(n) for n in ("femur", "Femur_1", "FEMORA 2", "tibia-3", " Tarsus ", "coxa")] == \
        ["femur", "femur", "femur", "tibia", "tarsus", "coxa"]
    assert [m(n) for n in ("profemur (left)", "femurlike", "tibial spur", "body")] == [None] * 4


def test_cli_new_file_input_untouched_unresolved_keep_generic(tmp_path):
    polys, _ = insect()
    coco, truth = to_coco([("ins.png", polys)])
    stray = {"id": 9999, "image_id": 1, "category_id": GENERIC["tarsus"], "iscrowd": 0,   # a tarsus far from any leg
             "segmentation": [[40, 40, 44, 36, 64, 56, 60, 60]]}
    coco["annotations"].append(stray)
    coco["images"].append(dict(coco["images"][0]))           # duplicate image entry (same id), as in real files
    src = tmp_path / "in.json"; src.write_text(json.dumps(coco)); before = src.read_text()
    out = tmp_path / "named.json"
    assert sh.main([str(src), "--out", str(out), "--view", "dorsal"]) == 0
    assert src.read_text() == before
    d = json.loads(out.read_text())
    names = {c["id"]: c["name"] for c in d["categories"]}
    assert all(names[a["category_id"]] == truth[a["id"]] for a in d["annotations"] if a["id"] in truth)
    kept = {a["id"]: a for a in coco["annotations"]}
    for a in d["annotations"]:                                 # only category ids of named instances change
        b = kept[a["id"]]
        assert {k: v for k, v in a.items() if k != "category_id"} == {k: v for k, v in b.items() if k != "category_id"}
        if a["id"] not in truth:
            assert a["category_id"] == b["category_id"]
    assert names[next(a for a in d["annotations"] if a["id"] == 9999)["category_id"]] == "tarsus"
    assert [u["annotation_id"] for u in d["info"]["descriptron_serial_homology"]["unresolved"]] == [9999]
    rows = list(csv.DictReader(open(tmp_path / "named_serial_homology.csv")))
    assert len(rows) == 19                                     # duplicate image entry processed once
    assert d["images"] == coco["images"]                       # images list (with its duplicate) kept as it was


# ------------------------------------------------------------------ trochanters
def test_trochanters_link_between_coxa_and_femur_in_ventral_view():
    polys, _ = insect(coxae=True, trochanters=True)
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "ventral")
    assert accuracy(rows, truth, swap_sides) == (30, 30)
    got = {truth[r["annotation_id"]]: r for r in rows}
    for side in ("left", "right"):
        for seg in sh.SEGMENTS:
            assert got[f"{seg}trochanter ({side})"]["linked_to"].startswith("coxa")
            assert got[f"{seg}femur ({side})"]["linked_to"].startswith("trochanter")
    assert got["mesotrochanter (right)"]["final_name"] == "mesotrochanter (left)"   # ventral: mirrored
    assert {r["anchor"].split()[0] for r in rows} == {"coxa"}
    for k in [k for k in polys if "coxa" in k]:                 # coxae hidden: the trochanter anchors the leg
        del polys[k]
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "ventral")
    assert accuracy(rows, truth, swap_sides) == (24, 24) and {r["anchor"].split()[0] for r in rows} == {"trochanter"}


def test_no_coxa_note_only_outside_dorsal_views():
    polys, _ = insect()                                         # femur, tibia, tarsus only
    coco, truth = to_coco([("ins.png", polys)])
    assert not any("coxa" in r["warnings"] for r in run(coco, "--view", "dorsal")[1])
    rows = run(coco, "--view", "ventral")[1]
    assert all("no coxa or trochanter found" in r["warnings"] for r in rows)
    assert accuracy(rows, truth, swap_sides) == (18, 18)


# ------------------------------------------------------------------ model first, rule second (--mode verify)
def verify(coco, *args):
    return run(coco, "--view", "dorsal", "--mode", "verify", *args)


def test_verify_keeps_correct_model_names_in_prefix_style():
    polys, _ = insect()
    names = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k)}
    coco, truth = to_coco([("ins.png", polys)], names)
    out, rows, _ = verify(coco, "--name_format", "prefix")
    assert all(r["status"] == "model" and r["final_name"] == prefix(truth[r["annotation_id"]]) for r in rows)
    assert out["categories"] == coco["categories"]              # nothing to add, nothing changed
    assert [a["category_id"] for a in out["annotations"]] == [a["category_id"] for a in coco["annotations"]]
    out, rows, _ = verify(coco)                                 # default style: written as 'profemur (left)'
    cat = {c["id"]: c["name"] for c in out["categories"]}
    assert all(cat[a["category_id"]] == truth[a["id"]] for a in out["annotations"] if a["id"] in truth)


def test_verify_fixes_a_duplicated_name():
    polys, _ = insect()
    names = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k)}
    names["metafemur (left)"] = "left_mesofemur"               # the model gives two femora one name
    coco, truth = to_coco([("ins.png", polys)], names)
    _, rows, _ = verify(coco, "--name_format", "prefix")
    got = {truth[r["annotation_id"]]: r for r in rows}
    assert got["metafemur (left)"]["final_name"] == "left_metafemur" and got["metafemur (left)"]["status"] == "rule"
    assert got["mesofemur (left)"]["final_name"] == "left_mesofemur" and got["mesofemur (left)"]["status"] == "model"
    assert all("(a)" in got[k]["reason"] for k in ("metafemur (left)", "mesofemur (left)"))
    assert accuracy(rows, truth, prefix) == (18, 18)


def test_verify_fixes_a_side_swapped_leg():
    polys, _ = insect()
    names = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k)}
    for p in ("femur", "tibia", "tarsus"):                      # both mesolegs given the other side's names
        names[f"meso{p} (left)"], names[f"meso{p} (right)"] = f"right_meso{p}", f"left_meso{p}"
    coco, truth = to_coco([("ins.png", polys)], names)
    _, rows, _ = verify(coco, "--name_format", "prefix")
    assert accuracy(rows, truth, prefix) == (18, 18)
    fixed = [r for r in rows if r["status"] == "rule"]
    assert len(fixed) == 6 and all("(b) side" in r["reason"] for r in fixed)


def test_verify_fixes_an_inconsistent_chain():
    polys, _ = insect()
    del polys["protarsus (right)"]                              # (so the wrong name below is not also a duplicate)
    names = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k)}
    names["protarsus (left)"] = "right_protarsus"              # its femur and tibia say left
    coco, truth = to_coco([("ins.png", polys)], names)
    _, rows, _ = verify(coco, "--name_format", "prefix", "--verify_margin", "5")   # (b) cannot fire: only (d)
    got = {truth[r["annotation_id"]]: r for r in rows}
    t = got["protarsus (left)"]
    assert t["final_name"] == "left_protarsus" and t["status"] == "rule" and "(d)" in t["reason"]
    assert "(b)" not in t["reason"]
    assert got["protibia (left)"]["status"] == "model" and "(d)" in got["protibia (left)"]["reason"]
    assert accuracy(rows, truth, prefix) == (17, 17)


def test_verify_side_less_names_and_low_scores():
    polys, _ = insect()
    names = {k: sh.parse_name(k)[1] + sh.parse_name(k)[2] for k in polys if sh.FULL_NAME_RE.match(k)}   # --fold_lr
    coco, truth = to_coco([("ins.png", polys)], names)
    _, rows, _ = verify(coco)
    assert accuracy(rows, truth) == (18, 18) and all("side from the rule" in r["reason"] for r in rows)
    names = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k)}
    names["profemur (left)"] = "left_mesofemur"                 # wrong and unsure; the real mesofemur is sure
    coco, truth = to_coco([("ins.png", polys)], names)
    for a in coco["annotations"]:
        a["score"] = 0.2 if truth.get(a["id"]) == "profemur (left)" else 0.9
    _, rows, _ = verify(coco, "--min_score", "0.5")
    got = {truth[r["annotation_id"]]: r for r in rows}
    assert got["profemur (left)"]["final_name"] == "profemur (left)" and "score 0.20" in got["profemur (left)"]["reason"]
    assert got["mesofemur (left)"]["status"] == "model"         # no duplicate once the unsure name is set aside


def test_prefix_input_and_output_in_rule_mode():
    polys, _ = insect()
    named = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k) and "(left)" in k}   # left legs named already
    coco, truth = to_coco([("ins.png", polys)], named)
    out, rows, _ = run(coco, "--view", "dorsal", "--name_format", "prefix")
    got = {truth[r["annotation_id"]]: r for r in rows}
    assert all(got[k]["status"] == "input" and got[k]["final_name"] == v for k, v in named.items())   # untouched
    assert all(got[k]["final_name"] == prefix(k) and got[k]["status"] == "rule" for k in truth.values() if "(right)" in k)
    cat = {c["id"]: c["name"] for c in out["categories"]}
    assert all(cat[a["category_id"]] == prefix(truth[a["id"]]) for a in out["annotations"] if a["id"] in truth)
    assert sh.parse_name("Left_Pro_Femur") == ("left", "pro", "femur") and sh.parse_name("mesotibia (right)") == \
        ("right", "meso", "tibia") and sh.parse_name("metatarsus") == (None, "meta", "tarsus")
    assert sh.parse_name("left_elytron") is None and sh.parse_name("femur") is None


def test_compressed_rle_masks_are_read():
    # DINOMask writes compressed RLE strings; this one was encoded by pycocotools.mask.encode (no pycocotools here)
    m = np.zeros((40, 30), bool); m[3:20, 4:9] = True; m[25:38, 10:28] = True; m[0:2, 0:30] = True
    counts = ("02V10000000kN?c0A]O?c0A]O?c0A]O?c0A]O?c0Ab00A;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;"
              "[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OEe0;[OET100")
    assert np.array_equal(sh.rle_mask({"size": [40, 30], "counts": counts}), m)
    assert sh.n_pieces(m) == 3

def test_verify_side_less_names_completed_from_the_lateral_near_side_without_a_rank():
    polys = lateral()
    for q in ("femur", "tibia", "tarsus"):                      # two legs and no abdomen landmark: no rank possible
        del polys[f"meso{q} (left)"]
    names = {k: sh.parse_name(k)[1] + sh.parse_name(k)[2] for k in polys if sh.FULL_NAME_RE.match(k)}
    coco, truth = to_coco([("ins.png", polys)], names)
    _, rows, _ = run(coco, "--view", "lateral", "--mode", "verify")
    assert all(r["rule_name"] == "" for r in rows)              # the rule cannot rank them ...
    assert accuracy(rows, truth) == (6, 6)                      # ... but knows the near side


def test_verify_inconsistent_chain_without_a_rule_name_keeps_the_model_name():
    polys, _ = insect()
    for side in ("left", "right"):                              # two legs per side, no landmarks: no rule names
        for q in ("femur", "tibia", "tarsus"):
            del polys[f"meso{q} ({side})"]
    names = {k: prefix(k) for k in polys if sh.FULL_NAME_RE.match(k)}
    names["protarsus (left)"] = "left_mesotarsus"               # wrong, but no other part has that name
    coco, truth = to_coco([("ins.png", polys)], names)
    _, rows, _ = verify(coco, "--name_format", "prefix")
    got = {truth[r["annotation_id"]]: r for r in rows}
    t = got["protarsus (left)"]
    assert t["status"] == "model" and t["final_name"] == "left_mesotarsus" and "(d)" in t["reason"]
    assert "model name kept" in t["reason"]
    assert got["profemur (left)"]["final_name"] == "left_profemur"


# ------------------------------------------------------------------ lateral (profile) views
def lateral(far_side=False, dorsal_masks=True):
    """Profile of a beetle, dorsum up, head on the image's LEFT; three near-side legs hang below (their bases inside
    the body outline, in front of the body); optionally a far-side mesoleg emerging from behind the body outline."""
    cx, cy = 400.0, 260.0
    bottom = lambda x: cy + 70 * math.sqrt(max(0.0, 1 - ((x - cx) / 220) ** 2))
    polys = {"body": [ellipse(cx, cy, 220, 70)], "head": [ellipse(160, 255, 45, 42)]}
    if dorsal_masks:
        polys["pronotum"] = [ellipse(255, 222, 55, 36)]; polys["elytra"] = [ellipse(455, 222, 150, 40)]
    down = lambda deg: np.array([math.sin(math.radians(deg)), math.cos(math.radians(deg))])   # + = backwards (+x)
    legs = [("pro", 265, -25, -10, -75, "left", 20), ("meso", 345, 0, 8, 70, "left", 20),
            ("meta", 425, 25, 20, 75, "left", 20)]
    if far_side:
        legs.append(("meso", 362, 15, 25, 80, "right", -2))    # base just below the outline: behind the body
    for seg, x, fa, ta, sa, side, inside in legs:
        base = np.array([x, bottom(x) - inside])
        knee = base + 70 * down(fa); ank = knee + 80 * down(ta); tip = ank + 45 * down(sa)
        polys[f"{seg}femur ({side})"] = [strip(base, knee, 16, 13)]
        polys[f"{seg}tibia ({side})"] = [strip(knee, ank, 10, 8)]
        polys[f"{seg}tarsus ({side})"] = [strip(ank, tip, 6, 4)]
    return polys


def test_lateral_head_left_is_left_side_head_right_is_right_and_order_head_to_tail():
    polys = lateral()
    coco, truth = to_coco([("ins.png", polys)])
    _, rows, _ = run(coco, "--view", "lateral")
    assert accuracy(rows, truth) == (9, 9)                      # pro, meso, meta from the head backwards, left side
    assert all(r["side_margin_px"] == "" for r in rows)        # no midline step in a profile
    mirrored = transform(polys, lambda x, y: (800 - x, y))     # head on the image's right: the right side faces us
    coco, truth = to_coco([("ins.png", mirrored)])
    assert accuracy(run(coco, "--view", "lateral")[1], truth, swap_sides) == (9, 9)


def test_lateral_dorsum_down_inverts_and_auto_finds_it():
    upside_down = transform(lateral(), lambda x, y: (x, 600 - y))   # head still left, back now down
    coco, truth = to_coco([("ins.png", upside_down)])
    assert accuracy(run(coco, "--view", "lateral", "--dorsum", "down")[1], truth, swap_sides) == (9, 9)
    assert accuracy(run(coco, "--view", "lateral", "--dorsum", "auto")[1], truth, swap_sides) == (9, 9)
    assert accuracy(run(coco, "--view", "lateral")[1], truth)[0] == 9   # default --dorsum up: wrong input, wrong side
    no_masks = transform(lateral(dorsal_masks=False), lambda x, y: (x, 600 - y))
    coco, truth = to_coco([("ins.png", no_masks)])                 # auto without dorsal masks: legs are ventral
    _, rows, _ = run(coco, "--view", "lateral", "--dorsum", "auto")
    assert accuracy(rows, truth, swap_sides) == (9, 9) and "legs on the ventral side" in rows[0]["anchor"]


def test_lateral_rotated_180_keeps_the_side_with_dorsum_auto():
    coco, truth = to_coco([("ins.png", rotate(lateral(), 180))])
    assert accuracy(run(coco, "--view", "lateral", "--dorsum", "auto")[1], truth) == (9, 9)


def test_lateral_far_side_leg_flagged_not_assigned():
    coco, truth = to_coco([("ins.png", lateral(far_side=True))])
    far = {a for a, n in truth.items() if n.endswith("(right)")}
    _, rows, _ = run(coco, "--view", "lateral", "--lateral_far_side", "flag")
    got = {r["annotation_id"]: r for r in rows}
    assert all(got[a]["final_name"] == "" and got[a]["side"] == "far side?" and "far side?" in got[a]["warnings"]
               for a in far)
    assert "possibly the far-side mesofemur" in got[next(a for a in far if "femur" in truth[a])]["warnings"]
    near = {a: n for a, n in truth.items() if a not in far}
    assert accuracy(rows, near) == (9, 9)
    # default (ignore): every leg is near side; four legs cannot be ranked without landmarks -> nothing guessed
    _, rows, _ = run(coco, "--view", "lateral")
    assert all(r["final_name"] == "" for r in rows) and "--lateral_far_side flag" in rows[0]["warnings"]


def test_lateral_view_from_file_name():
    coco, truth = to_coco([("sp1_lateral.png", lateral()), ("sp1_dorsal.png", insect()[0])])
    _, rows, _ = run(coco, "--view_regex", r"_(dorsal|ventral|lateral)\b")
    assert accuracy(rows, truth) == (27, 27)


# ------------------------------------------------------------------ hand-labelled Oxytelus (dorsal)
def oxytelus_generic():
    coco = json.loads(OXY.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    nid = max(cats) + 1
    gid = {p: nid + i for i, p in enumerate(sh.PARTS)}
    coco["categories"] += [{"id": v, "name": k, "supercategory": ""} for k, v in gid.items()]
    truth = {}
    for a in coco["annotations"]:
        m = sh.FULL_NAME_RE.match(cats[a["category_id"]])
        if m:
            truth[a["id"]] = cats[a["category_id"]]; a["category_id"] = gid[m.group(2)]
    return coco, truth


def xform(coco, f, size=None):
    d = copy.deepcopy(coco)
    for a in d["annotations"]:
        if isinstance(a.get("segmentation"), list):
            a["segmentation"] = [[v for x, y in zip(p[0::2], p[1::2]) for v in f(x, y)] for p in a["segmentation"]]
    if size:
        for im in d["images"]:
            im["width"], im["height"] = size
    return d


@pytest.mark.skipif(not OXY.exists(), reason="Oxytelus hand-labelled file not on this machine")
def test_oxytelus_round_trip_mirrored_rotated():
    coco, truth = oxytelus_generic()
    W, H = coco["images"][0]["width"], coco["images"][0]["height"]
    with_geometry = {a["id"] for a in coco["annotations"] if a["id"] in truth and sh.ann_polygons(a)}
    assert len(truth) == 19 and len(with_geometry) == 18       # 'mesocoxa (right)' has no polygon
    cases = [("original", coco, "dorsal", lambda n: n),
             ("mirror dorsal", xform(coco, lambda x, y: (W - x, y)), "dorsal", swap_sides),
             ("mirror ventral", xform(coco, lambda x, y: (W - x, y)), "ventral", lambda n: n),
             ("rot90", xform(coco, lambda x, y: (H - y, x), (H, W)), "dorsal", lambda n: n)]
    for label, c, view, expect in cases:
        _, rows, notes = run(c, "--view", view)
        assert len(rows) == 19, label                          # image listed twice -> processed once
        got = {r["annotation_id"]: r["final_name"] for r in rows}
        assert sum(got[a] == expect(truth[a]) for a in with_geometry) == 18, label
        empty = next(a for a in truth if a not in with_geometry)
        assert got[empty] == "", label                         # no geometry -> left generic, not guessed


@pytest.mark.skipif(not OXY.exists(), reason="Oxytelus hand-labelled file not on this machine")
def test_oxytelus_check_existing(tmp_path):
    out = tmp_path / "check.json"
    sh.main([str(OXY), "--out", str(out), "--view", "dorsal", "--check_existing"])
    rows = list(csv.DictReader(open(tmp_path / "check_serial_homology.csv")))
    assert sum(r["agrees"] == "True" for r in rows) == 18 and len(rows) == 19


@pytest.mark.skipif(not OXY.exists(), reason="Oxytelus hand-labelled file not on this machine")
def test_oxytelus_verify_mode_with_the_hand_labels_as_model():
    coco = json.loads(OXY.read_text())
    truth = {}
    for c in coco["categories"]:                                # hand labels written the DINOMask way
        if sh.FULL_NAME_RE.match(c["name"]):
            c["name"] = prefix(c["name"])
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    truth = {a["id"]: cat[a["category_id"]] for a in coco["annotations"] if sh.parse_name(cat[a["category_id"]])}
    _, rows, _ = run(coco, "--view", "dorsal", "--mode", "verify", "--name_format", "prefix")
    assert len(rows) == 19 and all(r["status"] == "model" and r["final_name"] == truth[r["annotation_id"]] for r in rows)
    # corrupt it as a model might: swap the sides of both mesolegs, give the left metatibia the mesotibia's name
    bad = copy.deepcopy(coco)
    name_id = {c["name"]: c["id"] for c in bad["categories"]}
    swap = {f"{s}_meso{p}": f"{o}_meso{p}" for s, o in (("left", "right"), ("right", "left")) for p in ("femur", "tibia", "tarsus")}
    for a in bad["annotations"]:
        n = cat[a["category_id"]]
        if n in swap:
            a["category_id"] = name_id[swap[n]]
        elif n == "left_metatibia":
            a["category_id"] = name_id["left_mesotibia"]
    _, rows, _ = run(bad, "--view", "dorsal", "--mode", "verify", "--name_format", "prefix")
    got = {r["annotation_id"]: r for r in rows}
    with_geometry = [a for a in truth if got[a]["rule_name"] or "no mask geometry" not in got[a]["warnings"]]
    assert sum(got[a]["final_name"] == truth[a] for a in with_geometry) == 18
