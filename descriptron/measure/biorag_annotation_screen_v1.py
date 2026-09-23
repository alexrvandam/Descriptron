#!/usr/bin/env python3
"""
biorag_annotation_screen_v1.py — does each polygon actually sit on the specimen?
================================================================================

A misplaced annotation is worse than a missing one: it produces a number, the number
enters the matrix, and nothing downstream can tell it from a measurement. The check is
the one a person makes instantly by eye and that no statistic downstream can make —
look at the mask on the image and see whether it encloses specimen or empty slide.

On transmitted-light slide mounts the specimen is darker than the mount, so the test is
simply what fraction of each polygon covers material rather than background. A correctly
placed outline is almost entirely on the specimen; one that has drifted sits on blank
slide and is caught whatever its shape. Two further checks come free:

  coverage    share of the polygon lying on specimen rather than background
  containment a sub-structure (a wing cell) must lie inside its parent (the wing);
              one that does not is misplaced even if it happens to land on material
  dimensions  COCO width/height that disagree with the file on disk, which silently
              rescales every coordinate in that image

Nothing is deleted. The screen writes an exclusion list in the same form as the manual
one, so the two can be merged and a person can overrule either.

  python biorag_annotation_screen_v1.py --coco <coco.json> --image_dir <dir> \\
      --out_dir "$M/annotation_screen" [--min_coverage 0.35] [--parent whole_wing]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

VERSION = "1.1"
Image.MAX_IMAGE_PIXELS = None

# 1.1 (2026-09-21): low coverage alone no longer excludes. On cleared, slide-mounted material a
# correctly drawn polygon around a transparent wing cell, a pale tibia or an aedeagus is NOT darker
# than the mount, and version 1.0 set 36 such annotations aside on Diaphorina (69% of its flags).
# What does show a misplaced annotation is that the polygons of an image fit the specimen much
# better after one of the eight rigid moves of the frame (the image was rotated or mirrored after
# it was annotated): that test is now what flags, together with the containment check. Low coverage
# is still reported, as something to look at. --exclude_low_coverage restores the 1.0 behaviour for
# opaque material on a plain ground, where coverage is informative.
try:
    from biorag_annotation_reorient_v1 import TRANSFORMS, move, coverage as _moved_coverage
except Exception:                                                         # pragma: no cover
    TRANSFORMS = None


def foreground(im: np.ndarray) -> np.ndarray:
    """Specimen against a pale mount: darker than the bright mode of the image."""
    g = im.mean(2)
    return (g < np.percentile(g, 60) * 0.92).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Flag annotations that do not sit on the specimen")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--merge_with", default=None,
                    help="manual exclusion CSV (image_filename, category_name, ...): written back "
                         "out together with the flagged annotations as exclusion_list_merged.csv")
    ap.add_argument("--min_coverage", type=float, default=0.35)
    ap.add_argument("--parent", default=None,
                    help="category every other annotation on the same image should fall inside "
                         "(e.g. whole_wing); omit to skip the containment check")
    ap.add_argument("--reduce", type=int, default=2, help="downscale factor while testing")
    ap.add_argument("--exclude_low_coverage", action="store_true",
                    help="also set aside every annotation below --min_coverage (version 1.0 behaviour; "
                         "right for opaque specimens on a plain ground, wrong for cleared or pale structures)")
    ap.add_argument("--turn_gain", type=float, default=0.18,
                    help="an image counts as turned after annotation when a rigid move of its polygons "
                         "raises their mean coverage by at least this much ...")
    ap.add_argument("--turn_good", type=float, default=0.58, help="... to at least this ...")
    ap.add_argument("--turn_margin", type=float, default=0.05, help="... and beats the next best move by this")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = json.loads(Path(a.coco).read_text())
    cats = {c["id"]: c["name"] for c in j.get("categories", [])}
    imgs = {i["id"]: i for i in j.get("images", [])}
    by_img = {}
    for ann in j.get("annotations", []):
        by_img.setdefault(ann["image_id"], []).append(ann)

    rows, dim_bad, turned = [], [], {}
    for n, (iid, anns) in enumerate(sorted(by_img.items()), 1):
        rec = imgs.get(iid)
        if not rec:
            continue
        p = Path(a.image_dir) / rec["file_name"]
        if not p.exists():
            continue
        with Image.open(p) as raw:
            w, h = raw.size
            if (rec.get("width") or 0, rec.get("height") or 0) != (w, h):
                dim_bad.append({"image": rec["file_name"],
                                "coco": f"{rec.get('width')}x{rec.get('height')}",
                                "actual": f"{w}x{h}"})
            im = np.asarray(raw.convert("RGB").reduce(a.reduce)).astype(np.float32)
        fg = foreground(im)
        if TRANSFORMS is not None:
            polys = [ann["segmentation"][0] for ann in anns
                     if isinstance(ann.get("segmentation"), list) and ann["segmentation"] and ann["segmentation"][0]]
            if polys:
                sc = {}
                for name, _q, fn in TRANSFORMS:
                    cv_ = [_moved_coverage(fg, move(pp, fn, w, h), a.reduce) for pp in polys]
                    sc[name] = float(np.nanmean(cv_)) if np.isfinite(cv_).any() else np.nan
                best = max(sc, key=lambda k: -1 if np.isnan(sc[k]) else sc[k])
                rest = sorted((v for k, v in sc.items() if k != best and not np.isnan(v)), reverse=True)
                if (best != "as is" and sc[best] >= a.turn_good and sc[best] - sc["as is"] >= a.turn_gain
                        and (not rest or sc[best] - rest[0] >= a.turn_margin)):
                    turned[rec["file_name"]] = (best, round(sc["as is"], 3), round(sc[best], 3))
        masks = {}
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or not isinstance(seg, list) or not seg[0]:
                continue
            pts = (np.asarray(seg[0], float).reshape(-1, 2) / a.reduce).astype(np.int32)
            m = np.zeros(fg.shape, np.uint8)
            cv2.fillPoly(m, [pts], 1)
            if m.sum() < 25:
                continue
            masks[ann["id"]] = (cats.get(ann["category_id"], "?"), m)
            rows.append({"image": rec["file_name"], "annotation_id": ann["id"],
                         "category": cats.get(ann["category_id"], "?"),
                         "coverage": round(float((fg & m).sum() / m.sum()), 4),
                         "area_px": int(m.sum() * a.reduce ** 2)})
        if a.parent:
            par = [m for c, m in masks.values() if c == a.parent]
            if par:
                P = par[0]
                for r in rows[-len(masks):]:
                    mm = next((m for i2, (c, m) in masks.items()
                               if i2 == r["annotation_id"]), None)
                    if mm is not None and r["category"] != a.parent:
                        r["inside_parent"] = round(float((mm & P).sum() / mm.sum()), 4)
        if n % 60 == 0:
            print(f"  {n}/{len(by_img)} images")

    d = pd.DataFrame(rows)
    d["fails_coverage"] = d["coverage"] < a.min_coverage
    if "inside_parent" in d:
        d["fails_containment"] = d["inside_parent"].fillna(1.0) < 0.5
    else:
        d["fails_containment"] = False
    d["image_turned_after_annotation"] = d["image"].isin(turned)
    d["fits_after"] = d["image"].map(lambda n: turned.get(n, ("",))[0])
    if a.exclude_low_coverage or TRANSFORMS is None:
        d["flag"] = d["fails_coverage"] | d["fails_containment"] | d["image_turned_after_annotation"]
    else:
        d["flag"] = d["fails_containment"] | d["image_turned_after_annotation"]
    d["look_at"] = d["fails_coverage"] & ~d["flag"]        # pale / transparent, or misplaced: a person decides
    d.to_csv(out / "annotation_screen.tsv", sep="\t", index=False)
    pd.DataFrame(dim_bad).to_csv(out / "dimension_mismatches.tsv", sep="\t", index=False)

    bad = d[d["flag"]]
    excl = (bad.groupby("image")
            .agg(n_flagged=("annotation_id", "size"),
                 categories=("category", lambda s: "; ".join(sorted(set(s)))),
                 worst_coverage=("coverage", "min")).reset_index())
    excl["reason"] = ["image rotated or mirrored after annotation (polygons fit after: "
                      f"{turned[i][0]}); repair with biorag_annotation_reorient_v1.py" if i in turned
                      else "annotation does not sit on the specimen (automatic screen)" for i in excl["image"]]
    excl.to_csv(out / "auto_exclusion_list.csv", index=False)

    # The same flags, one row per annotation, in the format biorag_key_feature_filter_v2.py
    # reads (image_filename, category_name, annotation_id, reason) and merged with the manual
    # list when one is given. The per-image summary above is for a person to read; it cannot
    # be passed to the matrix builder, and a list that has to be converted by hand is a list
    # that does not get applied.
    flt = pd.DataFrame({"image_filename": bad["image"], "category_name": bad["category"],
                        "annotation_id": bad["annotation_id"],
                        "reason": [(f"image rotated or mirrored after annotation (fits after: {turned[i][0]}) - "
                                    "repair with biorag_annotation_reorient_v1.py" if i in turned else
                                    "annotation does not sit on the specimen - automatic screen, "
                                    f"coverage {c:.3f}") for i, c in zip(bad["image"], bad["coverage"])]})
    n_manual = 0
    if a.merge_with and Path(a.merge_with).exists():
        manual = pd.read_csv(a.merge_with)
        n_manual = len(manual)
        have = set(zip(manual["image_filename"], manual["category_name"]))
        flt = pd.concat([manual, flt[[k not in have for k in
                                      zip(flt["image_filename"], flt["category_name"])]]],
                        ignore_index=True)
    flt.to_csv(out / "exclusion_list_merged.csv", index=False)

    summary = {"version": VERSION, "annotations_tested": int(len(d)),
               "images": int(d["image"].nunique()),
               "median_coverage": float(d["coverage"].median()),
               "flagged": int(len(bad)),
               "images_turned_after_annotation": {k: {"fits_after": v[0], "coverage_as_is": v[1],
                                                      "coverage_after": v[2]} for k, v in turned.items()},
               "low_coverage_not_excluded": int(d["look_at"].sum()),
               "low_coverage_excludes": bool(a.exclude_low_coverage),
               "flagged_percent": round(100 * len(bad) / max(1, len(d)), 2),
               "images_with_a_flag": int(bad["image"].nunique()) if len(bad) else 0,
               "dimension_mismatches": len(dim_bad),
               "min_coverage": a.min_coverage,
               "manual_list_entries": n_manual,
               "exclusion_list_merged_entries": int(len(flt)),
               "note": "nothing deleted; pass exclusion_list_merged.csv to the matrix builder "
                       "(--exclude_list) to leave the flagged annotations out"}
    (out / "annotation_screen_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    if turned:
        print(f"\n{len(turned)} image(s) were rotated or mirrored AFTER they were annotated. Their polygons are set "
              "aside here; to repair them instead run biorag_annotation_reorient_v1.py on the same COCO file.")
    if int(d["look_at"].sum()):
        print(f"{int(d['look_at'].sum())} annotation(s) cover little that is darker than the mount and were NOT set "
              "aside: on cleared material that is usually a pale structure drawn correctly (column look_at).")
    if len(excl):
        print("\nimages with flagged annotations:")
        print(excl.sort_values("n_flagged", ascending=False).head(20).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
