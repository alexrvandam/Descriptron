#!/usr/bin/env python3
"""
descriptron_check_cross_image_copies_v1.py - find annotations copied onto images they do not belong to
=======================================================================================================

Before GUI v81, "Load Annotations" drew EVERY annotation of a COCO file on the image on screen and saved
them under that image. A file that was loaded while another image was showing can therefore carry the
same outline twice: once under its own image, once under an image it was never drawn on. A copy keeps
the exact pixel position AND the outline points of the original, so it is found by comparing annotations
filed under DIFFERENT images: a pair that overlaps almost completely (IoU >= --min_iou, default 0.95) and
shares most of its outline vertices exactly (>= --min_shared, default 0.9) is a copy. Two people tracing
similar structures on aligned images can overlap at 0.95 but share only a few percent of their points (1-5%
on aligned ant heads), so overlap alone is not enough. Where either mask is RLE there are no vertices; such
a pair is reported on overlap alone and marked "check". Pairs that share 20-90% of their points are listed
as "partly shared" (an outline copied and then edited, e.g. a second photo of the same specimen).

The program cannot tell which of a pair is the original (both may since have been re-labelled), so it
reports both and changes nothing. To write a cleaned copy, name what to drop: --drop_ids <annotation ids>
and/or --drop_image <image file name> (every annotation filed under that image). The input file is
never modified; the cleaned file is written beside it as <name>_cleaned.json unless --output is given.

  python descriptron_check_cross_image_copies_v1.py --coco annotations.json
  python descriptron_check_cross_image_copies_v1.py --coco annotations.json --drop_image MJ_31308000000.tif

Writes <name>_cross_image_copies.tsv beside the input (or --report) and prints a summary.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

STEP = 4                                   # masks compared at 1/4 resolution


def _image_name(im):
    return os.path.basename(str(im.get("file_name") or im.get("id")))


def _mask(seg, shape, step=STEP):
    """annotation mask at 1/step resolution on a canvas shared by both images (same pixel coordinates)"""
    import cv2
    H, W = shape
    m = np.zeros((H // step + 1, W // step + 1), np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            p = (np.asarray(poly, float).reshape(-1, 2) / step).round().astype(np.int32)
            if len(p) >= 3:
                cv2.fillPoly(m, [p], 1)
    elif isinstance(seg, dict) and "counts" in seg:
        h, w = seg["size"]
        counts = seg["counts"]
        if isinstance(counts, str):
            from pycocotools import mask as mu
            full = mu.decode(seg)
        else:
            flat = np.zeros(h * w, np.uint8); i = 0; v = 0
            for r in counts:
                flat[i:i + r] = v; i += r; v = 1 - v
            full = flat.reshape((w, h)).T
        small = full[::step, ::step]
        m[:small.shape[0], :small.shape[1]] = small[:m.shape[0], :m.shape[1]]
    return m.astype(bool)


def _bbox(a):
    b = a.get("bbox")
    if b and len(b) == 4:
        return [float(x) for x in b]
    seg = a.get("segmentation")
    if isinstance(seg, list) and seg:
        p = np.concatenate([np.asarray(s, float).reshape(-1, 2) for s in seg if len(s) >= 6])
        return [p[:, 0].min(), p[:, 1].min(), np.ptp(p[:, 0]), np.ptp(p[:, 1])]
    return None


def _verts(a):
    s = a.get("segmentation")
    if not isinstance(s, list):
        return None
    v = set()
    for poly in s:
        q = np.round(np.asarray(poly, float).reshape(-1, 2)).astype(int)
        v.update(map(tuple, q.tolist()))
    return v


def shared_vertices(a, b):
    """fraction of the smaller outline's points found exactly in the other; None if either is RLE"""
    va, vb = _verts(a), _verts(b)
    if not va or not vb:
        return None
    return len(va & vb) / min(len(va), len(vb))


def find_copies(d, min_iou=0.95, corner_tol=0.05, min_shared=0.9, partly=0.2):
    """pairs (a, b, iou) of mask annotations under different images whose masks coincide"""
    imgs = {im["id"]: im for im in d.get("images", [])}
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    anns = [a for a in d.get("annotations", []) if a.get("segmentation") and _bbox(a)]
    out = []
    cache = {}
    for i in range(len(anns)):
        a = anns[i]; ba = _bbox(a)
        for j in range(i + 1, len(anns)):
            b = anns[j]
            if a.get("image_id") == b.get("image_id"):
                continue
            bb = _bbox(b)
            tol = corner_tol * max(ba[2], ba[3], bb[2], bb[3], 1.0)
            if abs(ba[0] - bb[0]) > tol or abs(ba[1] - bb[1]) > tol:       # copies sit at the same corner
                continue
            H = int(max(ba[1] + ba[3], bb[1] + bb[3])) + 2
            W = int(max(ba[0] + ba[2], bb[0] + bb[2])) + 2
            for k, x in ((i, a), (j, b)):
                if (k, H, W) not in cache:
                    cache[(k, H, W)] = _mask(x["segmentation"], (H, W))
            ma, mb = cache[(i, H, W)], cache[(j, H, W)]
            inter = np.logical_and(ma, mb).sum(); union = np.logical_or(ma, mb).sum()
            iou = inter / union if union else 0.0
            if iou >= min_iou:
                sh = shared_vertices(a, b)
                if sh is None:
                    verdict = "check (RLE: overlap only)"
                elif sh >= min_shared:
                    verdict = "copy"
                elif sh >= partly:
                    verdict = "partly shared"
                else:
                    continue                                  # similar shapes traced independently
                out.append({"verdict": verdict, "iou": round(float(iou), 4),
                            "shared_vertices": "" if sh is None else round(sh, 3),
                            "annotation_id_1": a.get("id"), "image_1": _image_name(imgs.get(a.get("image_id"), {"id": a.get("image_id")})),
                            "category_1": cats.get(a.get("category_id"), a.get("category_id")),
                            "annotation_id_2": b.get("id"), "image_2": _image_name(imgs.get(b.get("image_id"), {"id": b.get("image_id")})),
                            "category_2": cats.get(b.get("category_id"), b.get("category_id"))})
    return out


def drop(d, ids=(), image=None):
    """a copy of the COCO dict without the named annotations / every annotation of the named image"""
    ids = {str(x) for x in ids}
    gone_imgs = set()
    if image:
        want = os.path.splitext(os.path.basename(image))[0].lower()
        gone_imgs = {im["id"] for im in d.get("images", [])
                     if os.path.splitext(_image_name(im))[0].lower() == want
                     or os.path.basename(str(im["id"])).lower() == want}
        if not gone_imgs:
            sys.exit(f"--drop_image {image}: no such image in the file")
    out = dict(d)
    out["annotations"] = [a for a in d.get("annotations", [])
                          if str(a.get("id")) not in ids and a.get("image_id") not in gone_imgs]
    out["images"] = [im for im in d.get("images", []) if im["id"] not in gone_imgs]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="COCO .json or .rlejson")
    ap.add_argument("--min_iou", type=float, default=0.95)
    ap.add_argument("--min_shared", type=float, default=0.9, help="share of identical outline points for a copy")
    ap.add_argument("--report", default=None, help="TSV of the pairs found (default: beside the input)")
    ap.add_argument("--drop_ids", nargs="*", default=[], help="annotation ids to leave out of the cleaned copy")
    ap.add_argument("--drop_image", default=None, help="leave out this image and everything filed under it")
    ap.add_argument("--output", default=None, help="cleaned copy (default: <name>_cleaned<ext> beside the input)")
    a = ap.parse_args()
    src = Path(a.coco)
    d = json.load(open(src))
    pairs = find_copies(d, a.min_iou, min_shared=a.min_shared)
    rep = Path(a.report) if a.report else src.with_name(src.stem + "_cross_image_copies.tsv")
    with open(rep, "w", newline="") as f:
        cols = ["verdict", "iou", "shared_vertices", "annotation_id_1", "image_1", "category_1", "annotation_id_2", "image_2", "category_2"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(pairs)
    n_img = len(d.get("images", [])); n_ann = len(d.get("annotations", []))
    n_copy = sum(p["verdict"] == "copy" for p in pairs)
    print(f"{src.name}: {n_img} images, {n_ann} annotations; {n_copy} cross-image cop"
          f"{'y' if n_copy == 1 else 'ies'}, {len(pairs) - n_copy} other pair(s) to look at")
    for p in pairs:
        sv = "" if p["shared_vertices"] == "" else f", {100 * p['shared_vertices']:.0f}% points shared"
        print(f"  {p['verdict']:24s} IoU {p['iou']:.3f}{sv}  {p['image_1']} [{p['category_1']}, id {p['annotation_id_1']}]"
              f"  ==  {p['image_2']} [{p['category_2']}, id {p['annotation_id_2']}]")
    print(f"report -> {rep}")
    if a.drop_ids or a.drop_image:
        out = Path(a.output) if a.output else src.with_name(src.stem + "_cleaned" + src.suffix)
        if out.resolve() == src.resolve():
            sys.exit("--output must not be the input file")
        cleaned = drop(d, a.drop_ids, a.drop_image)
        tmp = out.with_suffix(out.suffix + ".tmp")
        json.dump(cleaned, open(tmp, "w")); os.replace(tmp, out)
        print(f"cleaned copy: {len(d['annotations'])} -> {len(cleaned['annotations'])} annotations -> {out}")
    elif n_copy:
        print("nothing changed; to write a cleaned copy add --drop_ids <id ...> or --drop_image <file name>")


if __name__ == "__main__":
    main()
