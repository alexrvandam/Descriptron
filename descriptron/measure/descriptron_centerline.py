#!/usr/bin/env python3
"""
descriptron_centerline.py — curved length and shape of thin structures from their masks
=========================================================================================

For veins, antennae, legs, setae, genitalic processes and other elongate structures the length along
the structure matters, and a straight measurement (bounding box or principal axis, as used for the
blob-like structures in the measurement script) under-reads anything that curves. This program adds
centre-line measurements; it does not replace or change the existing measurements.

Method (per mask):
  1. clean the mask (largest piece, holes filled);
  2. thin it to a one-pixel skeleton and treat that as a graph;
  3. MAIN-PATH mode (default): keep the longest path through the skeleton — side branches ignored;
     BRANCH mode (--branches): keep the whole branch graph, pruning spurs shorter than --min-branch;
  4. extend each free end outwards to the mask's own tip (a plain skeleton stops about half the
     structure's width short of each end);
  5. smooth the path (Gaussian, sigma --smooth px) so the pixel staircase does not inflate the length,
     then measure.

Measurements (pixels, and mm when a scale is known): curved length, chord (straight end-to-end)
length, sinuosity (curved / chord), mean / max / min width along the path, mean and max curvature.
In branch mode: number of branch points and tips, total length and each segment's length.

Output: a CSV (one row per mask; per-segment rows in branch mode), a COCO file with the centre lines as
line annotations (open it in the GUI to check them), and optional overlay PNGs.

Scale (px per mm), first found wins: images[].scale_px_per_mm in the COCO; --scales CSV with an image
column and a pixels-per-unit column (the measurement script's combined_scales.csv works); --px-per-mm.

    python descriptron_centerline.py ann.json --categories vein_R vein_M antenna --out-dir centerlines/
    python descriptron_centerline.py ann.json --images photos/ --scales combined_scales.csv --overlays --out-dir cl/
    python descriptron_centerline.py ann.json --branches --min-branch 15 --out-dir cl_branches/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

# ------------------------------------------------------------------ masks
def polygon_mask(seg, h: int, w: int) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.uint8)
    if isinstance(seg, dict):                                   # RLE
        from pycocotools import mask as mu
        rle = seg if isinstance(seg.get("counts"), (str, bytes)) else mu.frPyObjects(seg, h, w)
        return mu.decode(rle).astype(bool)
    if seg and not isinstance(seg[0], list):
        seg = [seg]
    for p in seg or []:
        if len(p) >= 6:
            cv2.fillPoly(m, [np.round(np.array(p, float).reshape(-1, 2)).astype(np.int32)], 1)
    return m.astype(bool)


def clean(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(keep)


# ------------------------------------------------------------------ skeleton graph
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_graph(sk: np.ndarray):
    ys, xs = np.nonzero(sk)
    idx = -np.ones(sk.shape, np.int64)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, wts = [], [], []
    H, W = sk.shape
    for dy, dx in _NB:
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        j = np.full(len(ys), -1); j[ok] = idx[y2[ok], x2[ok]]
        m = j >= 0
        rows.append(np.nonzero(m)[0]); cols.append(j[m]); wts.append(np.full(m.sum(), math.hypot(dy, dx)))
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    g = coo_matrix((w, (r, c)), shape=(len(ys), len(ys))).tocsr()
    deg = np.diff(g.indptr)
    return g, np.stack([xs, ys], 1).astype(float), deg


def longest_path(g, pts, deg) -> np.ndarray:
    """Double sweep from the tips: exact for trees, which the skeleton of a filled mask is."""
    tips = np.nonzero(deg == 1)[0]
    start = tips[0] if len(tips) else 0
    d0 = dijkstra(g, indices=start)
    a = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(g, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    return pts[path[::-1]]


def extend_to_tip(path: np.ndarray, mask: np.ndarray, k: int = 8) -> np.ndarray:
    """March from each end along its local direction until the mask ends."""
    if len(path) < 3:
        return path
    H, W = mask.shape
    out = [path]
    for end in (0, 1):
        seg = path[:k] if end == 0 else path[-k:][::-1]
        d = seg[0] - seg[-1]
        n = np.hypot(*d)
        if n == 0:
            continue
        d = d / n
        p, step, new = seg[0].copy(), 0.5, []
        for _ in range(int(2 * max(H, W))):
            q = p + d * step
            x, y = int(round(q[0])), int(round(q[1]))
            if not (0 <= x < W and 0 <= y < H) or not mask[y, x]:
                break
            p = q
            new.append(p.copy())
        if new:
            new = np.array(new[::2] + [new[-1]])
            out = [new[::-1], *out] if end == 0 else [*out, new]
    return np.vstack(out)


def smooth_resample(path: np.ndarray, sigma: float) -> np.ndarray:
    if len(path) < 3:
        return path
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return path[:1]
    t = np.arange(0, s[-1], 0.5)
    xy = np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])], 1)
    if sigma > 0:
        sm = ndi.gaussian_filter1d(xy, sigma / 0.5, axis=0, mode="nearest")
        sm[0], sm[-1] = xy[0], xy[-1]                      # keep the tips where they are
        xy = sm
    return np.vstack([xy, path[-1:]])


def path_metrics(xy: np.ndarray, dist: np.ndarray) -> Dict[str, float]:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    length = float(seg.sum())
    chord = float(np.hypot(*(xy[-1] - xy[0])))
    H, W = dist.shape
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)
    width = 2 * dist[yi, xi]
    core = width[len(width) // 10: max(len(width) - len(width) // 10, len(width) // 10 + 1)]  # tips excluded
    step = max(1, int(round(4 / max(np.median(seg[seg > 0]) if (seg > 0).any() else 1, 1e-6))))
    pts = xy[::step]
    curv = []
    for i in range(1, len(pts) - 1):
        a, b = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
        la, lb = np.hypot(*a), np.hypot(*b)
        if la > 0 and lb > 0:
            ang = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
            curv.append(abs(ang) / ((la + lb) / 2))
    return {"length_px": length, "chord_px": chord, "sinuosity": length / chord if chord > 0 else float("nan"),
            "width_mean_px": float(core.mean()) if core.size else float("nan"),
            "width_max_px": float(width.max()) if width.size else float("nan"),
            "width_min_px": float(core.min()) if core.size else float("nan"),
            "curvature_mean_per_px": float(np.mean(curv)) if curv else 0.0,
            "curvature_max_per_px": float(np.max(curv)) if curv else 0.0}


def branch_segments(g, pts, deg, min_branch: float) -> List[np.ndarray]:
    """Split the skeleton at branch points; drop tip-to-junction spurs shorter than min_branch (repeatedly)."""
    alive = np.ones(len(pts), bool)
    for _ in range(20):
        segs = _trace_segments(g, deg, alive)
        spurs = [s for s in segs if (deg_alive(g, alive)[s[0]] == 1) ^ (deg_alive(g, alive)[s[-1]] == 1)
                 and _len(pts[s]) < min_branch]
        if not spurs:
            return [pts[s] for s in segs]
        for s in spurs:
            d = deg_alive(g, alive)
            for i in s:
                if d[i] <= 2:
                    alive[i] = False
    return [pts[s] for s in _trace_segments(g, deg, alive)]


def deg_alive(g, alive):
    sub = g.multiply(alive[:, None]).multiply(alive[None, :]).tocsr()
    return np.where(alive, np.diff(sub.indptr), 0)


def _len(p):
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _trace_segments(g, deg, alive):
    d = deg_alive(g, alive)
    nodes = set(np.nonzero(alive & (d != 2))[0])
    seen, segs = set(), []
    for n in nodes:
        for nb in g.indices[g.indptr[n]:g.indptr[n + 1]]:
            if not alive[nb] or (n, nb) in seen:
                continue
            seg, prev, cur = [n], n, nb
            while True:
                seg.append(cur)
                if cur in nodes:
                    break
                nxt = [x for x in g.indices[g.indptr[cur]:g.indptr[cur + 1]] if alive[x] and x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            seen.add((n, seg[1])); seen.add((seg[-1], seg[-2]))
            segs.append(np.array(seg))
    return segs


# ------------------------------------------------------------------ one mask
def centerline(mask: np.ndarray, branches: bool = False, sigma: float = 2.0, min_branch: Optional[float] = None):
    from skimage.morphology import skeletonize
    mask = clean(mask)
    if mask.sum() < 5:
        return None
    dist = ndi.distance_transform_edt(mask)
    sk = skeletonize(mask)
    g, pts, deg = skeleton_graph(sk)
    if len(pts) < 2:
        return None
    if not branches:
        path = extend_to_tip(longest_path(g, pts, deg), mask)
        xy = smooth_resample(path, sigma)
        return {"paths": [xy], **path_metrics(xy, dist)}
    mb = min_branch if min_branch is not None else 2 * float(dist.max())
    raw = branch_segments(g, pts, deg, mb)
    # a skeleton junction is a small cluster of pixels, not one pixel: steps inside a cluster are not segments
    junction = np.zeros(sk.shape, bool)
    jd = deg_alive(g, np.ones(len(pts), bool))
    junction[pts[jd >= 3, 1].astype(int), pts[jd >= 3, 0].astype(int)] = True
    jl, n_junctions = ndi.label(junction, structure=np.ones((3, 3)))
    def in_junction(p):
        return bool(junction[int(p[1]), int(p[0])])
    raw = [s for s in raw if not (_len(s) <= 3 and in_junction(s[0]) and in_junction(s[-1]))]
    segs = [smooth_resample(s, sigma) for s in raw]
    segs = [s for s in segs if len(s) > 1]
    return {"paths": segs, "segments": [path_metrics(s, dist) for s in segs],
            "n_segments": len(segs), "n_branch_points": int(n_junctions), "n_tips": int((jd == 1).sum()),
            "length_px": float(sum(_len(s) for s in segs))}


# ------------------------------------------------------------------ files
def load_scales(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    out = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter="\t" if path.endswith((".tsv", ".txt")) else ",")
        for r in rd:
            k = {c.lower(): v for c, v in r.items()}
            img = k.get("image_filename") or k.get("image") or k.get("filename") or k.get("file_name")
            val = k.get("pixels_per_unit") or k.get("px_per_mm") or k.get("pixels_per_mm") or k.get("scale_px_per_mm")
            try:
                if img and val:
                    out[Path(img).stem.lower()] = float(val)
            except ValueError:
                pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("Method (per mask):")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coco")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--categories", nargs="*", help="only these category names (default: every mask)")
    ap.add_argument("--images", help="image folder (for overlays)")
    ap.add_argument("--scales", help="CSV with image and pixels-per-unit (mm) columns")
    ap.add_argument("--px-per-mm", type=float, help="one scale for every image")
    ap.add_argument("--branches", action="store_true", help="keep the branch graph (default: longest single path)")
    ap.add_argument("--min-branch", type=float, help="branch mode: drop spurs shorter than this (px; default 2 x max width)")
    ap.add_argument("--smooth", type=float, default=2.0, help="smoothing sigma in px (default 2)")
    ap.add_argument("--overlays", action="store_true", help="write a PNG per image with the centre lines drawn")
    a = ap.parse_args(argv)

    coco = json.load(open(a.coco))
    ims = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    scales = load_scales(a.scales)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, lines, per_image = [], [], {}
    wanted = set(a.categories) if a.categories else None
    for ann in coco["annotations"]:
        seg = ann.get("segmentation")
        cname = cats.get(ann.get("category_id"), "?")
        if not seg or (wanted and cname not in wanted):
            continue
        im = ims.get(ann["image_id"])
        if not im or not im.get("width") or not im.get("height"):
            continue
        res = centerline(polygon_mask(seg, im["height"], im["width"]), a.branches, a.smooth, a.min_branch)
        if res is None:
            continue
        ppm = im.get("scale_px_per_mm") or scales.get(Path(im["file_name"]).stem.lower()) or a.px_per_mm
        to_mm = (lambda v: v / ppm) if ppm else (lambda v: float("nan"))
        base = {"image": im["file_name"], "category": cname, "annotation_id": ann.get("id"),
                "px_per_mm": ppm or ""}
        if a.branches:
            rows.append({**base, "segment": "all", "length_px": round(res["length_px"], 2),
                         "length_mm": round(to_mm(res["length_px"]), 5), "n_segments": res["n_segments"],
                         "n_branch_points": res["n_branch_points"], "n_tips": res["n_tips"]})
            for i, m in enumerate(res["segments"], 1):
                rows.append({**base, "segment": i, "length_px": round(m["length_px"], 2),
                             "length_mm": round(to_mm(m["length_px"]), 5), "chord_px": round(m["chord_px"], 2),
                             "sinuosity": round(m["sinuosity"], 4), "width_mean_px": round(m["width_mean_px"], 2)})
        else:
            r = {**base}
            for k in ("length", "chord"):
                r[f"{k}_px"] = round(res[f"{k}_px"], 2); r[f"{k}_mm"] = round(to_mm(res[f"{k}_px"]), 5)
            r["sinuosity"] = round(res["sinuosity"], 4)
            for k in ("width_mean", "width_max", "width_min"):
                r[f"{k}_px"] = round(res[f"{k}_px"], 2); r[f"{k}_mm"] = round(to_mm(res[f"{k}_px"]), 5)
            r["curvature_mean_per_mm"] = round(res["curvature_mean_per_px"] * ppm, 4) if ppm else ""
            r["curvature_max_per_mm"] = round(res["curvature_max_per_px"] * ppm, 4) if ppm else ""
            rows.append(r)
        for p in res["paths"]:
            lines.append({"image_id": im["id"], "category": f"{cname}_centerline", "points": p})
        per_image.setdefault(im["id"], []).extend(res["paths"])

    if not rows:
        sys.exit("no masks measured (check --categories and that images have width/height)")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "centerline_measurements.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # COCO with the centre lines as GUI line annotations
    cl_cats = {n: i + 1 for i, n in enumerate(dict.fromkeys(l["category"] for l in lines))}
    anns = []
    for i, l in enumerate(lines, 1):
        p = l["points"]
        anns.append({"id": i, "image_id": l["image_id"], "category_id": cl_cats[l["category"]], "iscrowd": 0,
                     "segmentation": [], "area": 0.0, "is_line": True, "line_thickness": 2,
                     "line_points": [[int(round(x)), int(round(y))] for x, y in p[::max(1, len(p) // 60)]] +
                                    [[int(round(p[-1][0])), int(round(p[-1][1]))]],
                     "num_line_keypoints": min(len(p), 61),
                     "bbox": [float(p[:, 0].min()), float(p[:, 1].min()), float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))]})
    json.dump({"images": [ims[i] for i in per_image], "annotations": anns,
               "categories": [{"id": i, "name": n, "supercategory": "centerline"} for n, i in cl_cats.items()]},
              open(out / "centerlines_coco.json", "w"), indent=1)
    if a.overlays and a.images:
        import cv2
        od = out / "overlays"; od.mkdir(exist_ok=True)
        for iid, paths in per_image.items():
            fn = ims[iid]["file_name"]
            cand = [p for p in Path(a.images).rglob(Path(fn).stem + ".*")]
            img = cv2.imread(str(cand[0]), cv2.IMREAD_COLOR) if cand else None
            if img is None:
                continue
            for p in paths:
                cv2.polylines(img, [np.round(p).astype(np.int32)], False, (0, 140, 255), max(2, img.shape[1] // 600))
                for q in (p[0], p[-1]):
                    cv2.circle(img, tuple(int(v) for v in q), max(3, img.shape[1] // 300), (255, 0, 200), -1)
            cv2.imwrite(str(od / (Path(fn).stem + "_centerline.png")), img)
    n_mm = sum(1 for r in rows if r.get("px_per_mm"))
    print(f"measured {len(rows)} {'rows' if a.branches else 'masks'} ({n_mm} with a mm scale) -> "
          f"{out/'centerline_measurements.csv'}; centre lines for the GUI -> {out/'centerlines_coco.json'}")


if __name__ == "__main__":
    main()
