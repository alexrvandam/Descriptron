#!/usr/bin/env python3
"""
descriptron_mask_tools_v1.py - turn predicted or hand-made segmentations into the masks an analysis needs
==========================================================================================================

Nothing here knows any organism: categories are named by options or by a profile file (YAML or JSON). Three
subcommands, all reading and writing COCO JSON (polygon output, as the rest of Descriptron):

  combine   build new categories from the categories of a prediction COCO (RLE or polygons), one image at a time,
            by a list of operations per output (a "recipe"):
              union: [pred:NAME | out:NAME ...]   union of prediction categories / outputs already built
              foreground: R                     restrict to the specimen's foreground mask; with R > 0 first close
                                                gaps up to R px (bridging pale membranes or sutures that the
                                                foreground mask calls background), fill holes and keep within the
                                                foreground grown by R
              open: R / close: R                morphological opening / closing with a disk of radius R
              largest: true                     keep the largest connected piece and fill its holes
              subtract: [out:NAME ...]          remove outputs already built
              subtract_dilated: {masks: [out:NAME], size: [H, W]}   remove them grown by an H x W rectangle (e.g.
                                                1 x 41: the thin margin 20 px left and right of a neighbouring part)
            each output can be limited to images whose attribute matches (only: {view: [dorsal]}); image attributes
            (view, species, scale) come from an image table CSV. A one-output recipe can be given on the command
            line (--output NAME --from CAT[,CAT] [--subtract CAT] [--bridge_px R] [--open_px R] [--largest]).
  trim_end  trim every mask of one category at one end along an axis: (1) drop the part before the first line
            (row or column) where the mask reaches --width_frac of its largest width in the first --width_search of
            its length (a narrow lobe or dome); (2) if a band of lines much paler than the mask (median L* in the
            central --central share of the width more than --pale_dL above the mask's median) lies within the first
            --snap_frac of the length, cut at the far edge of the band farthest from that end (a membrane or suture);
            the largest piece is kept. --axis y|x, --end start|end (start = top or left).
  tile      split a container category into parts that tile it exactly, from a part-prediction COCO: container
            pixels predicted as one of the parts keep that part, every other container pixel takes the nearest part;
            each part is then its largest piece (pieces cut off go back to the nearest part), holes filled. Optional
            rules, all along --axis (y: rows, the first row being the front):
              --merge_narrow A:B:F   the front lines of B narrower than F x B's widest line, in front of that line,
                                     go to A (e.g. a neck predicted as pronotum goes to the head)
              --trim_wedge A:B:F:S   A may reach in front of its shoulder (the first line reaching S x its widest
                                     line) by at most F x its length; the rest goes to B (e.g. a scutellar wedge)
              --extra NAME=CATEGORY  a category of the container COCO copied as one more part (e.g. abdomen)
              --order_check A,B,C    the parts' centroids must follow this order along the axis (reported)

A profile (--profile) holds the same settings, as a section `mask_tools:` of a Descriptron taxon profile (the YAML
that run_full_pipeline_v2 --taxon_profile reads and biorag_make_taxon_profile_v1.py drafts; the other sections are
ignored here) or as a file with that section alone; command-line options override it. The Platydracus settings are
shipped as biorag_prompts/taxon_profiles/platydracus_mask_tools_profile.yaml. A flower would be, for example:

  mask_tools:
    combine:
      outputs:
        - {name: corolla, ops: [{union: [pred:petal]}, {foreground: 6}, {open: 2}, {largest: true}]}
        - {name: flower, ops: [{union: [out:corolla, pred:sepal]}, {close: 3}, {largest: true}]}
        - {name: calyx, ops: [{union: [out:flower]}, {subtract: [out:corolla]}, {open: 2}, {largest: true}]}
    tile: {container: flower, parts: [petal, sepal], axis: y}

  python descriptron_mask_tools_v1.py combine --profile platydracus_mask_tools_profile.yaml \\
      --predictions dinomask/predictions.json --image_dir crops/ --image_table crops/plate_split_panels.csv \\
      --out soma_v5.json
  python descriptron_mask_tools_v1.py trim_end --profile platydracus_mask_tools_profile.yaml --coco soma_v5.json \\
      --image_dir crops/ --out soma_v5b.json --report ventral_top_report.csv
  python descriptron_mask_tools_v1.py tile --profile platydracus_mask_tools_profile.yaml \\
      --predictions parts_predictions.json --coco soma_v5b.json --out parts_v2.json --report parts_v2_report.csv

Reproduces soma_v5/make_soma_v5.py, soma_v5/fix_ventral_top.py and parts_v1/make_parts_v2.py of the Platydracus
colour study byte for byte with the shipped profile (tests/test_descriptron_mask_tools.py checks synthetic inputs).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ============================================================================ helpers
def load_profile(path: Optional[str], section: str) -> Dict:
    """the `mask_tools: {<section>: ...}` part of a profile (YAML or JSON; a taxon profile or a file holding only
    that section, with or without the top-level `mask_tools` key)."""
    if not path:
        return {}
    txt = Path(path).read_text()
    if path.endswith(".json"):
        D = json.loads(txt)
    else:
        import yaml
        D = yaml.safe_load(txt) or {}
    D = D.get("mask_tools", D)
    return dict(D.get(section, {}) or {})


def disk(r: int):
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def largest_filled(m: np.ndarray) -> np.ndarray:
    """the largest connected piece with its holes filled (an empty mask stays empty)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(m)
    return ndi.binary_fill_holes(lab == 1 + np.argmax(ndi.sum(m, lab, range(1, n + 1)))) if n else m


def poly_mask(segs, h: int, w: int) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(p).reshape(-1, 1, 2).astype(np.int32) for p in segs], 1)
    return m > 0


def decode(seg, h: int, w: int) -> np.ndarray:
    """a COCO segmentation (polygons, RLE or uncompressed RLE) as a boolean mask."""
    if isinstance(seg, list):
        return poly_mask(seg, h, w)
    from pycocotools import mask as mu
    rle = mu.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
    return mu.decode(rle).astype(bool)


def to_polygons(m: np.ndarray):
    """outer contours (cv2.RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) as COCO polygons and their bounding box."""
    import cv2
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = [q.reshape(-1).astype(float).tolist() for q in cs if len(q) >= 3]
    box = list(cv2.boundingRect(np.vstack(cs))) if cs else [0, 0, 0, 0]
    return poly, box


def _orient(m: np.ndarray, axis: str, end: str) -> np.ndarray:
    """view the array so that the chosen end of the chosen axis is row 0 (the inverse is _unorient)."""
    a = m if axis == "y" else np.swapaxes(m, 0, 1)
    return a[::-1] if end == "end" else a


def _unorient(m: np.ndarray, axis: str, end: str) -> np.ndarray:
    a = m[::-1] if end == "end" else m
    return a if axis == "y" else np.swapaxes(a, 0, 1)


# ============================================================================ combine
def apply_recipe(ops: List[Dict], pred: Dict[str, np.ndarray], built: Dict[str, np.ndarray], fg: Optional[np.ndarray],
                 shape) -> np.ndarray:
    """run one output's operations (see the module docstring) on one image."""
    import cv2
    from scipy import ndimage as ndi
    z = np.zeros(shape, bool)
    m = z.copy()

    def get(ref):
        kind, _, nm = ref.partition(":")
        if not nm:
            kind, nm = "pred", kind
        return (pred if kind == "pred" else built).get(nm, z)

    for op in ops:
        (k, v), = op.items()
        if k == "union":
            for ref in v:
                m = m | get(ref)
        elif k == "foreground":
            if fg is None:
                raise ValueError("a 'foreground' step needs the foreground masks (--image_dir and --fg_pattern)")
            r = int(v)
            if r:
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(r)).astype(bool)
                m = ndi.binary_fill_holes(m) & cv2.dilate(fg.astype(np.uint8), disk(r)).astype(bool)
            else:
                m = m & fg
        elif k == "open":
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(int(v))).astype(bool)
        elif k == "close":
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, disk(int(v))).astype(bool)
        elif k == "largest":
            if v:
                m = largest_filled(m)
        elif k == "subtract":
            for ref in v:
                m = m & ~get(ref)
        elif k == "subtract_dilated":
            hh, ww = v.get("size", [1, 41])
            for ref in v["masks"]:
                m = m & ~cv2.dilate(get(ref).astype(np.uint8), np.ones((int(hh), int(ww)), np.uint8)).astype(bool)
        else:
            raise ValueError(f"unknown operation '{k}'")
    return m


def _read_table(path: Optional[str], file_col: str) -> Dict[str, Dict]:
    if not path:
        return {}
    return {r[file_col]: r for r in csv.DictReader(open(path))}


def _cast(v, typ: str):
    if typ == "float":
        return float(v) if v not in (None, "") else None
    if typ == "int":
        return int(v) if v not in (None, "") else None
    return v


def cmd_combine(a) -> int:
    import cv2
    P = load_profile(a.profile, "combine")
    outputs = P.get("outputs", [])
    if a.output:
        ops = [{"union": [f"pred:{c}" for c in a.from_.split(",")]}]
        if a.bridge_px is not None:
            ops.append({"foreground": a.bridge_px})
        if a.open_px:
            ops.append({"open": a.open_px})
        if a.largest:
            ops.append({"largest": True})
        if a.subtract:
            ops.append({"subtract": [f"pred:{c}" for c in a.subtract.split(",")]})
        outputs = [{"name": a.output, "ops": ops}]
    if not outputs:
        sys.exit("combine: no outputs (give --profile with mask_tools: combine: outputs, or --output/--from)")
    cats = P.get("categories") or [o["name"] for o in outputs]
    write_order = P.get("write_order") or cats
    tab = P.get("image_table", {})
    file_col = a.file_col or tab.get("file", "file_name")
    attrs = tab.get("attrs", [{"name": "species", "column": "species"}, {"name": "view", "column": "view"}])
    filt = P.get("image_filter", {})
    fg_pattern = a.fg_pattern or P.get("fg_pattern", "{stem}_mask.png")
    D = json.load(open(a.predictions))
    cat = {c["id"]: c["name"] for c in D["categories"]}
    ims = {i["id"]: i for i in D["images"]}
    G: Dict[str, Dict[str, np.ndarray]] = {}
    for an in D["annotations"]:
        im = ims[an["image_id"]]
        d = G.setdefault(im["file_name"], {})
        m = decode(an["segmentation"], im["height"], im["width"])
        nm = cat[an["category_id"]]
        d[nm] = d[nm] | m if nm in d else m
    table = _read_table(a.image_table or tab.get("path"), file_col)
    files = list(table) if table else [i["file_name"] for i in D["images"]]
    images, anns, log = [], [], []
    for fn in files:
        r = table.get(fn, {})
        if any(str(r.get(k, "")) not in [str(x) for x in v] for k, v in filt.items()):
            continue
        fg = None
        if a.image_dir:
            fp = Path(a.image_dir) / fg_pattern.format(stem=Path(fn).stem, file=fn)
            if fp.exists():
                fg = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE) > 127
        if fg is not None:
            shape = fg.shape
        else:
            im = next((i for i in D["images"] if i["file_name"] == fn), None)
            if im is None:
                log.append((fn, "no foreground mask and no prediction image")); continue
            shape = (im["height"], im["width"])
        pred = G.get(fn, {})
        built: Dict[str, np.ndarray] = {}
        for o in outputs:
            only = o.get("only", {})
            if any(str(r.get(k, "")) not in [str(x) for x in v] for k, v in only.items()):
                continue
            built[o["name"]] = apply_recipe(o["ops"], pred, built, fg, shape)
        h, w = shape
        rec = {"id": len(images) + 1, "file_name": fn, "width": w, "height": h}
        for at in attrs:
            rec[at["name"]] = _cast(r.get(at["column"]), at.get("type", "str")) if r else None
        images.append(rec)
        for k in [c for c in write_order if c in built]:
            m = built[k]
            poly, box = to_polygons(m)
            if not poly:
                log.append((fn, f"empty {k}")); continue
            anns.append({"id": len(anns) + 1, "image_id": rec["id"], "category_id": cats.index(k) + 1,
                         "segmentation": poly, "area": int(m.sum()), "bbox": box, "iscrowd": 0})
    json.dump({"images": images, "annotations": anns, "categories": [{"id": i + 1, "name": n} for i, n in enumerate(cats)]},
              open(a.out, "w"))
    print(len(images), "images,", len(anns), "masks; problems:", log)
    return 0


# ============================================================================ trim_end
def trim_end_mask(m: np.ndarray, L: Optional[np.ndarray], width_frac: float, width_search: float, snap_frac: float,
                  pale_dL: float, central: float, axis: str = "y", end: str = "start") -> Dict:
    """trim one mask at one end (module docstring); L = L* image (0-100) for the pale-band snap, or None. Returns the
    new mask and the lines (in the oriented frame: 0 = the chosen end) old_top, flat_row, membrane_row, new_top."""
    from scipy import ndimage as ndi
    mo = _orient(m, axis, end)
    Lo = _orient(L, axis, end) if L is not None else None
    ys = np.where(mo.any(1))[0]
    y0, y1 = ys[0], ys[-1]; H = y1 - y0 + 1
    width = mo.sum(1); wmax = width[y0:y0 + int(width_search * H)].max()
    flat = y0 + int(np.argmax(width[y0:] >= width_frac * wmax))
    snap = None
    if Lo is not None and snap_frac > 0:
        med = np.median(Lo[mo]); prof = []
        mg = (1 - central) / 2
        for y in range(y0, y0 + int(snap_frac * H) + 1):
            xs = np.where(mo[y])[0]
            if len(xs) < 5:
                prof.append(0.0); continue
            w = xs[-1] - xs[0]; c = Lo[y, xs[0] + int(mg * w): xs[-1] - int(mg * w) + 1]
            prof.append(np.median(c) - med)
        pale = np.array(prof) > pale_dL
        if pale.any():
            lab, n = ndi.label(pale)
            last = max(range(1, n + 1), key=lambda k: np.where(lab == k)[0].max())
            snap = y0 + int(np.where(lab == last)[0].max()) + 1
    cut = max(flat, snap or 0)
    m2 = mo.copy(); m2[:cut] = False
    lab, n = ndi.label(m2)
    m2 = lab == 1 + np.argmax(ndi.sum(m2, lab, range(1, n + 1))) if n else m2
    return {"mask": _unorient(m2, axis, end), "old_top": y0, "flat_row": flat, "membrane_row": snap, "new_top": cut}


def cmd_trim_end(a) -> int:
    import cv2
    P = load_profile(a.profile, "trim_end")
    g = lambda k, d: getattr(a, k) if getattr(a, k) is not None else P.get(k, d)
    category = g("category", None)
    if not category:
        sys.exit("trim_end: --category (or mask_tools: trim_end: category in the profile) is required")
    D = json.load(open(a.coco))
    cid = [c["id"] for c in D["categories"] if c["name"] == category][0]
    ims = {i["id"]: i for i in D["images"]}
    rows = []
    for an in D["annotations"]:
        if an["category_id"] != cid:
            continue
        im = ims[an["image_id"]]
        m = decode(an["segmentation"], im["height"], im["width"])
        L = None
        if a.image_dir:
            img = cv2.imread(str(Path(a.image_dir) / im["file_name"]))
            L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float) * 100 / 255
        R = trim_end_mask(m, L, g("width_frac", 0.8), g("width_search", 0.5), g("snap_frac", 0.15), g("pale_dL", 15.0),
                          g("central", 0.6), g("axis", "y"), g("end", "start"))
        m2 = R["mask"]
        an["segmentation"], an["bbox"] = to_polygons(m2)
        an["area"] = int(m2.sum())
        rows.append((im["file_name"], R["old_top"], R["flat_row"], R["membrane_row"] if R["membrane_row"] is not None else "",
                     R["new_top"], round(1 - m2.sum() / m.sum(), 3)))
    json.dump(D, open(a.out, "w"))
    if a.report:
        with open(a.report, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["file_name", "old_top", "flat_row", "membrane_row", "new_top", "share_removed"])
            w.writerows(rows)
    r = np.array([x[-1] for x in rows]) if rows else np.zeros(1)
    print(len(rows), f"{category} masks; share removed: median %.3f, >0.05: %d, >0.15: %d; snapped to a pale band: %d"
          % (np.median(r), (r > 0.05).sum(), (r > 0.15).sum(), sum(1 for x in rows if x[3] != "")))
    return 0


# ============================================================================ tile
def tile_container(container: np.ndarray, lab0: np.ndarray, parts: List[str], merge_narrow: List[str],
                   trim_wedge: List[str], axis: str = "y") -> Optional[np.ndarray]:
    """label image (0 = outside, i + 1 = parts[i]) tiling the container (module docstring), from lab0, the predicted
    labels (i + 1 = parts[i], 0 = none); None when a part has no predicted pixel in the container."""
    from scipy import ndimage as ndi
    lab = lab0.astype(np.int8).copy()
    lab[~container] = 0
    if any(not (lab == i + 1).any() for i in range(len(parts))):
        return None
    for _ in range(2):
        un = container & (lab == 0)
        if un.any():
            _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
            lab[un] = lab[iy[un], ix[un]]
        for k in range(1, len(parts) + 1):
            m = lab == k
            l_, n_ = ndi.label(m)
            if n_ > 1:
                big = 1 + np.argmax(ndi.sum(m, l_, range(1, n_ + 1)))
                lab[m & (l_ != big)] = 0
    lo = _orient(lab, axis, "start")                       # a view: writes go to lab
    for rule in merge_narrow:
        A, B, F = rule.split(":"); F = float(F); ia, ib = parts.index(A) + 1, parts.index(B) + 1
        mb = lo == ib
        if mb.any():
            w = mb.sum(1); ys = np.where(w > 0)[0]; wmax = w.max()
            ymax = ys[np.argmax(w[ys])]
            nr = [y for y in ys if y < ymax and w[y] < F * wmax]
            if nr:
                nm = mb.copy(); nm[max(nr) + 1:] = False
                lo[nm] = ia
    for rule in trim_wedge:
        A, B, F, S = rule.split(":"); F, S = float(F), float(S); ia, ib = parts.index(A) + 1, parts.index(B) + 1
        ma = lo == ia
        if ma.any() and (lo == ib).any():
            we = ma.sum(1); ys = np.where(we > 0)[0]; wmax = we.max()
            y_sh = ys[np.argmax(we[ys] >= S * wmax)]
            Lg = ys.max() - ys.min() + 1
            ytop = y_sh - int(F * Lg)
            wedge = ma.copy(); wedge[ytop:] = False
            lo[wedge] = ib
    return lab


def cmd_tile(a) -> int:
    from scipy import ndimage as ndi
    P = load_profile(a.profile, "tile")
    g = lambda k, d: getattr(a, k) if getattr(a, k) not in (None, []) else P.get(k, d)
    container = g("container", None); parts = g("parts", [])
    if not container or not parts:
        sys.exit("tile: --container and --parts (or the profile's mask_tools: tile) are required")
    extra = dict(x.split("=", 1) for x in g("extra", [])) if isinstance(g("extra", []), list) else dict(g("extra", {}))
    merge_narrow = g("merge_narrow", []); trim_wedge = g("trim_wedge", [])
    order = g("order_check", []); order = order.split(",") if isinstance(order, str) else list(order)
    axis = g("axis", "y")
    filt = g("image_filter", {})
    if isinstance(filt, list):
        filt = dict(x.split("=", 1) for x in filt)
    Pd = json.load(open(a.predictions)); pc = {c["id"]: c["name"] for c in Pd["categories"]}
    pim = {i["file_name"]: i["id"] for i in Pd["images"]}
    pann = collections.defaultdict(list)
    for an in Pd["annotations"]:
        pann[an["image_id"]].append(an)
    Dc = json.load(open(a.coco)); cid = {c["name"]: c["id"] for c in Dc["categories"]}
    names = list(parts) + list(extra)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    images, anns, rows = [], [], []
    for im in Dc["images"]:
        if any(str(im.get(k)) != str(v) for k, v in filt.items()):
            continue
        h, w = im["height"], im["width"]
        cont = np.zeros((h, w), bool); ext = {n: np.zeros((h, w), bool) for n in extra}
        for an in Dc["annotations"]:
            if an["image_id"] == im["id"]:
                if an["category_id"] == cid[container]:
                    cont |= decode(an["segmentation"], h, w)
                for n, c in extra.items():
                    if an["category_id"] == cid[c]:
                        ext[n] |= decode(an["segmentation"], h, w)
        # predicted labels in annotation order (a later prediction overwrites an earlier one where they overlap)
        lab0 = np.zeros((h, w), np.int8)
        for an in pann.get(pim.get(im["file_name"]), []):
            n = pc[an["category_id"]]
            if n in parts:
                lab0[decode(an["segmentation"], h, w) & cont] = parts.index(n) + 1
        lab = tile_container(cont, lab0, parts, merge_narrow, trim_wedge, axis)
        if lab is None:
            missing = [p for i, p in enumerate(parts) if not (lab0 == i + 1).any()]
            rows.append({"file_name": im["file_name"], "status": "missing " + ",".join(missing)}); continue
        masks = {p: ndi.binary_fill_holes(lab == i + 1) & cont for i, p in enumerate(parts)}
        masks.update(ext)
        ax_ = 0 if axis == "y" else 1
        cy = {p: ndi.center_of_mass(m)[ax_] for p, m in masks.items() if m.any()}
        order_ok = all(cy[x] < cy[y] for x, y in zip(order, order[1:])) if order and all(p in cy for p in order) else True
        images.append(dict(im))
        for k, (p, m) in enumerate(masks.items()):
            poly, box = to_polygons(m)
            if poly:
                anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": k + 1, "segmentation": poly,
                             "area": int(m.sum()), "bbox": box, "iscrowd": 0})
        sa = cont.sum()
        rows.append({"file_name": im["file_name"], "status": "ok" if order_ok else "order wrong",
                     **{f"{p}_share_of_{container}": round(masks[p].sum() / sa, 3) for p in parts}})
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(a.out, "w"))
    if a.report:
        with open(a.report, "w", newline="") as fh:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(len(images), "images;", collections.Counter(r["status"] for r in rows))
    return 0


# ============================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("combine", help="new categories from prediction categories by a recipe of operations")
    c.add_argument("--profile", default=None, help="YAML/JSON with mask_tools: combine: (outputs, categories, "
                   "write_order, image_table, image_filter, fg_pattern)")
    c.add_argument("--predictions", required=True, help="prediction COCO (RLE or polygons)")
    c.add_argument("--image_dir", default=None, help="folder with the foreground masks (--fg_pattern)")
    c.add_argument("--fg_pattern", default=None, help="foreground mask file name, {stem} = image stem, {file} = "
                   "image file name (default {stem}_mask.png)")
    c.add_argument("--image_table", default=None, help="CSV with one row per image (file name column --file_col) and "
                   "its attributes (profile image_table: attrs)")
    c.add_argument("--file_col", default=None)
    c.add_argument("--output", default=None, help="one-output recipe on the command line: the new category")
    c.add_argument("--from", dest="from_", default=None, help="its prediction categories (comma-separated)")
    c.add_argument("--subtract", default=None, help="prediction categories removed from it")
    c.add_argument("--bridge_px", type=int, default=None, help="foreground step: bridge gaps up to this radius")
    c.add_argument("--open_px", type=int, default=0)
    c.add_argument("--largest", action="store_true")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_combine)
    t = sub.add_parser("trim_end", help="trim a category at one end along an axis (narrow lobe, pale band)")
    t.add_argument("--profile", default=None)
    t.add_argument("--coco", required=True); t.add_argument("--image_dir", default=None,
                                                           help="photographs (for the pale-band snap)")
    t.add_argument("--category", default=None)
    t.add_argument("--axis", choices=["y", "x"], default=None); t.add_argument("--end", choices=["start", "end"], default=None)
    t.add_argument("--width_frac", type=float, default=None, help="default 0.8")
    t.add_argument("--width_search", type=float, default=None, help="share of the length searched for the widest "
                   "line (default 0.5)")
    t.add_argument("--snap_frac", type=float, default=None, help="default 0.15 (0 = no snap)")
    t.add_argument("--pale_dL", type=float, default=None, help="default 15 L* units")
    t.add_argument("--central", type=float, default=None, help="share of the width measured (default 0.6)")
    t.add_argument("--out", required=True); t.add_argument("--report", default=None)
    t.set_defaults(fn=cmd_trim_end)
    s = sub.add_parser("tile", help="split a container category into parts from a part-prediction COCO")
    s.add_argument("--profile", default=None)
    s.add_argument("--predictions", required=True, help="part predictions (COCO, RLE or polygons)")
    s.add_argument("--coco", required=True, help="COCO holding the container (and --extra) categories")
    s.add_argument("--container", default=None); s.add_argument("--parts", nargs="+", default=[])
    s.add_argument("--extra", nargs="+", default=[], metavar="NAME=CATEGORY")
    s.add_argument("--merge_narrow", nargs="+", default=[], metavar="A:B:FRAC")
    s.add_argument("--trim_wedge", nargs="+", default=[], metavar="A:B:FRAC:SHOULDER")
    s.add_argument("--order_check", default=None, metavar="A,B,C")
    s.add_argument("--axis", choices=["y", "x"], default=None)
    s.add_argument("--image_filter", nargs="+", default=[], metavar="ATTR=VALUE", help="e.g. view=dorsal")
    s.add_argument("--out", required=True); s.add_argument("--report", default=None)
    s.set_defaults(fn=cmd_tile)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
