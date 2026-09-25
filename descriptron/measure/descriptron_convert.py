#!/usr/bin/env python3
"""
descriptron_convert.py — bring landmark and annotation files into Descriptron's COCO (and back out)
=====================================================================================================

Descriptron's tools (the GUI, SAM2-PAL, DINOLand, Detectron2, the measurement and BioRAG pipelines) all
read one COCO JSON: images with width/height, polygon masks, and landmarks as one keypoint annotation
per specimen with a fixed numbering (`point_order`) and named landmarks in the category. This program
converts the formats morphologists already have into it, and writes the landmark formats back out.

  tps2coco        tpsDig / tpsUtil .tps (LM=, CURVES=, POINTS=, IMAGE=, ID=, SCALE=)
  coco2tps        COCO keypoints -> .tps (for tpsRelw, MorphoJ, geomorph::readland.tps)
  morphoj2coco    MorphoJ text dataset: ID, x1, y1, x2, y2, ... (optional header row)
  coco2morphoj    COCO keypoints -> MorphoJ text dataset
  stereomorph2coco  StereoMorph shape files (<landmarks.pixel>, <curves.pixel>, ...), one per image
  table2coco      long table: image, landmark, x, y (CSV or TSV, e.g. from R or a spreadsheet)
  via2coco        VGG Image Annotator 2: project JSON, region-data JSON, or VIA's own COCO export.
                  Unlike VIA's COCO exporter, points and polylines are kept (as landmarks and lines).

Coordinates. Descriptron uses image pixels with the origin at the TOP-left (y down). tpsDig puts the
origin at the BOTTOM-left (y up), so .tps y values are flipped with the image height (the images must be
found with --images). MorphoJ files usually come from tpsDig and are treated the same way by default.
Use --y-origin to override either.

Scale. tpsDig's SCALE= is units per pixel (e.g. mm/pixel). Descriptron records pixels per mm
(images[].scale_px_per_mm). A SCALE above 1 is implausible as mm per pixel for specimen photographs and
is read as pixels per mm with a warning (older Descriptron .tps files wrote it that way); force the
reading with --scale-units.

Missing landmarks: negative coordinates (tpsDig / geomorph convention) or NA become visibility 0 and
are written back as -1 -1.

Examples
    python descriptron_convert.py tps2coco wings.tps --images photos/ --out wings.json --names names.txt
    python descriptron_convert.py coco2tps wings.json --images photos/ --out wings.tps
    python descriptron_convert.py via2coco via_project.json --images photos/ --out annotations.json
    python descriptron_convert.py tps2coco wings.tps --images photos/ --out wings.json --per-image refs/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXT = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif")


def warn(msg: str):
    sys.stderr.write(f"warning: {msg}\n")


# ------------------------------------------------------------------ images
class ImageIndex:
    """Find an image by name: exact, then case-insensitive, then same stem with any extension."""

    def __init__(self, folder: Optional[str]):
        self.folder = Path(folder) if folder else None
        self.by_name, self.by_stem = {}, {}
        if self.folder and self.folder.is_dir():
            for p in sorted(self.folder.rglob("*")):
                if p.suffix.lower() in IMG_EXT and not p.name.startswith("._"):
                    self.by_name.setdefault(p.name.lower(), p)
                    self.by_stem.setdefault(p.stem.lower(), p)
        self._size = {}

    def find(self, name: str) -> Optional[Path]:
        base = re.split(r"[\\/]", name.strip())[-1]
        p = self.by_name.get(base.lower()) or self.by_stem.get(Path(base).stem.lower())
        return p

    def size(self, name: str) -> Tuple[Optional[Path], int, int]:
        p = self.find(name)
        if p is None:
            return None, 0, 0
        if p not in self._size:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    self._size[p] = im.size
            except Exception as e:
                warn(f"cannot read size of {p}: {e}")
                self._size[p] = (0, 0)
        w, h = self._size[p]
        return p, w, h


# ------------------------------------------------------------------ COCO building
class CocoBuilder:
    def __init__(self):
        self.images: "OrderedDict[str, dict]" = OrderedDict()
        self.annotations: List[dict] = []
        self.categories: "OrderedDict[str, dict]" = OrderedDict()

    def image(self, file_name: str, width: int, height: int, **extra) -> int:
        if file_name not in self.images:
            self.images[file_name] = {"id": len(self.images) + 1, "file_name": file_name,
                                      "width": int(width), "height": int(height), **extra}
        else:
            self.images[file_name].update({k: v for k, v in extra.items() if v is not None})
        return self.images[file_name]["id"]

    def category(self, name: str, keypoints: Optional[List[str]] = None, supercategory: str = "") -> int:
        if name not in self.categories:
            c = {"id": len(self.categories) + 1, "name": name, "supercategory": supercategory or name}
            if keypoints is not None:
                c["keypoints"], c["skeleton"] = list(keypoints), []
            self.categories[name] = c
        elif keypoints is not None:
            self.categories[name]["keypoints"] = list(keypoints)
        return self.categories[name]["id"]

    def add(self, ann: dict):
        ann["id"] = len(self.annotations) + 1
        ann.setdefault("iscrowd", 0)
        self.annotations.append(ann)

    def keypoints(self, image_id: int, cat_id: int, pts: List[Optional[Tuple[float, float]]], extra=None):
        kp, xs, ys = [], [], []
        for p in pts:
            if p is None:
                kp += [0, 0, 0]
            else:
                kp += [round(float(p[0]), 3), round(float(p[1]), 3), 2]
                xs.append(p[0]); ys.append(p[1])
        bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)] if xs else [0, 0, 0, 0]
        self.add({"image_id": image_id, "category_id": cat_id, "keypoints": kp,
                  "num_keypoints": len(xs), "point_order": list(range(1, len(pts) + 1)),
                  "bbox": [round(v, 3) for v in bbox], "area": round(max(1.0, bbox[2] * bbox[3]), 3),
                  **(extra or {})})

    def polygon(self, image_id: int, cat_id: int, xs: List[float], ys: List[float], extra=None):
        seg = [round(float(v), 3) for xy in zip(xs, ys) for v in xy]
        if len(seg) < 6:
            return
        area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs))))
        self.add({"image_id": image_id, "category_id": cat_id, "segmentation": [seg],
                  "bbox": [round(min(xs), 3), round(min(ys), 3), round(max(xs) - min(xs), 3), round(max(ys) - min(ys), 3)],
                  "area": round(area, 3), **(extra or {})})

    def line(self, image_id: int, cat_id: int, xs: List[float], ys: List[float], thickness: int = 3):
        """A polyline in the GUI's line-annotation form: an outline polygon plus the points themselves."""
        seg = _line_outline(xs, ys, thickness)
        self.add({"image_id": image_id, "category_id": cat_id, "segmentation": [seg] if seg else [],
                  "bbox": [round(min(xs), 3), round(min(ys), 3), round(max(xs) - min(xs), 3), round(max(ys) - min(ys), 3)],
                  "area": 0.0, "is_line": True, "line_points": [[int(round(x)), int(round(y))] for x, y in zip(xs, ys)],
                  "line_thickness": thickness, "num_line_keypoints": len(xs)})

    def to_dict(self) -> dict:
        return {"images": list(self.images.values()), "annotations": self.annotations,
                "categories": list(self.categories.values())}


def _line_outline(xs, ys, thickness):
    """Closed outline around a polyline of the given thickness (for the mask the GUI shows)."""
    try:
        import numpy as np, cv2
        pts = np.array(list(zip(xs, ys)), float)
        x0, y0 = pts.min(0) - thickness - 2
        w, h = (pts.max(0) - pts.min(0) + 2 * thickness + 5).astype(int)
        m = np.zeros((max(h, 1), max(w, 1)), np.uint8)
        cv2.polylines(m, [(pts - [x0, y0]).round().astype(np.int32)], False, 1, thickness)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return []
        c = max(cs, key=cv2.contourArea).reshape(-1, 2) + [x0, y0]
        return [round(float(v), 2) for xy in c for v in xy]
    except Exception:
        return []


def _write(coco: dict, out: str, per_image: Optional[str]):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco, f, indent=1)
    n_kp = sum(1 for a in coco["annotations"] if "keypoints" in a)
    print(f"wrote {out}: {len(coco['images'])} images, {len(coco['annotations'])} annotations "
          f"({n_kp} landmark sets), {len(coco['categories'])} categories")
    if per_image:
        d = Path(per_image); d.mkdir(parents=True, exist_ok=True)
        for im in coco["images"]:
            anns = [dict(a, id=i + 1) for i, a in enumerate(a for a in coco["annotations"] if a["image_id"] == im["id"])]
            if not anns:
                continue
            with open(d / f"{Path(im['file_name']).stem}_keypoints.json", "w") as f:
                json.dump({"images": [im], "annotations": anns, "categories": coco["categories"]}, f, indent=1)
        print(f"  one file per image in {d}/ (DINOLand references take this form)")


def _read_names(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    return [l.strip() for l in open(path) if l.strip()]


_SCALE_NOTES: List[str] = []


def _scale_to_px_per_unit(scale: Optional[float], mode: str, where: str) -> Optional[float]:
    if not scale or scale <= 0:
        return None
    if mode == "per-pixel":
        return 1.0 / scale
    if mode == "pixels-per-unit":
        return scale
    if scale > 1:
        _SCALE_NOTES.append(where)
        if len(_SCALE_NOTES) == 1:
            warn(f"{where}: SCALE={scale} read as pixels per unit (as mm per pixel it would be implausible); "
                 f"use --scale-units per-pixel to force the tpsDig reading (reported once)")
        return scale
    return 1.0 / scale


def _to_top(x: float, y: float, h: int, origin: str) -> Tuple[float, float]:
    return (x, h - y) if origin == "bottom" else (x, y)


def _missing(x, y) -> bool:
    return x is None or y is None or (isinstance(x, float) and math.isnan(x)) or x < 0 or y < 0


# ------------------------------------------------------------------ TPS
def parse_tps(path: str) -> List[dict]:
    """Specimens: {'lm': [(x,y)...], 'curves': [[(x,y)...]...], 'image', 'id', 'scale', 'comment'}."""
    lines = [l.rstrip("\r\n") for l in open(path, encoding="utf-8", errors="replace")]
    specs, i = [], 0

    def nums(line):
        parts = line.replace(",", " ").split()
        try:
            return [float(p) for p in parts[:2]] if len(parts) >= 2 else None
        except ValueError:
            return None

    def header(line) -> bool:
        """IMAGE=/ID=/SCALE=/COMMENT= may come before or after the coordinates (tpsDig writes them after;
        other tools, including older Descriptron, before). Returns True if the line was one of them."""
        k, _, v = line.strip().partition("=")
        k = k.strip().upper()
        if cur is None or k not in ("IMAGE", "ID", "SCALE", "COMMENT", "VARIABLES"):
            return False
        if k == "SCALE":
            try:
                cur["scale"] = float(v)
            except ValueError:
                pass
        elif k in ("IMAGE", "ID", "COMMENT"):
            cur[k.lower()] = v.strip()
        return True

    def take(n, start):
        out, j = [], start
        while len(out) < n and j < len(lines):
            v = nums(lines[j]); j += 1
            if v is not None:
                out.append(tuple(v))
            elif "=" in lines[j - 1] and not header(lines[j - 1]):
                j -= 1; break            # LM=/CURVES=/POINTS= where coordinates were expected: stop
        return out, j

    cur = None
    while i < len(lines):
        s = lines[i].strip(); i += 1
        if not s:
            continue
        key, _, val = s.partition("=")
        key = key.strip().upper()
        if key == "LM":
            cur = {"lm": [], "curves": [], "image": None, "id": None, "scale": None, "comment": None}
            specs.append(cur)
            cur["lm"], i = take(int(float(val)), i)
        elif cur is None:
            continue
        elif key == "CURVES":
            for _ in range(int(float(val))):
                while i < len(lines) and not lines[i].strip().upper().startswith("POINTS="):
                    i += 1
                if i >= len(lines):
                    break
                n = int(float(lines[i].split("=", 1)[1])); i += 1
                pts, i = take(n, i)
                cur["curves"].append(pts)
        elif key == "IMAGE":
            cur["image"] = val.strip()
        elif key == "ID":
            cur["id"] = val.strip()
        elif key == "SCALE":
            try:
                cur["scale"] = float(val)
            except ValueError:
                pass
        elif key == "COMMENT":
            cur["comment"] = val.strip()
    return specs


def tps2coco(a):
    specs = parse_tps(a.input)
    if not specs:
        sys.exit(f"no LM= blocks found in {a.input}")
    idx, b = ImageIndex(a.images), CocoBuilder()
    n_lm = max(len(s["lm"]) for s in specs)
    names = _read_names(a.names) or [str(k) for k in range(1, n_lm + 1)]
    if len(names) < n_lm:
        sys.exit(f"--names has {len(names)} names but the file has up to {n_lm} landmarks")
    cat = b.category(a.category, names)
    curve_cats = {}
    skipped = 0
    for k, s in enumerate(specs, 1):
        name = s["image"] or s["id"] or f"specimen_{k}"
        p, w, h = idx.size(name)
        if a.y_origin == "bottom" and not h:
            warn(f"{name}: image not found under --images, so y cannot be flipped (tpsDig origin is bottom-left); skipped")
            skipped += 1
            continue
        fname = p.name if p else re.split(r"[\\/]", name)[-1]
        ppu = _scale_to_px_per_unit(s["scale"], a.scale_units, name)
        iid = b.image(fname, w, h, **({"scale_px_per_mm": round(ppu, 6)} if ppu else {}),
                      **({"tps_id": s["id"]} if s["id"] else {}))
        pts = [None if _missing(x, y) else _to_top(x, y, h, a.y_origin) for x, y in s["lm"]]
        pts += [None] * (len(names) - len(pts))
        b.keypoints(iid, cat, pts)
        for ci, curve in enumerate(s["curves"], 1):
            cname = f"{a.category}_curve{ci}"
            if cname not in curve_cats:
                curve_cats[cname] = b.category(cname, [f"c{ci}_{j}" for j in range(1, len(curve) + 1)])
            b.keypoints(iid, curve_cats[cname], [None if _missing(x, y) else _to_top(x, y, h, a.y_origin) for x, y in curve],
                        extra={"semilandmarks": True})
    if skipped:
        warn(f"{skipped} of {len(specs)} specimens skipped (images not found)")
    _write(b.to_dict(), a.out, a.per_image)


def _kp_sets(coco: dict, category: Optional[str]) -> List[Tuple[dict, dict, dict]]:
    ims = {i["id"]: i for i in coco["images"]}
    cats = {c["id"]: c for c in coco["categories"]}
    out = []
    for ann in coco["annotations"]:
        if "keypoints" not in ann or ann.get("semilandmarks"):
            continue
        c = cats.get(ann["category_id"], {})
        if category and c.get("name") != category:
            continue
        out.append((ims[ann["image_id"]], c, ann))
    if not out:
        sys.exit("no landmark (keypoint) annotations found" + (f" in category {category}" if category else ""))
    return out


def _ordered_points(ann) -> List[Optional[Tuple[float, float]]]:
    kp = ann["keypoints"]
    order = ann.get("point_order") or list(range(1, len(kp) // 3 + 1))
    pts = {}
    for j, o in enumerate(order):
        x, y, v = kp[3 * j:3 * j + 3]
        pts[int(o)] = None if v == 0 else (x, y)
    n = max(pts) if pts else 0
    return [pts.get(k) for k in range(1, n + 1)]


def _scales_csv(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t" if path.endswith((".tsv", ".txt")) else ","):
            keys = {k.lower(): v for k, v in r.items()}
            img = keys.get("image") or keys.get("filename") or keys.get("file_name")
            val = keys.get("px_per_mm") or keys.get("pixels_per_mm") or keys.get("scale_px_per_mm")
            if img and val:
                out[Path(img).stem.lower()] = float(val)
    return out


def coco2tps(a):
    coco = json.load(open(a.input))
    idx, scales = ImageIndex(a.images), _scales_csv(a.scales)
    with open(a.out, "w") as f:
        for k, (im, c, ann) in enumerate(_kp_sets(coco, a.category), 1):
            h = im.get("height") or idx.size(im["file_name"])[2]
            if a.y_origin == "bottom" and not h:
                warn(f"{im['file_name']}: no image height, y left as in the image (top origin)")
            pts = _ordered_points(ann)
            f.write(f"LM={len(pts)}\n")
            for p in pts:
                if p is None:
                    f.write("-1.00 -1.00\n")
                else:
                    x, y = p
                    if a.y_origin == "bottom" and h:
                        y = h - y
                    f.write(f"{x:.2f} {y:.2f}\n")
            f.write(f"IMAGE={im['file_name']}\nID={k}\n")
            ppm = im.get("scale_px_per_mm") or scales.get(Path(im["file_name"]).stem.lower())
            if ppm:
                f.write(f"SCALE={1.0 / float(ppm):.8f}\n")      # tpsDig: units (mm) per pixel
            f.write("\n")
    print(f"wrote {a.out}: {k} specimens (y origin {a.y_origin}; SCALE in mm per pixel where known)")


# ------------------------------------------------------------------ MorphoJ
def morphoj2coco(a):
    rows = [r for r in csv.reader(open(a.input, newline="", encoding="utf-8", errors="replace"),
                                  delimiter="\t" if "\t" in open(a.input).readline() else ",")
            if r and any(c.strip() for c in r)]

    def isnum(v):
        try:
            float(v); return True
        except ValueError:
            return v.strip().upper() in ("NA", "NAN", "")
    header = None
    if not all(isnum(v) for v in rows[0][1:]):
        header, rows = rows[0], rows[1:]
    idx, b = ImageIndex(a.images), CocoBuilder()
    n_lm = (len(rows[0]) - 1) // 2
    names = _read_names(a.names) or [str(k) for k in range(1, n_lm + 1)]
    cat = b.category(a.category, names)
    id_map = {}
    if a.id_to_image:
        for r in csv.reader(open(a.id_to_image)):
            if len(r) >= 2:
                id_map[r[0].strip()] = r[1].strip()
    skipped = 0
    for r in rows:
        sid = r[0].strip()
        name = id_map.get(sid, sid)
        p, w, h = idx.size(name)
        if a.y_origin == "bottom" and not h:
            warn(f"{sid}: image not found (use --id-to-image to map MorphoJ IDs to image files); skipped")
            skipped += 1
            continue
        vals = [float(v) if v.strip() and v.strip().upper() not in ("NA", "NAN") else float("nan") for v in r[1:1 + 2 * n_lm]]
        pts = []
        for j in range(n_lm):
            x, y = vals[2 * j], vals[2 * j + 1]
            pts.append(None if _missing(x, y) else _to_top(x, y, h, a.y_origin))
        iid = b.image(p.name if p else name, w, h, morphoj_id=sid)
        b.keypoints(iid, cat, pts)
    if skipped:
        warn(f"{skipped} of {len(rows)} specimens skipped")
    _write(b.to_dict(), a.out, a.per_image)


def coco2morphoj(a):
    coco = json.load(open(a.input))
    idx = ImageIndex(a.images)
    sets = _kp_sets(coco, a.category)
    n = max(len(_ordered_points(ann)) for _, _, ann in sets)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ID"] + [f"{ax}{k}" for k in range(1, n + 1) for ax in ("x", "y")])
        for im, c, ann in sets:
            h = im.get("height") or idx.size(im["file_name"])[2]
            row = [Path(im["file_name"]).stem]
            pts = _ordered_points(ann) + [None] * (n - len(_ordered_points(ann)))
            for p in pts:
                if p is None:
                    row += ["NA", "NA"]
                else:
                    x, y = p
                    row += [f"{x:.2f}", f"{(h - y) if (a.y_origin == 'bottom' and h) else y:.2f}"]
            w.writerow(row)
    print(f"wrote {a.out}: {len(sets)} specimens, {n} landmarks (first row = column labels; missing = NA)")


# ------------------------------------------------------------------ StereoMorph
def parse_stereomorph(path: str) -> dict:
    """StereoMorph shape file: tagged blocks <tag> ... </tag>; landmark rows 'name x y'."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = {}
    for m in re.finditer(r"<([\w.]+)>(.*?)</\1>", txt, re.S):
        blocks[m.group(1)] = m.group(2).strip()
    out = {"image": None, "landmarks": OrderedDict(), "curves": OrderedDict()}
    for tag in ("image.filename", "image.id", "image.file"):
        if blocks.get(tag):
            out["image"] = blocks[tag].splitlines()[0].strip()
            break

    def rows(tag):
        for line in blocks.get(tag, "").splitlines():
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    x = float(parts[-2]) if parts[-2].upper() != "NA" else float("nan")
                    y = float(parts[-1]) if parts[-1].upper() != "NA" else float("nan")
                except ValueError:
                    continue
                yield " ".join(parts[:-2]), x, y
    for name, x, y in rows("landmarks.pixel"):
        out["landmarks"][name] = (x, y)
    for name, x, y in rows("curves.pixel"):
        out["curves"].setdefault(name, []).append((x, y))
    return out


def stereomorph2coco(a):
    files = []
    for p in a.input:
        pp = Path(p)
        files += sorted(q for q in pp.glob("*.txt")) if pp.is_dir() else [pp]
    shapes = [(f, parse_stereomorph(str(f))) for f in files]
    shapes = [(f, s) for f, s in shapes if s["landmarks"] or s["curves"]]
    if not shapes:
        sys.exit("no StereoMorph landmarks or curves found")
    names = _read_names(a.names) or list(OrderedDict.fromkeys(n for _, s in shapes for n in s["landmarks"]))
    idx, b = ImageIndex(a.images), CocoBuilder()
    cat = b.category(a.category, names) if names else None
    for f, s in shapes:
        name = s["image"] or f.stem
        p, w, h = idx.size(name)
        iid = b.image(p.name if p else name, w, h)
        if cat:
            pts = []
            for n in names:
                xy = s["landmarks"].get(n)
                pts.append(None if xy is None or _missing(*xy) else _to_top(xy[0], xy[1], h, a.y_origin))
            b.keypoints(iid, cat, pts)
        for cname, cpts in s["curves"].items():
            cid = b.category(re.sub(r"\s+", "_", cname))
            good = [q for q in cpts if not _missing(*q)]
            if len(good) >= 2:
                b.line(iid, cid, [q[0] for q in good], [_to_top(q[0], q[1], h, a.y_origin)[1] for q in good])
    _write(b.to_dict(), a.out, a.per_image)


# ------------------------------------------------------------------ long table
def table2coco(a):
    delim = "\t" if a.input.endswith((".tsv", ".txt")) else ","
    with open(a.input, newline="") as f:
        rd = csv.DictReader(f, delimiter=delim)
        cols = {c.lower().strip(): c for c in rd.fieldnames}

        def col(*opts):
            for o in opts:
                if o in cols:
                    return cols[o]
            sys.exit(f"{a.input}: need a column named one of {opts} (found {list(cols)})")
        ci, cl, cx, cy = (col("image", "file", "filename", "file_name", "specimen", "id"),
                          col("landmark", "lm", "name", "point", "landmark_id"), col("x"), col("y"))
        data = OrderedDict()
        for r in rd:
            try:
                x, y = float(r[cx]), float(r[cy])
            except (TypeError, ValueError):
                x = y = float("nan")
            data.setdefault(r[ci].strip(), OrderedDict())[r[cl].strip()] = (x, y)
    names = _read_names(a.names) or list(OrderedDict.fromkeys(n for d in data.values() for n in d))
    if all(re.fullmatch(r"\d+", n) for n in names) and not a.names:
        names = sorted(names, key=int)
    idx, b = ImageIndex(a.images), CocoBuilder()
    cat = b.category(a.category, names)
    for img, lm in data.items():
        p, w, h = idx.size(img)
        if a.y_origin == "bottom" and not h:
            warn(f"{img}: image not found, skipped"); continue
        iid = b.image(p.name if p else img, w, h)
        b.keypoints(iid, cat, [None if (lm.get(n) is None or _missing(*lm[n])) else _to_top(*lm[n], h, a.y_origin)
                               for n in names])
    _write(b.to_dict(), a.out, a.per_image)


# ------------------------------------------------------------------ VIA 2
def _clean_name(s: str, underscores: bool) -> str:
    s = str(s).strip()
    return re.sub(r"\s+", "_", s) if underscores else s


def _via_label(attrs: dict, class_attr: Optional[str]) -> str:
    if class_attr:
        v = attrs.get(class_attr, "")
    else:
        v = next((v for v in attrs.values() if isinstance(v, str) and v.strip()), "")
    if isinstance(v, dict):                       # checkbox attribute: {option: true}
        v = next((k for k, on in v.items() if on), "")
    return str(v).strip() or "region"


def via2coco(a):
    d = json.load(open(a.input))
    if isinstance(d, dict) and "images" in d and "annotations" in d:
        return _via_coco_fix(d, a)
    if isinstance(d, dict) and "_via_img_metadata" in d:
        meta = d["_via_img_metadata"]
    elif isinstance(d, dict) and all(isinstance(v, dict) and "regions" in v for v in d.values()):
        meta = d                                  # "Export annotations (as json)"
    elif isinstance(d, dict) and "project" in d and "file" in d:
        sys.exit("this looks like a VIA 3 project; export it from VIA as COCO or VIA2 JSON first")
    else:
        sys.exit("not a VIA 2 project, region-data export or COCO export")
    idx, b = ImageIndex(a.images), CocoBuilder()
    point_sets, lines, polys = OrderedDict(), [], []
    for key, v in meta.items():
        fname = v.get("filename") or key
        p, w, h = idx.size(fname)
        name = p.name if p else fname
        iid = b.image(name, w, h)
        for r in v.get("regions", []) if isinstance(v.get("regions"), list) else v.get("regions", {}).values():
            sh, at = r.get("shape_attributes", {}), r.get("region_attributes", {})
            label = _clean_name(_via_label(at, a.class_attr), a.underscores)
            kind = sh.get("name")
            if kind == "point":
                point_sets.setdefault(iid, OrderedDict())[label] = (sh["cx"], sh["cy"])
            elif kind == "polyline":
                lines.append((iid, label, sh["all_points_x"], sh["all_points_y"]))
            elif kind == "polygon":
                polys.append((iid, label, sh["all_points_x"], sh["all_points_y"]))
            elif kind == "rect":
                x, y, rw, rh = sh["x"], sh["y"], sh["width"], sh["height"]
                polys.append((iid, label, [x, x + rw, x + rw, x], [y, y, y + rh, y + rh]))
            elif kind in ("circle", "ellipse"):
                cx, cy = sh["cx"], sh["cy"]
                rx = sh.get("r", sh.get("rx", 0)); ry = sh.get("r", sh.get("ry", 0)); th = sh.get("theta", 0)
                t = [2 * math.pi * k / 36 for k in range(36)]
                xs = [cx + rx * math.cos(u) * math.cos(th) - ry * math.sin(u) * math.sin(th) for u in t]
                ys = [cy + rx * math.cos(u) * math.sin(th) + ry * math.sin(u) * math.cos(th) for u in t]
                polys.append((iid, label, xs, ys))
    for iid, label, xs, ys in polys:
        b.polygon(iid, b.category(label), xs, ys)
    for iid, label, xs, ys in lines:
        b.line(iid, b.category(label), xs, ys, a.line_thickness)
    if point_sets:
        names = _read_names(a.names) or list(OrderedDict.fromkeys(n for s in point_sets.values() for n in s))
        if not a.names and all(re.fullmatch(r"\d+", n) for n in names):
            names = sorted(names, key=int)
        cat = b.category(a.category, names)
        for iid, s in point_sets.items():
            extra = [n for n in s if n not in names]
            if extra:
                warn(f"image {iid}: points {extra} are not in the landmark list and were left out")
            b.keypoints(iid, cat, [s.get(n) for n in names])
    coco = b.to_dict()
    missing = [i["file_name"] for i in coco["images"] if not i["width"]]
    if missing:
        warn(f"{len(missing)} images not found under --images; their width/height are 0 (e.g. {missing[0]})")
    _write(coco, a.out, a.per_image)


def _via_coco_fix(d: dict, a):
    """VIA's COCO export -> Descriptron COCO: integer ids, nested polygons, bbox/area, clean names,
    real file names and sizes. VIA's exporter drops points and polylines; use the project JSON for those."""
    idx, b = ImageIndex(a.images), CocoBuilder()
    old_img = {}
    for im in d["images"]:
        p, w, h = idx.size(im.get("file_name", ""))
        old_img[str(im["id"])] = b.image(p.name if p else im["file_name"], w or im.get("width", 0), h or im.get("height", 0))
    old_cat = {}
    for c in d.get("categories", []):
        old_cat[str(c["id"])] = b.category(_clean_name(c["name"], a.underscores), supercategory=_clean_name(c.get("supercategory") or c["name"], a.underscores))
    kept = unlabeled = 0
    for ann in d["annotations"]:
        seg = ann.get("segmentation") or []
        if seg and not isinstance(seg[0], list):
            seg = [seg]
        iid = old_img.get(str(ann.get("image_id")))
        if "category_id" in ann:
            cid = old_cat.get(str(ann["category_id"]))
        else:                                     # VIA exports a region with no class without category_id
            cid = b.category("unlabeled"); unlabeled += 1
        if iid is None or cid is None:
            continue
        for poly in seg:
            if len(poly) >= 6:
                b.polygon(iid, cid, poly[0::2], poly[1::2]); kept += 1
    coco = b.to_dict()
    if unlabeled:
        warn(f"{unlabeled} regions had no class in VIA; they are in category 'unlabeled'; name them in the GUI")
    print(f"VIA COCO export: {kept} polygons kept. Points and polylines are not in VIA's COCO export; "
          f"convert the VIA project JSON to keep them.")
    _write(coco, a.out, a.per_image)


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Examples")[-1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, y_default, landmarks=True):
        p.add_argument("--images", help="folder with the images (sizes; y flip; name matching by stem)")
        p.add_argument("--out", required=True)
        p.add_argument("--y-origin", dest="y_origin", choices=["top", "bottom"], default=y_default,
                       help=f"where y=0 is in the INPUT/OUTPUT file (default {y_default})")
        if landmarks:
            p.add_argument("--names", help="text file, one landmark name per line, in numbering order")
            p.add_argument("--category", default="landmarks", help="COCO category for the landmarks")
        return p

    for name, fn, yd in (("tps2coco", tps2coco, "bottom"), ("morphoj2coco", morphoj2coco, "bottom"),
                         ("table2coco", table2coco, "top")):
        p = common(sub.add_parser(name), yd)
        p.add_argument("input")
        p.add_argument("--per-image", dest="per_image", help="also write one COCO file per image here")
        if name == "tps2coco":
            p.add_argument("--scale-units", dest="scale_units", choices=["auto", "per-pixel", "pixels-per-unit"],
                           default="auto", help="how to read SCALE= (tpsDig: per-pixel)")
        if name == "morphoj2coco":
            p.add_argument("--id-to-image", dest="id_to_image", help="CSV: MorphoJ ID, image file name")
        p.set_defaults(fn=fn)
    p = common(sub.add_parser("stereomorph2coco"), "top")
    p.add_argument("input", nargs="+", help="shape files or a folder of them")
    p.add_argument("--per-image", dest="per_image")
    p.set_defaults(fn=stereomorph2coco)
    for name, fn, yd in (("coco2tps", coco2tps, "bottom"), ("coco2morphoj", coco2morphoj, "bottom")):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("--out", required=True)
        p.add_argument("--images")
        p.add_argument("--y-origin", dest="y_origin", choices=["top", "bottom"], default=yd)
        p.add_argument("--category", help="only this landmark category")
        if name == "coco2tps":
            p.add_argument("--scales", help="CSV/TSV with image and px_per_mm columns (else images[].scale_px_per_mm)")
        p.set_defaults(fn=fn)
    p = common(sub.add_parser("via2coco"), "top")
    p.add_argument("input", help="VIA 2 project JSON, region-data JSON, or VIA's COCO export")
    p.add_argument("--class-attr", dest="class_attr", help="region attribute holding the class/landmark name")
    p.add_argument("--keep-spaces", dest="underscores", action="store_false",
                   help="keep spaces in category names (default: spaces -> underscores)")
    p.add_argument("--line-thickness", dest="line_thickness", type=int, default=3)
    p.add_argument("--per-image", dest="per_image")
    p.set_defaults(fn=via2coco)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
