#!/usr/bin/env python3
"""descriptron_sam3_instances.py - find every instance of a structure with SAM 3 (multi-instance prediction).

Give SAM 3 a few outlined examples (for example five setae on a wing) and/or a text prompt ("seta"); it proposes
every matching instance in each image. The result is a COCO file Descriptron opens like any other prediction, so
the taxonomist deletes false ones and adds misses; corrected files can then train the Mask R-CNN detector.

Small, thin structures disappear when a whole photograph is shrunk to SAM 3's working size (1008 px), so every
image is cut into overlapping tiles at its own resolution, each tile is prompted, masks are mapped back to the
full image, and duplicates from overlapping tiles are merged (box IoU).

Prompts per tile:
  --text "seta"               text prompt (all tiles)
  --exemplars examples.json   COCO file with outlined examples of --category; each example box is a positive
                              visual prompt on the tiles that contain it (SAM 3 box prompts work within an image)
  --boxes / --points          prompts drawn in the GUI on ONE image (x0,y0,x1,y1;... and x,y,label;...): each
                              box is an example; clicked points first outline one example (SAM 3's SAM-1-style
                              interactive mode), whose box then becomes an example
  --tile 0                    no tiling: the whole image at SAM 3's working size
Runs in its own environment (Python 3.12, PyTorch >= 2.7): Descriptron's `sam3` conda env. The checkpoints are
gated: request access at https://huggingface.co/facebook/sam3 and log in (`hf auth login`) once.

  python descriptron_sam3_instances.py --images wings/ --text seta --category seta --out setae_sam3.json
  python descriptron_sam3_instances.py --images wings/ --exemplars five_setae.json --category seta \\
      --text seta --tile 1008 --overlap 0.25 --confidence 0.5 --out setae_sam3.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_boxes(text):
    """'x0,y0,x1,y1;x0,y0,x1,y1' -> [[x0, y0, x1, y1], ...] (corners in any order)."""
    out = []
    for part in (text or "").split(";"):
        if part.strip():
            x0, y0, x1, y1 = (float(v) for v in part.split(","))
            out.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
    return out


def parse_points(text):
    """'x,y,label;...' -> ([[x, y], ...], [label, ...]); label 1 = on the structure, 0 = background."""
    pts, labs = [], []
    for part in (text or "").split(";"):
        if part.strip():
            x, y, lab = part.split(","); pts.append([float(x), float(y)]); labs.append(int(float(lab)))
    return pts, labs


def point_window(pts, w, h, size):
    """a window at native resolution around clicked points: at least size x size (smaller if the image is), centred
    on the points, inside the image. Thin structures vanish when a big photo is shrunk to SAM 3's 1008 px."""
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ww = min(w, max(size, x1 - x0 + size // 4)); hh = min(h, max(size, y1 - y0 + size // 4))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(min(max(0, cx - ww / 2), w - ww)); top = int(min(max(0, cy - hh / 2), h - hh))
    return left, top, int(ww), int(hh)


def outline_points(proc, model, im, pts, labs, size):
    """SAM 3 point outline (SAM-1 style) on a native-resolution window; returns (full-image mask, score)."""
    W, H = im.size
    pos = [p for p, l in zip(pts, labs) if l == 1] or pts
    x0, y0, ww, hh = point_window(pos, W, H, size)
    keep = [(p, l) for p, l in zip(pts, labs) if x0 <= p[0] < x0 + ww and y0 <= p[1] < y0 + hh]
    st = proc.set_image(im.crop((x0, y0, x0 + ww, y0 + hh)))
    pm, ps, _ = model.predict_inst(st, point_coords=np.array([[p[0] - x0, p[1] - y0] for p, _ in keep], float),
                                   point_labels=np.array([l for _, l in keep]), multimask_output=False)
    full = np.zeros((H, W), bool)
    m = np.squeeze(np.asarray(pm)[0])[:hh, :ww] > 0
    if m.sum() > 0.5 * ww * hh:              # the click missed the structure: SAM returns the background
        print(f"point {pos[0]}: outline covers most of its window (click on background?) - skipped", flush=True)
        return full, 0.0
    full[y0:y0 + hh, x0:x0 + ww] = m
    return full, float(np.asarray(ps).ravel()[0])


def find_bpe_vocab():
    """SAM 3's text-encoder vocabulary: sam3's own copy (git install), else the copy shipped with Descriptron
    (the sam3 0.1.4 wheel on PyPI leaves it out)."""
    name = "bpe_simple_vocab_16e6.txt.gz"
    cands = []
    try:
        import sam3
        d = os.path.dirname(sam3.__file__)
        cands += [os.path.join(d, "assets", name), os.path.join(d, "..", "assets", name)]
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, "sam3_assets", name), os.path.join(here, "..", "sam3_assets", name)]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def tiles(w, h, size, overlap):
    """top-left corners of overlapping tiles covering a w x h image (one tile if the image is small; size 0 = one)."""
    if size <= 0 or (w <= size and h <= size):
        return [(0, 0, w, h)]
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(w - size, 0) + 1, step)); ys = list(range(0, max(h - size, 0) + 1, step))
    if xs[-1] + size < w: xs.append(w - size)
    if ys[-1] + size < h: ys.append(h - size)
    return [(x, y, min(size, w), min(size, h)) for y in ys for x in xs]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge(dets, iou):
    """greedy non-maximum suppression across tiles (keeps the higher-scoring of two overlapping detections)."""
    dets = sorted(dets, key=lambda d: -d["score"]); kept = []
    for d in dets:
        if all(box_iou(d["box"], k["box"]) < iou for k in kept):
            kept.append(d)
    return kept


def mask_to_polygons(mask):
    import cv2
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.flatten().astype(float).tolist() for c in cs if len(c) >= 3]


def load_exemplars(path, category):
    """example boxes [x0, y0, x1, y1] per image file name, from a COCO file (annotations of `category`)."""
    d = json.load(open(path)); cats = {c["id"]: c["name"] for c in d["categories"]}
    names = {im["id"]: os.path.basename(im["file_name"]) for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        if category and cats.get(a["category_id"]) != category:
            continue
        x, y, w, h = a["bbox"]; out.setdefault(names[a["image_id"]], []).append([x, y, x + w, y + h])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of images (or one image file)")
    ap.add_argument("--text", default=None, help="text prompt, e.g. seta")
    ap.add_argument("--exemplars", default=None, help="COCO file with a few outlined examples (positive box prompts)")
    ap.add_argument("--boxes", default=None, help="example boxes on a single image: x0,y0,x1,y1;...")
    ap.add_argument("--points", default=None, help="clicked points on a single image: x,y,label;... (outline one example)")
    ap.add_argument("--points_mode", default="example", choices=["example", "each", "examples"],
                    help="example: all points outline ONE example and SAM 3 finds all like it; each: one mask per "
                         "positive point (negative points shared), no search; examples: one outline per positive "
                         "point, then all outlines together are the examples for the search")
    ap.add_argument("--point_window", type=int, default=1008,
                    help="clicked points are outlined on a window of this many native pixels around them (default 1008)")
    ap.add_argument("--category", default=None, help="category name to write (and to read from --exemplars)")
    ap.add_argument("--tile", type=int, default=1008, help="tile size in pixels (SAM 3 works at 1008; 0 = whole image, no tiling)")
    ap.add_argument("--overlap", type=float, default=0.25, help="tile overlap fraction (default 0.25)")
    ap.add_argument("--confidence", type=float, default=0.5, help="keep detections at or above this score")
    ap.add_argument("--merge_iou", type=float, default=0.5, help="box IoU above which tile duplicates merge")
    ap.add_argument("--min_area", type=int, default=4, help="drop masks smaller than this many pixels")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output COCO json")
    a = ap.parse_args(argv)
    if not (a.text or a.exemplars or a.boxes or a.points):
        ap.error("give --text, --exemplars, --boxes and/or --points")
    if a.points_mode in ("each", "examples") and not a.points:
        ap.error("--points_mode each needs --points")
    if (a.boxes or a.points) and not Path(a.images).is_file():
        ap.error("--boxes / --points are drawn on one image: give that image file as --images")
    cat_name = a.category or a.text or "instance"
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    try:
        model = build_sam3_image_model(bpe_path=find_bpe_vocab(), enable_inst_interactivity=bool(a.points))
    except Exception as e:
        sys.exit(f"Could not load SAM 3 ({e}).\nThe checkpoints are gated: request access at "
                 "https://huggingface.co/facebook/sam3, then run `hf auth login` once in the sam3 environment.")
    proc = Sam3Processor(model, resolution=a.tile if a.tile > 0 else 1008, device=a.device,
                         confidence_threshold=a.confidence)
    ex = load_exemplars(a.exemplars, a.category) if a.exemplars else {}
    p = Path(a.images); files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": cat_name}],
            "info": {"description": "SAM 3 multi-instance proposals (descriptron_sam3_instances.py); review before use",
                     "text_prompt": a.text, "exemplars": a.exemplars, "tile": a.tile, "overlap": a.overlap,
                     "confidence": a.confidence}}
    t0 = time.time(); aid = 0
    for k, f in enumerate(files, 1):
        im = Image.open(f).convert("RGB"); W, H = im.size
        coco["images"].append({"id": k, "file_name": f.name, "width": W, "height": H})
        boxes_here = ex.get(f.name, []) + parse_boxes(a.boxes)
        dets = []
        if a.points and a.points_mode in ("each", "examples"):   # one structure per positive click
            pts, labs = parse_points(a.points)
            neg = [q for q, l in zip(pts, labs) if l == 0]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                for q, l in zip(pts, labs):
                    if l != 1:
                        continue
                    m, sc = outline_points(proc, model, im, [q] + neg, [1] + [0] * len(neg), a.point_window)
                    if m.sum() > 0.5 * W * H:              # a click on the background returns the background
                        print(f"point {q}: mask covers most of the image (background?) - skipped", flush=True)
                    elif m.sum() >= a.min_area:
                        ys_, xs_ = np.nonzero(m)
                        dets.append({"mask": m, "score": sc,
                                     "box": [float(xs_.min()), float(ys_.min()), float(xs_.max() + 1), float(ys_.max() + 1)]})
            if a.points_mode == "each":                # nothing else
                boxes_here = []
                tile_list = []
            else:                                      # "examples": every outline's box is an example, together
                boxes_here += [d["box"] for d in dets]
                print(f"{len(dets)} clicked outline(s) -> examples", flush=True)
                tile_list = tiles(W, H, a.tile, a.overlap)
        else:
            tile_list = tiles(W, H, a.tile, a.overlap)
        if a.points and a.points_mode == "example":   # clicked points -> one outlined example (whole image) -> its box
            pts, labs = parse_points(a.points)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                m, _sc = outline_points(proc, model, im, pts, labs, a.point_window)
            if m.any():
                ys_, xs_ = np.nonzero(m)
                boxes_here.append([float(xs_.min()), float(ys_.min()), float(xs_.max() + 1), float(ys_.max() + 1)])
                print(f"points -> example box {boxes_here[-1]}", flush=True)
            else:
                print("points gave no example mask", flush=True)
        for (x0, y0, tw, th) in tile_list:
            tile = im.crop((x0, y0, x0 + tw, y0 + th))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                st = proc.set_image(tile)
                out = None
                if a.text:
                    out = proc.set_text_prompt(prompt=a.text, state=st)
                for (bx0, by0, bx1, by1) in boxes_here:             # examples inside this tile only
                    cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x0 + tw), min(by1, y0 + th)
                    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
                        continue
                    nb = [((cx0 + cx1) / 2 - x0) / tw, ((cy0 + cy1) / 2 - y0) / th, (cx1 - cx0) / tw, (cy1 - cy0) / th]
                    out = proc.add_geometric_prompt(box=nb, label=True, state=st)
                if out is None:
                    continue
            masks = out["masks"].float().cpu().numpy(); scores = out["scores"].float().cpu().numpy()
            boxes = out["boxes"].float().cpu().numpy()
            for m, s, b in zip(masks, scores, boxes):
                m = np.squeeze(m) > 0.5
                if m.sum() < a.min_area:
                    continue
                full = np.zeros((H, W), bool); full[y0:y0 + th, x0:x0 + tw] = m[:th, :tw]
                dets.append({"mask": full, "score": float(s),
                             "box": [float(b[0] + x0), float(b[1] + y0), float(b[2] + x0), float(b[3] + y0)]})
        kept = dets if a.points_mode == "each" and a.points else merge(dets, a.merge_iou)
        for dd in kept:
            polys = mask_to_polygons(dd["mask"])
            if not polys:
                continue
            ys_, xs_ = np.nonzero(dd["mask"]); aid += 1
            coco["annotations"].append({"id": aid, "image_id": k, "category_id": 1, "segmentation": polys,
                                        "area": float(dd["mask"].sum()), "iscrowd": 0, "score": dd["score"],
                                        "bbox": [float(xs_.min()), float(ys_.min()), float(xs_.max() - xs_.min() + 1),
                                                 float(ys_.max() - ys_.min() + 1)]})
        print(f"[{k}/{len(files)}] {f.name}: {len(kept)} instances", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(a.out, "w"))
    print(f"wrote {a.out}: {aid} instances in {len(files)} images ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
