#!/usr/bin/env python
"""BioRAG Annotate - DINOv3 patch-token extraction (v1).

Run as a subprocess by the BioRAG Annotate window (descriptron-v2-v89.py), in an env whose
transformers knows DINOv3 (the biorag env). One .npz per image in --cache_dir:

    tokens    (gh*gw, D) float16   patch tokens of layer --layer (register / CLS tokens dropped)
    grid      (gh, gw)             patch grid
    image_hw  (H, W)               original image size
    work_hw   (nh, nw)             size the image was resized to (multiples of 16)
    scale     (sy, sx)             nh / H, nw / W
    layer, size, model, source, source_size, source_mtime

Resolution rule (AVD): the image long side is resized to --size (1024 / 1536 / 2048) with cubic
interpolation - small images are UPSAMPLED, never kept small - and both sides are snapped to
multiples of 16. Images whose long side exceeds --tile are processed in overlapping tiles
(128 px overlap, tile centres kept), as in measure/biorag_probe_v1.py.

Progress goes to stdout as one line per image:
    BRA_PROGRESS<TAB>i<TAB>n<TAB>ok|cached|failed<TAB>seconds<TAB>file name<TAB>message
and a final line  BRA_DONE <n_ok> <n_failed>.

The pure helpers (load_rgb8, working_size, cache_path) import without torch, so the GUI side can
use them too.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CACHE_VERSION = 1


# ---------------------------------------------------------------- pure helpers (no torch)
def _to_uint8(arr):
    """Any numeric array -> uint8. 8-bit data is kept as is; deeper data is scaled by its maximum."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    mx = float(np.nanmax(a)) if a.size else 0.0
    if mx <= 0:
        return np.zeros(a.shape, np.uint8)
    if mx <= 1.0 and a.dtype.kind == "f":
        mx = 1.0
    elif mx <= 255:
        mx = 255.0
    return np.clip(a / mx * 255.0 + 0.5, 0, 255).astype(np.uint8)


def load_rgb8(path):
    """Image file -> HxWx3 uint8 RGB. Same pixel grid as the GUI (PIL, no EXIF rotation).
    8-bit images: PIL convert('RGB') exactly as the GUI does (RGBA drops alpha).
    16-bit / 32-bit / float images (where convert('RGB') would clip): scaled to 8 bit by their max.
    Falls back to OpenCV for files PIL cannot open (e.g. 48-bit RGB tif)."""
    from PIL import Image
    try:
        im = Image.open(path)
        if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            a = _to_uint8(np.array(im))
            return np.ascontiguousarray(np.stack([a] * 3, -1))
        return np.asarray(im.convert("RGB"))
    except Exception:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise
        a = _to_uint8(a)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        elif a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGB)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(a)


def working_size(h, w, long_side):
    """(H, W) -> (nh, nw): long side resized to long_side (up OR down), both sides multiples of 16."""
    s = float(long_side) / max(h, w)
    nh = max(16, int(round(h * s / 16.0)) * 16)
    nw = max(16, int(round(w * s / 16.0)) * 16)
    return nh, nw


def cache_path(cache_dir, image_path, layer, size, box=None):
    """Cache file of one image (or of one crop box x0,y0,x1,y1 of it) for one (layer, size). The full
    file name (with extension) is kept, so a.tif and a.jpg never share a cache file."""
    crop = "" if box is None else "__crop_%d_%d_%d_%d" % tuple(int(v) for v in box)
    return os.path.join(cache_dir, f"{os.path.basename(image_path)}{crop}__L{int(layer)}_s{int(size)}.npz")


MAX_CROP_SCALE = 8.0


def crop_working_size(h, w, long_side, max_scale=MAX_CROP_SCALE):
    """Crop (h, w) -> working size: long side resized to long_side (cubic UPsampling for small crops),
    but at most max_scale times the crop's own pixels; sides multiples of 16."""
    s = min(float(long_side) / max(h, w), float(max_scale))
    return max(16, int(round(h * s / 16.0)) * 16), max(16, int(round(w * s / 16.0)) * 16)


def parse_list_size(line):
    """Optional third field of a list line: this crop's own size (long side, px), e.g. to extract a panned
    view at the same scale as the region it came from. None if absent."""
    parts = line.rstrip("\n").split("\t")
    return int(parts[2]) if len(parts) > 2 and parts[2].strip() else None


def parse_list_line(line):
    """'path' or 'path<TAB>x0,y0,x1,y1' -> (path, box or None)."""
    parts = line.rstrip("\n").split("\t")
    box = None
    if len(parts) > 1 and parts[1].strip():
        box = tuple(int(round(float(v))) for v in parts[1].split(","))
    return parts[0], box


def cache_is_valid(npz_path, image_path, layer, size, model):
    """True if the cache file exists and was made from this image file with these settings."""
    if not os.path.isfile(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            st = os.stat(image_path)
            return (int(z["layer"]) == int(layer) and int(z["size"]) == int(size)
                    and str(z["model"]) == str(model)
                    and int(z["source_size"]) == int(st.st_size)
                    and abs(float(z["source_mtime"]) - float(st.st_mtime)) < 1.0
                    and int(z.get("version", 0)) == CACHE_VERSION)
    except Exception:
        return False


def save_cache(npz_path, tokens, grid, image_hw, work_hw, layer, size, model, image_path, box=None):
    st = os.stat(image_path)
    if box is None:
        box = (0, 0, int(image_hw[1]), int(image_hw[0]))
    tmp = npz_path + ".tmp.npz"
    np.savez(tmp, tokens=np.asarray(tokens, np.float16), grid=np.asarray(grid, np.int32),
             image_hw=np.asarray(image_hw, np.int32), work_hw=np.asarray(work_hw, np.int32),
             scale=np.array([work_hw[0] / image_hw[0], work_hw[1] / image_hw[1]], np.float64),
             layer=np.int32(layer), size=np.int32(size), model=np.str_(model),
             source=np.str_(os.path.abspath(image_path)), source_size=np.int64(st.st_size),
             source_mtime=np.float64(st.st_mtime), version=np.int32(CACHE_VERSION),
             box=np.asarray(box, np.int32))
    os.replace(tmp, npz_path)


def load_cache(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- extraction (torch)
class Extractor:
    def __init__(self, model_id, device, layer, tile=2048, overlap=128):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.dev = device
        self.net = AutoModel.from_pretrained(model_id).to(device).eval()
        self.dim = self.net.config.hidden_size
        self.nreg = getattr(self.net.config, "num_register_tokens", 4)
        self.layer, self.tile, self.ov = int(layer), int(tile), int(overlap)

    def _one(self, x):
        torch = self.torch
        th, tw = x.shape[0] // 16, x.shape[1] // 16
        px = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
            hs = self.net(pixel_values=px, output_hidden_states=True).hidden_states
        t = hs[self.layer + 1][0, 1 + self.nreg:, :].float().reshape(th, tw, self.dim)
        out = t.cpu().numpy()
        del hs, t, px
        return out

    def grid(self, x):
        """x: normalised HxWx3, multiples of 16 -> [gh, gw, D] float32."""
        H, W = x.shape[:2]
        gh, gw = H // 16, W // 16
        if H <= self.tile and W <= self.tile:
            return self._one(x)
        OV, STEP = self.ov, self.tile - 2 * self.ov
        out = np.zeros((gh, gw, self.dim), np.float32)
        for y0 in range(0, H, STEP):
            for x0 in range(0, W, STEP):
                ya, xa = max(0, y0 - OV), max(0, x0 - OV)
                t = x[ya:min(H, y0 + STEP + OV), xa:min(W, x0 + STEP + OV)]
                t = t[:t.shape[0] // 16 * 16, :t.shape[1] // 16 * 16]
                T = self._one(t)
                ky, kx = (y0 - ya) // 16, (x0 - xa) // 16
                ny, nx = min(STEP, H - y0) // 16, min(STEP, W - x0) // 16
                out[y0 // 16:y0 // 16 + ny, x0 // 16:x0 // 16 + nx] = T[ky:ky + ny, kx:kx + nx]
        return out


def tile_edges(n, size, anchor):
    """Tile boundaries along one axis of length n (multiples of 16): tiles of `size` laid so that one tile
    starts at `anchor`; partial tiles at both ends."""
    a = int(anchor) % int(size)
    edges = sorted({0, int(n)} | set(range(a, int(n), int(size))))
    return [(e0, e1) for e0, e1 in zip(edges[:-1], edges[1:]) if e1 > e0]


def grid_tiled_like(ex, x, th, tw, ay, ax):
    """Tokens of x (multiples of 16) in NON-overlapping tiles of the zoomed region's own size (th x tw), laid
    so that one tile coincides with the region (anchor ay, ax; multiples of 16). The region's tokens are then
    reproduced (same pixels, same patch grid phase, same context) and the other tiles see the same kind of
    context as the region did (18 Oct: DINOv3 tokens changed by cosine ~0.13 when only the context window
    changed, and ~0.11 with a half-patch grid shift)."""
    H, W = x.shape[:2]
    out = np.zeros((H // 16, W // 16, ex.dim), np.float32)
    for y0, y1 in tile_edges(H, th, ay):
        for x0, x1 in tile_edges(W, tw, ax):
            out[y0 // 16:y1 // 16, x0 // 16:x1 // 16] = ex._one(np.ascontiguousarray(x[y0:y1, x0:x1]))
    return out


def extract_one(ex, image_path, size, box=None):
    """Whole image (box None) or the crop box x0,y0,x1,y1 (upsampled, at most MAX_CROP_SCALE x).
    The returned (H, W) is always the whole image's size."""
    import cv2
    rgb = load_rgb8(image_path)
    H, W = rgb.shape[:2]
    if box is not None:
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError(f"crop box {box} is empty")
        rgb = rgb[y0:y1, x0:x1]
        nh, nw = crop_working_size(y1 - y0, x1 - x0, size)
    else:
        nh, nw = working_size(H, W, size)
    rgb_ = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (rgb_.astype(np.float32) / 255.0 - MEAN) / STD
    G = ex.grid(x)
    gh, gw = G.shape[:2]
    return G.reshape(gh * gw, -1).astype(np.float16), (gh, gw), (H, W), (nh, nw)


def _ui_module():
    """biorag_annotate_v1 (or biorag_annotate_v1_dev when this file is the _dev copy) for its pure helpers."""
    import importlib
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sfx = "_dev" if os.path.basename(__file__).endswith("_dev.py") else ""
    return importlib.import_module("biorag_annotate_v1" + sfx)


# ---------------------------------------------------------------- trained DINOMask head (optional view)
# A DINOMask run folder holds head.pt + results.json (classes). The head was trained on the LAST-layer
# tokens (last_hidden_state) of DINOv3 ViT-L/16 at the run's size (2048 by default) with 4-shift
# averaging, so its features are computed here exactly like that (dinomask_predict_rot4_v1.py probs(),
# without the test-time rotation) - NOT from the layer-18 token cache.
#   classes  : argmax of the guided-filtered 4-shift-averaged probabilities, uint8, at the working size
#   hidden   : the head's 512-d activations after GELU, (gh*gw, 512) float16, from the UNSHIFTED pass
def dinomask_working_size(h, w, long_side):
    """As DINOMask probs(): s = size / max(H, W); sides floored to multiples of 16."""
    s = float(long_side) / max(h, w)
    return max(16, int(h * s) // 16 * 16), max(16, int(w * s) // 16 * 16)


def dinomask_run_id(run_dir):
    st = os.stat(os.path.join(run_dir, "head.pt"))
    import hashlib
    return hashlib.md5(f"{os.path.abspath(run_dir)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:10]


def dinomask_cache_path(cache_dir, image_path, run_dir, size):
    return os.path.join(cache_dir, f"{os.path.basename(image_path)}__dinomask_{dinomask_run_id(run_dir)}_s{int(size)}.npz")


def dinomask_classes(run_dir):
    import ast
    c = json.load(open(os.path.join(run_dir, "results.json")))["classes"]
    return ast.literal_eval(c) if isinstance(c, str) else list(c)


def dinomask_cache_is_valid(npz_path, image_path):
    if not os.path.isfile(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            st = os.stat(image_path)
            return int(z["source_size"]) == int(st.st_size) and abs(float(z["source_mtime"]) - float(st.st_mtime)) < 1.0
    except Exception:
        return False


class DinoMaskRunner:
    def __init__(self, run_dir, model_id, device):
        import torch
        import torch.nn as nn
        from transformers import AutoModel
        self.torch, self.dev = torch, device
        self.classes = dinomask_classes(run_dir)
        K = len(self.classes)
        self.net = AutoModel.from_pretrained(model_id).to(device).eval()
        self.nreg = self.net.config.num_register_tokens
        D = self.net.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, K)).to(device)
        sd = torch.load(os.path.join(run_dir, "head.pt"), map_location=device)
        self.head.load_state_dict({k.replace("net.", ""): v for k, v in sd.items()})
        self.head.eval()
        self.K = K

    def run(self, rgb, size):
        import cv2
        gfilter = _ui_module().gfilter
        torch = self.torch
        H, W = rgb.shape[:2]
        nh, nw = dinomask_working_size(H, W, size)
        rgb_ = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        base = (rgb_.astype(np.float32) / 255 - MEAN) / STD
        acc = np.zeros((nh, nw, self.K), np.float32)
        hidden = None
        for dx, dy in ((0, 0), (8, 0), (0, 8), (8, 8)):
            x = np.pad(base, ((dy, 16 - dy if dy else 0), (dx, 16 - dx if dx else 0), (0, 0)), mode="edge")
            gh, gw = x.shape[0] // 16, x.shape[1] // 16
            px = torch.from_numpy(x[:gh * 16, :gw * 16].transpose(2, 0, 1).copy())[None].to(self.dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                T = self.net(pixel_values=px).last_hidden_state[0, 1 + self.nreg:].float()
            with torch.no_grad():
                h = self.head[2](self.head[1](self.head[0](T)))          # LayerNorm -> Linear -> GELU
                pr = torch.softmax(self.head[4](h), -1).cpu().numpy().reshape(gh, gw, self.K)
            if dx == 0 and dy == 0:
                hidden = (h.cpu().numpy().astype(np.float16), (gh, gw))
            acc += cv2.resize(pr, (gw * 16, gh * 16), interpolation=cv2.INTER_LINEAR)[dy:dy + nh, dx:dx + nw]
        P = gfilter(cv2.cvtColor(rgb_, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255, acc / 4)
        return P.argmax(-1).astype(np.uint8), hidden[0], hidden[1], (H, W), (nh, nw)


def main_dinomask(a, paths):
    import json as _json  # noqa: F401
    os.makedirs(a.cache_dir, exist_ok=True)
    n = len(paths)
    todo, n_ok, n_fail = [], 0, 0
    for i, p in enumerate(paths, 1):
        cp = dinomask_cache_path(a.cache_dir, p, a.dinomask_run, a.size)
        if not a.force and dinomask_cache_is_valid(cp, p):
            n_ok += 1
            print(f"BRA_PROGRESS\t{i}\t{n}\tcached\t0.0\t{os.path.basename(p)}\t", flush=True)
        else:
            todo.append((i, p, cp))
    if todo:
        dev = a.device
        if dev == "auto":
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BRA_INFO loading DINOMask head {a.dinomask_run} + {a.model} on {dev} (last layer, size {a.size})", flush=True)
        try:
            R = DinoMaskRunner(a.dinomask_run, a.model, dev)
        except Exception as e:
            for i, p, _ in todo:
                print(f"BRA_PROGRESS\t{i}\t{n}\tfailed\t0.0\t{os.path.basename(p)}\tDINOMask: {e}", flush=True)
            print(f"BRA_DONE {n_ok} {len(todo)}", flush=True)
            return 1
        for i, p, cp in todo:
            t0 = time.time()
            try:
                lab, hid, grid, hw, whw = R.run(load_rgb8(p), a.size)
                st = os.stat(p)
                tmp = cp + ".tmp.npz"
                np.savez(tmp, classes_map=lab, hidden=hid, grid=np.asarray(grid, np.int32),
                         image_hw=np.asarray(hw, np.int32), work_hw=np.asarray(whw, np.int32),
                         classes=np.array(R.classes), run=np.str_(os.path.abspath(a.dinomask_run)),
                         size=np.int32(a.size), model=np.str_(a.model), source_size=np.int64(st.st_size),
                         source_mtime=np.float64(st.st_mtime))
                os.replace(tmp, cp)
                n_ok += 1
                print(f"BRA_PROGRESS\t{i}\t{n}\tok\t{time.time() - t0:.1f}\t{os.path.basename(p)}\t", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"BRA_PROGRESS\t{i}\t{n}\tfailed\t{time.time() - t0:.1f}\t{os.path.basename(p)}\t{str(e)[:200]}", flush=True)
        if dev == "cuda":
            print(f"BRA_INFO peak GPU memory {R.torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"BRA_DONE {n_ok} {n_fail}", flush=True)
    return 0 if n_fail == 0 else 2


# ---------------------------------------------------------------- apply a selection rule at fine scale (18 Oct)
# The rule (biorag_annotate_v1.save_rule) was built on zoomed-region tokens. Here every target image is
# tokenised WHOLE at the SAME scale as that region (tiled, --tile px with 128 px overlap), the rule is applied,
# and only the per-patch result (selected bool + score float16) is written; the fine tokens are not cached
# (a 5400 x 3600 image at 1.5x is ~160k tokens, ~330 MB as float16).
def applied_cache_path(cache_dir, image_path, rule_id):
    return os.path.join(cache_dir, f"{os.path.basename(image_path)}__applied_{rule_id}.npz")


def fine_working_size(h, w, sy, sx):
    """Whole image at the region's scale (sy, sx = working px per image px), sides multiples of 16."""
    return max(16, int(round(h * float(sy) / 16.0)) * 16), max(16, int(round(w * float(sx) / 16.0)) * 16)


def _rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except Exception:
        return float("nan")


def main_apply(a, paths):
    import cv2
    load_rule, rule_scores = _ui_module().load_rule, _ui_module().rule_scores
    rule = load_rule(a.apply_rule)
    rid = rule["rule_id"]
    sy, sx = (float(v) for v in rule["scale"])
    th, tw = (int(v) for v in rule.get("region_work_hw", (a.tile, a.tile)))
    src = rule.get("from_region") or [None, None]
    src_name, src_box = src[0], (tuple(int(v) for v in src[1]) if src[1] is not None else None)
    os.makedirs(a.cache_dir, exist_ok=True)
    n = len(paths)
    todo, n_ok, n_fail = [], 0, 0
    for i, p in enumerate(paths, 1):
        cp = applied_cache_path(a.cache_dir, p, rid)
        if not a.force and dinomask_cache_is_valid(cp, p):      # same source-file check
            n_ok += 1
            print(f"BRA_PROGRESS\t{i}\t{n}\tcached\t0.0\t{os.path.basename(p)}\t", flush=True)
        else:
            todo.append((i, p, cp))
    if todo:
        dev = a.device
        if dev == "auto":
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        layer = int(rule["layer"])
        print(f"BRA_INFO rule {rule['kind']} ({rid}): loading {a.model} on {dev} (layer {layer}, scale {sx:.3f})", flush=True)
        try:
            ex = Extractor(a.model, dev, layer, tile=a.tile)
        except Exception as e:
            for i, p, _ in todo:
                print(f"BRA_PROGRESS\t{i}\t{n}\tfailed\t0.0\t{os.path.basename(p)}\tmodel: {e}", flush=True)
            print(f"BRA_DONE {n_ok} {len(todo)}", flush=True)
            return 1
        for i, p, cp in todo:
            t0 = time.time()
            try:
                rgb = load_rgb8(p)
                H, W = rgb.shape[:2]
                nh, nw = fine_working_size(H, W, sy, sx)
                x = (cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0 - MEAN) / STD
                del rgb
                # tile like the region: on the region's own image the patch grid is put in phase with the region
                # (the <= 15 fine px before the first full patch are left out) and one tile IS the region
                ox = oy = 0
                if os.path.basename(p) == src_name and src_box is not None:
                    ox, oy = int(round(src_box[0] * nw / W)), int(round(src_box[1] * nh / H))
                px, py = ox % 16, oy % 16
                Hx, Wx = (nh - py) // 16 * 16, (nw - px) // 16 * 16
                G = grid_tiled_like(ex, x[py:py + Hx, px:px + Wx], th, tw, oy - py, ox - px)
                del x
                gh, gw = G.shape[:2]
                box = (int(round(px * W / nw)), int(round(py * H / nh)),
                       min(W, int(round((px + Wx) * W / nw))), min(H, int(round((py + Hx) * H / nh))))
                t1 = time.time()
                sel, score = rule_scores(G.reshape(gh * gw, -1), (gh, gw), rule)
                del G
                st = os.stat(p)
                tmp = cp + ".tmp.npz"
                np.savez(tmp, sel=sel.reshape(gh, gw), score=score.reshape(gh, gw).astype(np.float16),
                         grid=np.asarray((gh, gw), np.int32), image_hw=np.asarray((H, W), np.int32),
                         work_hw=np.asarray((Hx, Wx), np.int32), rule_id=np.str_(rid), box=np.asarray(box, np.int32),
                         source_size=np.int64(st.st_size), source_mtime=np.float64(st.st_mtime))
                os.replace(tmp, cp)
                n_ok += 1
                print(f"BRA_INFO {os.path.basename(p)}: {gh}x{gw} fine patches ({Wx}x{Hx} px, tiles {tw}x{th}"
                      f"{', anchored on the region' if ox or oy else ''}), tokens "
                      f"{t1 - t0:.1f} s, rule {time.time() - t1:.1f} s, {int(sel.sum())} selected, peak RSS "
                      f"{_rss_gb():.2f} GB", flush=True)
                print(f"BRA_PROGRESS\t{i}\t{n}\tok\t{time.time() - t0:.1f}\t{os.path.basename(p)}\t", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"BRA_PROGRESS\t{i}\t{n}\tfailed\t{time.time() - t0:.1f}\t{os.path.basename(p)}\t{str(e)[:200]}", flush=True)
            if dev == "cuda":
                try:
                    ex.torch.cuda.empty_cache()
                except Exception:
                    pass
        if dev == "cuda":
            print(f"BRA_INFO peak GPU memory {ex.torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    print(f"BRA_DONE {n_ok} {n_fail}", flush=True)
    return 0 if n_fail == 0 else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[], help="image files")
    ap.add_argument("--image_list", default="", help="text file, one image path per line (alternative to --images)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--size", type=int, default=1536, help="long side in px (1024 / 1536 / 2048)")
    ap.add_argument("--layer", type=int, default=18, help="0-based transformer block, 0-23 for ViT-L (e.g. 12 / 18 / 23)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--tile", type=int, default=2048, help="tile larger working images (px)")
    ap.add_argument("--force", action="store_true", help="re-extract even if a valid cache exists")
    ap.add_argument("--apply_rule", default="", help="selection rule npz (BioRAG Annotate): tokenise each image "
                    "whole at the rule's scale and write only the per-patch result")
    ap.add_argument("--dinomask_run", default="", help="DINOMask run folder (head.pt + results.json): compute its "
                    "class map + hidden activations instead of tokens (last layer, --size, 4-shift averaging)")
    a = ap.parse_args(argv)

    items = [(p, None) for p in a.images]
    sizes = [None] * len(items)
    if a.image_list:
        with open(a.image_list) as fh:
            lines = [l for l in fh if l.strip()]
        items += [parse_list_line(l) for l in lines]
        sizes += [parse_list_size(l) for l in lines]
    paths = [p for p, _ in items]
    if not paths:
        print("BRA_DONE 0 0", flush=True)
        return 0
    if a.dinomask_run:
        return main_dinomask(a, paths)
    if a.apply_rule:
        return main_apply(a, paths)
    os.makedirs(a.cache_dir, exist_ok=True)
    todo = []
    n = len(paths)
    for i, ((p, box), sz) in enumerate(zip(items, sizes), 1):
        sz = sz or a.size
        cp = cache_path(a.cache_dir, p, a.layer, sz, box)
        if not a.force and cache_is_valid(cp, p, a.layer, sz, a.model):
            print(f"BRA_PROGRESS\t{i}\t{n}\tcached\t0.0\t{os.path.basename(p)}\t", flush=True)
        else:
            todo.append((i, p, cp, box, sz))
    n_ok, n_fail = n - len(todo), 0
    if todo:
        dev = a.device
        if dev == "auto":
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BRA_INFO loading {a.model} on {dev} (layer {a.layer}, size {a.size})", flush=True)
        try:
            ex = Extractor(a.model, dev, a.layer, tile=a.tile)
        except Exception as e:
            for i, p, _, _, _ in todo:
                print(f"BRA_PROGRESS\t{i}\t{n}\tfailed\t0.0\t{os.path.basename(p)}\tmodel: {e}", flush=True)
            print(f"BRA_DONE {n_ok} {len(todo)}", flush=True)
            return 1
        for i, p, cp, box, sz in todo:
            t0 = time.time()
            try:
                T, grid, hw, whw = extract_one(ex, p, sz, box)
                save_cache(cp, T, grid, hw, whw, a.layer, sz, a.model, p, box)
                n_ok += 1
                print(f"BRA_PROGRESS\t{i}\t{n}\tok\t{time.time() - t0:.1f}\t{os.path.basename(p)}\t", flush=True)
            except Exception as e:
                n_fail += 1
                msg = str(e).replace("\n", " ")[:200]
                print(f"BRA_PROGRESS\t{i}\t{n}\tfailed\t{time.time() - t0:.1f}\t{os.path.basename(p)}\t{msg}", flush=True)
                if dev == "cuda":
                    try:
                        ex.torch.cuda.empty_cache()
                    except Exception:
                        pass
        if dev == "cuda":
            try:
                print(f"BRA_INFO peak GPU memory {ex.torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
            except Exception:
                pass
    print(f"BRA_DONE {n_ok} {n_fail}", flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
