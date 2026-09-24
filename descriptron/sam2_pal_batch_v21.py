#!/usr/bin/env python3
"""
SAM2-PAL: Palindrome-based Mask Propagation for Descriptron
Version 19 - RLE segmentation support

=== CHANGES IN V19 ===

1. FIXED: RLE segmentations loaded as blank masks (silently).
   load_template_from_coco() and load_training_templates() both did:
       for seg in ann['segmentation']:
           if isinstance(seg, list) and len(seg) >= 6:
   Iterating a dict yields its KEYS ('size', 'counts'), which are strings, so an
   RLE annotation produced an all-zero mask, "Loaded 0 masks from COCO JSON",
   and exit code 0 -- indistinguishable from a bad annotation.

   This matters because COCO polygons cannot represent holes: pycocotools unions
   multi-ring polygons, and the per-ring fillPoly here does the same. Descriptron
   therefore writes RLE whenever a region was erased inside a mask (e.g. the
   paper point mount visible between an ant's legs in profile view), and those
   templates could not be read at all.

   Both call sites now go through _coco_segmentation_to_mask(), which accepts
   uncompressed RLE (Descriptron), compressed RLE (pycocotools) and polygons.
   Polygon behaviour is byte-for-byte unchanged.

=== CHANGES IN V18 (CRITICAL BUG FIXES) ===

1. FIXED: CUDA OOM during inference (propagate_masks)
   - Added CPU offloading (offload_video_to_cpu=True, offload_state_to_cpu=True)
     to init_state(). Training used manual track_step on 2 images at a time, so
     it fitted in VRAM fine. But inference called init_state() on ALL frames
     (e.g. 489 for 244 targets with interleaving), loading everything onto GPU.
   - Removed bulk pre-anchoring of template masks at every interleaved frame.
     propagate_in_video_preflight() consolidates ALL pre-anchored masks at once
     via torch.nn.functional.interpolate on GPU — with ~245 pre-anchored frames
     this was the direct OOM trigger (line 520 in sam2_video_predictor.py).

2. NEW: 4-stage cycle-consistent inference (--cycle_consistency)
   - Forward propagation → save masks to disk
   - Memory reset (prevents state accumulation)
   - Backward propagation (reverse pass)
   - IoU comparison between passes → confidence scores per frame
   - Low-IoU frames flagged as potential tracking drift

3. NEW: Chunked inference mode (--chunk_size N)
   - Process images in chunks of N (default: all at once with CPU offloading)
   - Each chunk gets its own init_state + reset cycle
   - Prevents unbounded memory growth for very large datasets (1000+ images)

4. NEW: Load existing checkpoint for inference-only (--load_checkpoint)
   - Skip training entirely, load a previously fine-tuned .pt file
   - Works with any checkpoint produced by --pal_finetuning or --finetune
   - The SAM2 architecture yaml (--sam2_config) does NOT change between
     original and fine-tuned — only weights differ

5. All v17 features preserved:
   - Multi-mask training from COCO JSON
   - Optional LoRA fine-tuning (requires 'peft' library)
   - 4-step OC-CCL with memory reset
   - Multi-template training from separate training JSON

=== MULTI-MASK WORKFLOW ===

Training with multi-mask JSON:
    python sam2_pal_batch_v17.py \\
        --template_json masks.json \\        # Contains scape, antenna, eye
        --template_image template.jpg \\
        --image_dir ./specimens \\
        --output_dir ./output \\
        --pal_finetuning --num_epochs 50

This will:
1. Load all 3 masks from template image
2. Train on ALL masks (cycles through scape, antenna, eye)
3. Predict all 3 structures on target images

=== PAPER'S OC-CCL RECIPE (arxiv.org/abs/2501.06749) ===

Training setup:
  - 1 labeled image (x0, y0) - we support multiple!
  - 100 unlabeled images - disjoint from test set
  - LoRA fine-tuning of decoder + memory encoder

4-frame palindrome: {x0, x1, x1â€ , x0â€ }

Phase 1:
  Frame 0: x0 with mask prompt y0 â†' stored in memory
  Frame 1: x1 (unlabeled) â†' predict Å·1 using memory

*** MEMORY RESET (prevents cheating by remembering y0) ***

Phase 2:
  Frame 2: x1â€  with Å·1 as DIFFERENTIABLE prompt
  Frame 3: x0â€  â†' predict Å·0â€ , compute loss vs y0

Loss: BCE + Dice between Å·0â€  and y0

=== USAGE ===

    # With LoRA (recommended if peft installed)
    pip install peft
    python sam2_pal_batch.py --template_mask mask.png \\
                             --template_image template.jpg \\
                             --image_dir ./images \\
                             --output_dir ./output \\
                             --pal_finetuning --use_lora \\
                             --num_epochs 25 --learning_rate 1e-4

    # Without LoRA (full fine-tuning)
    python sam2_pal_batch.py --template_mask mask.png \\
                             --template_image template.jpg \\
                             --image_dir ./images \\
                             --output_dir ./output \\
                             --pal_finetuning \\
                             --num_epochs 75 --learning_rate 5e-6

Author: Descriptron Project (2025)
"""

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_image_and_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to simulate variations across specimens.
    This helps the fine-tuned model generalize better.
    """
    h, w = image.shape[:2]
    
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    # Random vertical flip (50% chance)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    
    # Random rotation (-25 to +25 degrees)
    angle = random.uniform(-25, 25)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Random scale (0.85 to 1.15)
    scale = random.uniform(0.85, 1.15)
    new_w, new_h = int(w * scale), int(h * scale)
    image_scaled = cv2.resize(image, (new_w, new_h))
    mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Crop or pad back to original size
    if scale > 1:
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        image = image_scaled[start_y:start_y+h, start_x:start_x+w]
        mask = mask_scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        pad_image = np.zeros((h, w, 3), dtype=image.dtype)
        pad_mask = np.zeros((h, w), dtype=mask.dtype)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        pad_image[start_y:start_y+new_h, start_x:start_x+new_w] = image_scaled
        pad_mask[start_y:start_y+new_h, start_x:start_x+new_w] = mask_scaled
        image = pad_image
        mask = pad_mask
    
    # Random brightness/contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-25, 25)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image, mask


def get_points_from_mask(mask: np.ndarray, num_points: int = 3) -> Optional[np.ndarray]:
    """
    Sample random points from inside the mask region.
    Points are used as prompts for SAM2.
    
    Returns array of shape (num_points, 1, 2) in xy format.
    """
    # Erode slightly to avoid boundary points
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    coords = np.argwhere(eroded > 0)
    
    if len(coords) == 0:
        coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        return None
    
    points = []
    for _ in range(num_points):
        idx = np.random.randint(len(coords))
        yx = coords[idx]
        points.append([[yx[1], yx[0]]])  # xy format
    
    return np.array(points)


def get_box_from_mask(mask: np.ndarray, padding: int = 5, pad: Optional[int] = None) -> Optional[np.ndarray]:
    """Get bounding box from a binary mask.

    Args:
        mask: HxW binary (0/1) mask (numpy).
        padding: pixels of padding to expand the box.
        pad: backwards-compatible alias for `padding` (some callers use pad=).

    Returns:
        np.ndarray of shape (1,4) in XYXY order: [[x_min, y_min, x_max, y_max]],
        dtype float32, or None if mask empty.
    """
    if pad is not None:
        padding = int(pad)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    h, w = mask.shape[:2]
    x_min = max(0, int(xs.min()) - int(padding))
    y_min = max(0, int(ys.min()) - int(padding))
    x_max = min(w - 1, int(xs.max()) + int(padding))
    y_max = min(h - 1, int(ys.max()) + int(padding))

    return np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)



def clamp_mask_to_box(mask, box_xyxy):
    """Zero out mask pixels outside box (XYXY).
    Works for numpy uint8 masks (H,W) and torch tensors.
    """
    import numpy as _np
    try:
        import torch as _torch
    except Exception:
        _torch = None

    if box_xyxy is None:
        return mask

    # Accept [[x1,y1,x2,y2]] or [x1,y1,x2,y2]
    if isinstance(box_xyxy, (list, tuple)):
        if len(box_xyxy) == 1 and isinstance(box_xyxy[0], (list, tuple, _np.ndarray)):
            b = box_xyxy[0]
        else:
            b = box_xyxy
        x1, y1, x2, y2 = map(int, b)
    else:
        b = _np.array(box_xyxy).reshape(-1)
        if b.size == 4:
            x1, y1, x2, y2 = map(int, b.tolist())
        else:
            b = b.reshape(-1, 4)[0]
            x1, y1, x2, y2 = map(int, b.tolist())

    if _torch is not None and isinstance(mask, _torch.Tensor):
        # mask expected HxW
        H, W = mask.shape[-2], mask.shape[-1]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        out = mask.clone()
        out[..., :y1, :] = 0
        out[..., y2+1:, :] = 0
        out[..., :, :x1] = 0
        out[..., :, x2+1:] = 0
        return out
    else:
        mask_np = mask
        H, W = mask_np.shape[:2]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        out = mask_np.copy()
        out[:y1, :] = 0
        out[y2+1:, :] = 0
        out[:, :x1] = 0
        out[:, x2+1:] = 0
        return out


def get_points_from_box(box_xyxy: np.ndarray, num_points: int = 3) -> Optional[np.ndarray]:
    """Sample random points uniformly from inside an XYXY box."""
    if box_xyxy is None:
        return None
    b = np.array(box_xyxy).reshape(-1)
    if b.size == 4:
        x1, y1, x2, y2 = b
    else:
        x1, y1, x2, y2 = b.reshape(-1,4)[0]
    x1, y1, x2, y2 = map(int, [x1,y1,x2,y2])
    if x2 <= x1 or y2 <= y1:
        return None
    pts = []
    for _ in range(int(num_points)):
        x = np.random.randint(x1, x2+1)
        y = np.random.randint(y1, y2+1)
        pts.append([[x, y]])
    return np.array(pts, dtype=np.float32)


def postprocess_mask(mask: np.ndarray, min_area: int = 30) -> np.ndarray:
    """Light cleanup: keep largest connected component + close small holes."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if m.sum() < min_area:
        return m
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:
        # label 0 is background
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = (0, 255, 0),
                 alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask on an image with the given color and transparency."""
    vis = image_bgr.copy()
    m = (mask > 0).astype(bool)
    if m.ndim == 3:
        m = m.squeeze()
    overlay = vis.copy()
    overlay[m] = color
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def _coco_segmentation_to_mask(segmentation, h: int, w: int) -> np.ndarray:
    """Rasterise a COCO 'segmentation' field to a binary mask.

    v19: adds RLE support. Descriptron writes RLE whenever the annotator erased
    a region inside a mask (a paper point mount between an ant's legs, say),
    because COCO polygons cannot express holes -- pycocotools unions multi-ring
    polygons and the per-ring fillPoly below does the same. Previously an RLE
    segmentation fell through the `isinstance(seg, list)` test (iterating a dict
    yields its keys) and produced a silently blank mask.

    Handles:
      * uncompressed RLE  {'size': [h, w], 'counts': [int, ...]}  (Descriptron)
      * compressed RLE    {'size': [h, w], 'counts': str | bytes} (pycocotools)
      * polygons          [[x1, y1, x2, y2, ...], ...]            (unchanged)

    Polygon handling is deliberately left as a union of independently filled
    rings: separate blobs (legs, antennae) are the common case, and switching to
    an even-odd single call would punch spurious holes wherever two blobs
    overlap.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if isinstance(segmentation, dict) and 'counts' in segmentation:
        counts = segmentation['counts']
        size = segmentation.get('size', [h, w])
        rh, rw = int(size[0]), int(size[1])

        if isinstance(counts, (str, bytes)):
            try:
                from pycocotools import mask as _mask_utils
                rle = dict(segmentation)
                if isinstance(counts, str):
                    rle['counts'] = counts.encode('ascii')
                mask = _mask_utils.decode(rle).astype(np.uint8)
            except Exception as e:
                logger.warning(f"Could not decode compressed RLE ({e}); empty mask")
                return np.zeros((h, w), dtype=np.uint8)
        else:
            # Uncompressed RLE: column-major runs, alternating, starting with 0.
            flat = np.zeros(rh * rw, dtype=np.uint8)
            idx, val = 0, 0
            for run in counts:
                run = int(run)
                flat[idx:idx + run] = val
                idx += run
                val = 1 - val
            mask = flat.reshape((rw, rh)).T.astype(np.uint8)

        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    for seg in segmentation:
        if isinstance(seg, list) and len(seg) >= 6:
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


# ============================================================================ v21: CANONICAL FRAME
# Every image that enters SAM2 - the template AND every target - is put into ONE canonical frame:
# optional foreground extraction (rembg) with the background replaced by a flat fill, then LETTERBOXED
# (aspect preserved, Lanczos) into a side x side square. The template masks are letterboxed with the
# identical geometry, so mask and image never disagree, and every prediction is mapped back to the
# target's own native pixel grid before it is saved. Before v21 the targets were STRETCHED to the
# template's pixel size, so any target whose aspect ratio differed from the template was distorted
# and its masks came back offset/misshapen - a bug this project has hit and re-fixed repeatedly.
def _letterbox_geometry(w: int, h: int, side: int):
    scale = side / float(max(w, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return scale, nw, nh, (side - nw) // 2, (side - nh) // 2


def _letterbox_rgb(img_pil, side: int, fill=(255, 255, 255)):
    w, h = img_pil.size
    scale, nw, nh, ox, oy = _letterbox_geometry(w, h, side)
    canvas = Image.new("RGB", (side, side), tuple(int(v) for v in fill))
    canvas.paste(img_pil.resize((nw, nh), Image.LANCZOS), (ox, oy))
    return canvas, (scale, ox, oy, w, h)


def _letterbox_mask(mask: np.ndarray, side: int) -> np.ndarray:
    h, w = mask.shape[:2]
    _scale, nw, nh, ox, oy = _letterbox_geometry(w, h, side)
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize((mask > 0).astype(np.uint8), (nw, nh),
                                                interpolation=cv2.INTER_NEAREST)
    return canvas


def _unletterbox_mask(mask_canvas: np.ndarray, lb) -> np.ndarray:
    scale, ox, oy, w, h = lb
    _s, nw, nh, _ox, _oy = _letterbox_geometry(w, h, mask_canvas.shape[0])
    crop = mask_canvas[oy:oy + nh, ox:ox + nw]
    return cv2.resize((crop > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def _rembg_session(model_name: str = "isnet-general-use"):
    from rembg import new_session
    return new_session(model_name)


def _foreground_only(img_pil, session, fill=(255, 255, 255)):
    """rembg alpha -> composite the specimen onto a flat background."""
    from rembg import remove
    rgba = remove(img_pil, session=session)
    bg = Image.new("RGB", img_pil.size, tuple(int(v) for v in fill))
    bg.paste(rgba, mask=rgba.split()[-1])
    return bg


def _dedup_categories_first_wins(categories_list: list) -> dict:
    """
    Build {category_id: category_dict} using FIRST-WINS deduplication.
    
    COCO JSONs from the Descriptron GBIF Annotator can accumulate duplicate
    category entries from multiple annotation rounds. For example:
        [0] id=1, name="pronotum"
        [6] id=2, name="pronotum"     ← WRONG: id=2 was "left_elytron"
    
    A dict comprehension {cat['id']: cat for cat in ...} takes the LAST entry
    (Python last-wins semantics), producing id=2 → "pronotum" instead of
    the correct "left_elytron".
    
    First-wins preserves the original annotation intent: the first category
    entry for each ID corresponds to the annotation round that created
    the actual polygon annotations.
    """
    result = {}
    for cat in categories_list:
        cid = cat.get('id')
        if cid is None:
            continue
        if cid not in result:
            result[cid] = cat
        elif result[cid].get('name') != cat.get('name'):
            logger.warning(
                f"Duplicate category id={cid}: keeping \"{result[cid]['name']}\" "
                f"(ignoring later \"{cat.get('name', '?')}\")"
            )
    return result


# v21: one FIXED, distinct colour per category (BGR), so 18 structures never share a colour and the
# same structure has the same colour on every image; plus a legend strip. The old visualisers cycled
# 5-6 colours over all masks, which made an 18-part template look like duplicated predictions.
_V21_PALETTE_RGB = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
                    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
                    (170, 255, 195), (128, 128, 0), (0, 0, 128), (255, 140, 0), (0, 200, 120)]


def _v21_colour_bgr(category_id: int):
    r, g, b = _V21_PALETTE_RGB[(int(category_id) - 1) % len(_V21_PALETTE_RGB)]
    return (b, g, r)


def _v21_legend(width: int, entries, cols: int = 3) -> np.ndarray:
    """entries: [(name, bgr), ...] -> a white legend strip of the given width."""
    rows = (len(entries) + cols - 1) // cols
    row_h = max(22, width // 60)
    strip = np.full((rows * row_h + 8, width, 3), 255, dtype=np.uint8)
    fs = max(0.5, width / 2400.0)
    for k, (name, bgr) in enumerate(entries):
        x = 10 + (k % cols) * (width // cols)
        y = 6 + (k // cols) * row_h
        cv2.rectangle(strip, (x, y), (x + row_h - 6, y + row_h - 6), bgr, -1)
        cv2.putText(strip, str(name), (x + row_h, y + row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return strip


def save_side_by_side(template_bgr: np.ndarray, template_mask: np.ndarray,
                      target_bgr: np.ndarray, target_masks: List[np.ndarray],
                      output_path: str, labels=None) -> None:
    """Save a side-by-side comparison of template and predicted target masks.
    v21: `labels` = [(name, category_id), ...] aligned with target_masks -> fixed colour per
    category and a legend; without labels the old 5-colour cycle is used."""
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # Template side
    t_vis = overlay_mask(template_bgr, template_mask, color=(0, 255, 0), alpha=0.4)
    
    # Target side
    tgt_vis = target_bgr.copy()
    for i, m in enumerate(target_masks):
        c = _v21_colour_bgr(labels[i][1]) if labels else colors[i % len(colors)]
        tgt_vis = overlay_mask(tgt_vis, m, color=c, alpha=0.4)
    
    # Resize to same height
    h = max(t_vis.shape[0], tgt_vis.shape[0])
    if t_vis.shape[0] != h:
        scale = h / t_vis.shape[0]
        t_vis = cv2.resize(t_vis, (int(t_vis.shape[1] * scale), h))
    if tgt_vis.shape[0] != h:
        scale = h / tgt_vis.shape[0]
        tgt_vis = cv2.resize(tgt_vis, (int(tgt_vis.shape[1] * scale), h))
    
    combined = np.hstack([t_vis, tgt_vis])
    if labels:
        combined = np.vstack([combined, _v21_legend(combined.shape[1],
                                                   [(nm, _v21_colour_bgr(cid)) for nm, cid in labels])])
    cv2.imwrite(output_path, combined)



class LoRALinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear that adds low-rank adapters.

    output = base_linear(x) + (x @ A^T @ B^T) * scaling
    Base weights are frozen; only A and B are trainable.
    """
    def __init__(self, base_linear: torch.nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        import math
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base_linear.in_features, base_linear.out_features
        device = base_linear.weight.device
        dtype = base_linear.weight.dtype
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, in_f, device=device, dtype=dtype))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora_to_module(module: torch.nn.Module, rank: int = 16, alpha: float = 32.0) -> int:
    """Replace all nn.Linear layers in module with LoRALinear wrappers.
    Returns the number of layers replaced."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            replaced += 1
        else:
            replaced += apply_lora_to_module(child, rank=rank, alpha=alpha)
    return replaced


def collect_lora_state_dict(module: torch.nn.Module, prefix: str = '') -> dict:
    """Extract only LoRA adapter weights from a module tree."""
    state = {}
    for name, child in module.named_children():
        full = f"{prefix}{name}." if prefix else f"{name}."
        if isinstance(child, LoRALinear):
            state[f"{full}lora_A"] = child.lora_A.data
            state[f"{full}lora_B"] = child.lora_B.data
        else:
            state.update(collect_lora_state_dict(child, full))
    return state


def load_lora_state_dict(module: torch.nn.Module, state: dict, prefix: str = ''):
    """Load LoRA adapter weights into a module tree that already has LoRALinear layers."""
    for name, child in module.named_children():
        full = f"{prefix}{name}." if prefix else f"{name}."
        if isinstance(child, LoRALinear):
            a_key = f"{full}lora_A"
            b_key = f"{full}lora_B"
            if a_key in state and b_key in state:
                child.lora_A.data.copy_(state[a_key])
                child.lora_B.data.copy_(state[b_key])
        else:
            load_lora_state_dict(child, state, full)


class SAM2PAL:
    """
    SAM2-PAL: Palindrome-based Mask Propagation

    Uses SAM2's video tracking to propagate masks through a pseudo-video
    created from static images.
    """
    
    def __init__(self, sam2_checkpoint: str, sam2_config: str, device: str = 'cuda'):
        """Initialize SAM2."""
        import torch
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # Build SAM2 model (shared between image and video predictors)
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        
        # Image predictor for fine-tuning (supports gradients)
        self.image_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Video predictor for inference
        self.video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=self.device)
        
        logger.info("SAM2-PAL initialized successfully")
        
        self.template_mask = None
        self.categories = {}
    
    def load_template_mask(self, mask_path: str, category_name: str = 'object') -> np.ndarray:
        """Load template mask from file."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        self.template_mask = (binary_mask > 0).astype(np.uint8)
        
        self.categories = {1: {'id': 1, 'name': category_name, 'supercategory': 'object'}}
        
        logger.info(f"Loaded template mask: {mask_path}")
        logger.info(f"Mask shape: {self.template_mask.shape}, non-zero pixels: {np.sum(self.template_mask > 0)}")
        
        return self.template_mask
    
    def _canon_image(self, path: str):
        """v21: load `path` into the canonical frame and remember its letterbox geometry.

        Returns (PIL RGB canvas, lb) where lb = (scale, ox, oy, orig_w, orig_h). With canvas_side == 0
        and no foreground session this is exactly the pre-v21 behaviour (plain RGB load)."""
        Image.MAX_IMAGE_PIXELS = None
        if not hasattr(self, "_canon"):
            self._canon = {}
        side = int(getattr(self, "canvas_side", 0) or 0)
        fill = getattr(self, "fg_fill", (255, 255, 255))
        img = Image.open(path).convert("RGB")
        session = getattr(self, "fg_session", None)
        if session is not None:
            cache = getattr(self, "canon_cache_dir", None)
            cp = os.path.join(cache, os.path.splitext(os.path.basename(path))[0] + "_fg.png") if cache else None
            if cp and os.path.exists(cp):
                img = Image.open(cp).convert("RGB")
            else:
                img = _foreground_only(img, session, fill)
                if cp:
                    os.makedirs(cache, exist_ok=True)
                    img.save(cp)
        if not side:
            lb = (1.0, 0, 0, img.size[0], img.size[1])
            self._canon[path] = lb
            return img, lb
        canvas, lb = _letterbox_rgb(img, side, fill)
        self._canon[path] = lb
        return canvas, lb

    def load_template_from_coco(self, coco_json_path: str, template_image_path: str) -> List[Dict]:
        """Load template masks from COCO JSON."""
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        template_filename = os.path.basename(template_image_path)
        template_stem = os.path.splitext(template_filename)[0]
        template_img = cv2.imread(template_image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #template_img = cv2.imread(template_image_path)
        h, w = template_img.shape[:2]

        template_image_id = None
        for img in coco_data.get('images', []):
            fn = img.get('file_name', '')
            fn_base = os.path.basename(fn)
            fn_stem = os.path.splitext(fn_base)[0]
            if fn == template_filename or fn_base == template_filename or fn_stem == template_stem:
                template_image_id = img['id']
                break

        if template_image_id is None and coco_data.get('images'):
            # Fall back to whichever image ID has the most annotations
            ann_counts = {}
            for ann in coco_data.get('annotations', []):
                iid = ann.get('image_id')
                ann_counts[iid] = ann_counts.get(iid, 0) + 1
            if ann_counts:
                template_image_id = max(ann_counts, key=ann_counts.get)
                logger.warning(f"Template not found by name, using image_id={template_image_id} (most annotations: {ann_counts[template_image_id]})")
            else:
                template_image_id = coco_data['images'][0]['id']
                logger.warning(f"Template not found by name and no annotations found, using first image")
        
        annotations = [
            ann for ann in coco_data.get('annotations', [])
            if ann.get('image_id') == template_image_id
        ]
        
        self.categories = _dedup_categories_first_wins(coco_data.get('categories', []))
        
        masks = []
        for ann in annotations:
            if 'segmentation' in ann:
                mask = _coco_segmentation_to_mask(ann['segmentation'], h, w)
                masks.append({
                    'mask': mask,
                    'category_id': ann.get('category_id', 1)
                })
        
        logger.info(f"Loaded {len(masks)} masks from COCO JSON")
        return masks
    
    def load_training_templates(self, training_json: Optional[str] = None,
                                 training_images_dir: Optional[str] = None,
                                 training_masks_dir: Optional[str] = None) -> List[Dict]:
        """
        Load multiple training templates for PAL fine-tuning.
        
        Can load from:
        1. COCO JSON + images directory
        2. Mask PNGs directory + images directory (masks named: imagename_*_mask.png)
        
        Returns list of dicts: [{'image_path': str, 'mask': np.ndarray, 'category_id': int}, ...]
        """
        templates = []
        
        # Option 1: Load from COCO JSON
        if training_json and os.path.exists(training_json):
            logger.info(f"Loading training templates from COCO JSON: {training_json}")
            
            with open(training_json, 'r') as f:
                coco_data = json.load(f)
            
            # Build image_id -> filename mapping
            images_by_id = {img['id']: img for img in coco_data.get('images', [])}
            
            # Group annotations by image
            anns_by_image = {}
            for ann in coco_data.get('annotations', []):
                img_id = ann.get('image_id')
                if img_id not in anns_by_image:
                    anns_by_image[img_id] = []
                anns_by_image[img_id].append(ann)
            
            self.categories = _dedup_categories_first_wins(coco_data.get('categories', []))
            
            for img_id, anns in anns_by_image.items():
                if img_id not in images_by_id:
                    continue
                    
                img_info = images_by_id[img_id]
                filename = img_info['file_name']
                h, w = img_info.get('height', 0), img_info.get('width', 0)
                
                # Find image path
                if training_images_dir:
                    img_path = os.path.join(training_images_dir, filename)
                else:
                    img_path = filename
                
                if not os.path.exists(img_path):
                    logger.warning(f"Training image not found: {img_path}")
                    continue
                
                # Load image to get dimensions if not in JSON
                if h == 0 or w == 0:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                
                # Create mask from annotations
                for ann in anns:
                    if 'segmentation' not in ann:
                        continue
                    
                    mask = _coco_segmentation_to_mask(ann['segmentation'], h, w)

                    if mask.sum() > 0:
                        templates.append({
                            'image_path': img_path,
                            'mask': mask,
                            'category_id': ann.get('category_id', 1)
                        })
            
            logger.info(f"Loaded {len(templates)} templates from COCO JSON")
        
        # Option 2: Load from mask PNG directory
        elif training_masks_dir and os.path.exists(training_masks_dir):
            logger.info(f"Loading training templates from masks directory: {training_masks_dir}")
            
            mask_files = sorted([
                f for f in os.listdir(training_masks_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'mask' in f.lower()
            ])
            
            for mask_file in mask_files:
                mask_path = os.path.join(training_masks_dir, mask_file)
                
                # Try to find corresponding image
                # Expected naming: imagename_category_mask.png -> imagename.jpg
                base_name = mask_file
                for suffix in ['_mask.png', '_mask.jpg', '.mask.png', '.mask.jpg']:
                    if base_name.lower().endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                # Remove category suffix if present (e.g., imagename_scrobe -> imagename)
                parts = base_name.rsplit('_', 1)
                if len(parts) > 1:
                    potential_base = parts[0]
                else:
                    potential_base = base_name
                
                # Search for image
                img_path = None
                if training_images_dir:
                    for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                        candidate = os.path.join(training_images_dir, potential_base + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                        # Also try the full base_name
                        candidate = os.path.join(training_images_dir, base_name + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                
                if img_path is None:
                    logger.warning(f"Could not find image for mask: {mask_file}")
                    continue
                
                # Load mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask = (mask > 0).astype(np.uint8)
                
                if mask.sum() > 0:
                    templates.append({
                        'image_path': img_path,
                        'mask': mask,
                        'category_id': 1  # Default category
                    })
            
            logger.info(f"Loaded {len(templates)} templates from mask directory")
        
        return templates
    
    def finetune(self, template_image_path: str, template_mask: np.ndarray,
                 output_checkpoint: str, num_epochs: int = 200,
                 learning_rate: float = 5e-5, num_points: int = 3,
                 use_box_prompt: bool = True, accumulation_steps: int = 4):
        """
        Fine-tune SAM2 using template image with augmentations.
        
        This uses SAM2ImagePredictor's internal components which support gradients:
        - sam_prompt_encoder: encodes point/box prompts
        - sam_mask_decoder: predicts masks
        
        The approach simulates OC-CCL by training the model to correctly segment
        the template under various augmentations (simulating the variations it
        will see in target images).
        
        Args:
            template_image_path: Path to template image
            template_mask: Ground truth binary mask
            output_checkpoint: Where to save fine-tuned weights
            num_epochs: Training iterations
            learning_rate: Learning rate (5e-5 recommended)
            num_points: Number of point prompts per sample
            use_box_prompt: Also use bounding box as prompt
            accumulation_steps: Gradient accumulation steps
        """
        import torch
        import torch.optim as optim
        
        logger.info("="*60)
        logger.info("Starting Fine-tuning")
        logger.info(f"Epochs: {num_epochs}, LR: {learning_rate}")
        if use_box_prompt:
            logger.info(f"Prompt strategy: {num_points} point(s) + bounding box")
        else:
            logger.info(f"Prompt strategy: {num_points} point(s) (no box)")
        logger.info("="*60)
        
        # Load template image
        #template_img = cv2.imread(template_image_path)
        #timg = cv2.imread(template_image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        template_img = cv2.imread(template_image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if template_img is None:
            raise ValueError(f"Could not load: {template_image_path}")
        template_img_rgb = cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB)
        
        # Resize to SAM2's size if needed (max 1024)
        h, w = template_img_rgb.shape[:2]
        r = min(1024 / w, 1024 / h, 1.0)
        if r < 1:
            new_w, new_h = int(w * r), int(h * r)
            template_img_rgb = cv2.resize(template_img_rgb, (new_w, new_h))
            template_mask = cv2.resize(template_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = new_h, new_w
        
        # Set model to training mode for decoder and prompt encoder
        self.sam2_model.sam_mask_decoder.train(True)
        self.sam2_model.sam_prompt_encoder.train(True)
        
        # Freeze image encoder (saves memory, preserves general features)
        for param in self.sam2_model.image_encoder.parameters():
            param.requires_grad = False
        
        # Collect trainable parameters
        trainable_params = list(self.sam2_model.sam_mask_decoder.parameters()) + \
                          list(self.sam2_model.sam_prompt_encoder.parameters())
        
        num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        logger.info(f"Trainable parameters: {num_trainable:,}")
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.6)
        scaler = torch.amp.GradScaler('cuda')
        
        best_iou = 0
        mean_iou = 0
        
        for epoch in range(num_epochs):
            # Apply augmentation to simulate variation
            aug_img, aug_mask = augment_image_and_mask(
                template_img_rgb.copy(), 
                template_mask.copy()
            )
            
            # Skip if mask is too small after augmentation
            if np.sum(aug_mask > 0) < 50:
                continue
            
            # Get prompts from mask
            # Use EITHER multiple points OR single point + box (not both with multiple)
            if use_box_prompt:
                # Points + box (stable and matches many SAM/SAM2 finetune recipes)
                input_points = get_points_from_mask(aug_mask, num_points)
                if input_points is None:
                    continue
                input_labels = np.ones((num_points,), dtype=np.int32)
                input_box = get_box_from_mask(aug_mask)
            else:
                # Multiple points, no box
                input_points = get_points_from_mask(aug_mask, num_points)
                if input_points is None:
                    continue
                input_labels = np.ones((num_points,), dtype=np.int32)
                input_box = None
            
            with torch.amp.autocast('cuda'):
                # Encode image
                self.image_predictor.set_image(aug_img)
                
                # Prepare prompts using internal method
                mask_input, unnorm_coords, labels, unnorm_box = self.image_predictor._prep_prompts(
                    input_points, input_labels, 
                    box=input_box, 
                    mask_logits=None, 
                    normalize_coords=True
                )
                
                if unnorm_coords is None or labels is None:
                    continue


                # --- NEW: robust prompt tensor shaping (fix batch mismatch) ---
                # SAM2 prompt encoder expects:
                #   coords: (B, N, 2) float
                #   labels: (B, N)   long/int
                #   boxes:  (B, 2, 2) float (preferred)
                import torch

                # Convert to torch on the correct device first (even if _prep_prompts returned numpy)
                dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
                if unnorm_coords is not None and not torch.is_tensor(unnorm_coords):
                    unnorm_coords = torch.as_tensor(unnorm_coords, device=dev)
                if labels is not None and not torch.is_tensor(labels):
                    labels = torch.as_tensor(labels, device=dev)
                if unnorm_box is not None and not torch.is_tensor(unnorm_box):
                    unnorm_box = torch.as_tensor(unnorm_box, device=dev)

                # Coerce labels shape: (N,1) -> (N,), then add batch -> (1,N)
                if labels is not None:
                    if labels.dim() == 2 and labels.shape[1] == 1:
                        labels = labels.squeeze(1)
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(0)
                    # if labels somehow becomes (N,) after squeeze, ensure batched
                    if labels.dim() == 0:
                        labels = labels.view(1, 1)
                    labels = labels.long()

                # Coerce coords shape:
                #  (N,2) -> (1,N,2)
                #  (N,1,2) -> squeeze -> (N,2) -> (1,N,2)
                #  (1,N,2) is already good
                if unnorm_coords is not None:
                    if unnorm_coords.dim() == 3 and unnorm_coords.shape[1] == 1 and unnorm_coords.shape[2] == 2:
                        unnorm_coords = unnorm_coords.squeeze(1)
                    if unnorm_coords.dim() == 2 and unnorm_coords.shape[-1] == 2:
                        unnorm_coords = unnorm_coords.unsqueeze(0)
                    # final sanity: ensure (B,N,2)
                    if unnorm_coords.dim() != 3 or unnorm_coords.shape[-1] != 2:
                        raise RuntimeError(f"Unexpected unnorm_coords shape: {tuple(unnorm_coords.shape)}")
                    unnorm_coords = unnorm_coords.float()

                # Coerce box shape to (1,2,2) when present
                if unnorm_box is not None:
                    # allow (4,) or (1,4)
                    if unnorm_box.dim() == 1 and unnorm_box.numel() == 4:
                        x1, y1, x2, y2 = unnorm_box.tolist()
                        unnorm_box = torch.tensor([[[x1, y1], [x2, y2]]], device=dev)
                    elif unnorm_box.dim() == 2 and unnorm_box.shape == (2, 2):
                        unnorm_box = unnorm_box.unsqueeze(0)
                    elif unnorm_box.dim() == 3 and unnorm_box.shape[0] != 1 and unnorm_box.shape[1:] == (2, 2):
                        # sometimes comes as (N,2,2) by accident; take first box
                        unnorm_box = unnorm_box[:1]
                    # final sanity
                    if unnorm_box.dim() != 3 or unnorm_box.shape[1:] != (2, 2):
                        raise RuntimeError(f"Unexpected unnorm_box shape: {tuple(unnorm_box.shape)}")
                    unnorm_box = unnorm_box.float()

                # Ensure batch dimensions are aligned (B must match)
                Bc = unnorm_coords.shape[0] if unnorm_coords is not None else 1
                Bl = labels.shape[0] if labels is not None else 1
                Bb = unnorm_box.shape[0] if unnorm_box is not None else 1
                if not (Bc == Bl == Bb):
                    raise RuntimeError(f"Prompt batch mismatch: coords B={Bc}, labels B={Bl}, box B={Bb}")


                logger.debug(f"Prompt shapes: coords={tuple(unnorm_coords.shape)}, labels={tuple(labels.shape) if labels is not None else None}, box={tuple(unnorm_box.shape) if unnorm_box is not None else None}")
                # Encode prompts

                sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), 
                    boxes=unnorm_box,
                    masks=None
                )
                
                # Get high-res features
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [
                    feat_level[-1].unsqueeze(0) 
                    for feat_level in self.image_predictor._features["high_res_feats"]
                ]
                
                # Decode mask
                low_res_masks, prd_scores, _, _ = self.sam2_model.sam_mask_decoder(
                    image_embeddings=self.image_predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                
                # Upsample masks to original resolution
                prd_masks = self.image_predictor._transforms.postprocess_masks(
                    low_res_masks, self.image_predictor._orig_hw[-1]
                )
                
                # Prepare ground truth (ensure batched tensors: [B, H, W])
                gt_mask = torch.from_numpy(aug_mask.astype(np.float32)).to(prd_masks.device)
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0)  # [1, H, W]

                # Predicted mask probabilities: [B, H, W]
                prd_mask = torch.sigmoid(prd_masks[:, 0])  # first mask output
                if prd_mask.ndim == 2:
                    prd_mask = prd_mask.unsqueeze(0)

                # If SAM is in "repeat_image" batched mode, expand GT to match batch
                if gt_mask.shape[0] == 1 and prd_mask.shape[0] > 1:
                    gt_mask = gt_mask.expand(prd_mask.shape[0], -1, -1)

                # BCE segmentation loss
                seg_loss = (
                    -gt_mask * torch.log(prd_mask + 1e-6)
                    - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)
                ).mean()

                # IoU calculation for monitoring (per-batch)
                prd_bin = (prd_mask > 0.5).float()
                inter = (gt_mask * prd_bin).sum(dim=(-1, -2))
                union = gt_mask.sum(dim=(-1, -2)) + prd_bin.sum(dim=(-1, -2)) - inter
                iou = inter / (union + 1e-6)

                # Score loss (match confidence to IoU)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                # Combined loss
                loss = seg_loss + score_loss * 0.05
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            if (epoch + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # scheduler.step() is called after optimizer.step() to avoid skipping first LR
            
            # Track IoU
            current_iou = iou.mean().item()
            mean_iou = mean_iou * 0.99 + 0.01 * current_iou
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - IoU: {mean_iou:.4f}, Loss: {loss.item()*accumulation_steps:.4f}")
            
            # Save best model
            if mean_iou > best_iou:
                best_iou = mean_iou
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.sam2_model.state_dict(),
                    'iou': best_iou,
                }, output_checkpoint)
        
        logger.info("="*60)
        logger.info("Fine-tuning Complete!")
        logger.info(f"Best IoU: {best_iou:.4f}")
        logger.info(f"Checkpoint: {output_checkpoint}")
        logger.info("="*60)
        
        # Rebuild video predictor with fine-tuned weights
        self._load_finetuned_weights(output_checkpoint)
        
        return {'best_iou': best_iou}
    
    def _load_finetuned_weights(self, checkpoint_path: str):
        """Load fine-tuned weights into BOTH image and video predictor components.

        Handles two checkpoint formats:
        - LoRA checkpoint: contains 'lora_state' → applies LoRA wrappers then loads adapters
        - Full checkpoint: contains 'model_state_dict' → loads into sam2_model and copies to video predictor
        """
        import torch

        logger.info("Loading fine-tuned weights from: %s", checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # --- LoRA checkpoint ---
        if ckpt.get('use_lora') and 'lora_state' in ckpt:
            lora_rank = ckpt.get('lora_rank', 16)
            lora_state = ckpt['lora_state']
            logger.info(f"Detected LoRA checkpoint (rank={lora_rank})")

            vp = getattr(self, 'video_predictor', None)
            if vp is None:
                logger.warning("No video_predictor; cannot load LoRA weights.")
                return

            for attr, adapter_weights in lora_state.items():
                mod = getattr(vp, attr, None)
                if mod is None:
                    logger.warning(f"video_predictor has no attribute '{attr}', skipping")
                    continue
                n = apply_lora_to_module(mod, rank=lora_rank, alpha=lora_rank * 2.0)
                load_lora_state_dict(mod, adapter_weights)
                logger.info(f"Loaded LoRA adapters into video_predictor.{attr} ({n} layers)")

            self._lora_modules = {
                attr: getattr(vp, attr) for attr in lora_state if hasattr(vp, attr)
            }
            logger.info("LoRA weights loaded successfully")
            return

        # --- Full checkpoint (non-LoRA) ---
        state = ckpt.get('model_state_dict', ckpt)

        missing, unexpected = self.sam2_model.load_state_dict(state, strict=False)
        if missing:
            logger.debug(f"Fine-tune load (image model) missing keys: {len(missing)}")
        if unexpected:
            logger.debug(f"Fine-tune load (image model) unexpected keys: {len(unexpected)}")

        vp = getattr(self, 'video_predictor', None)
        if vp is None:
            logger.warning("No video_predictor present; cannot transfer fine-tuned weights.")
            return

        copied_any = False

        for attr in ['sam_prompt_encoder', 'sam_mask_decoder', 'memory_encoder', 'memory_attention',
                     'obj_ptr_proj', 'obj_ptr_tpos_proj', 'mask_downsample']:
            if hasattr(vp, attr) and hasattr(self.sam2_model, attr):
                try:
                    getattr(vp, attr).load_state_dict(getattr(self.sam2_model, attr).state_dict(), strict=False)
                    copied_any = True
                    logger.info(f"Transferred fine-tuned weights: video_predictor.{attr}")
                except Exception as e:
                    logger.warning(f"Could not transfer {attr} into video predictor: {e}")

        for flag in ['multimask_output_in_sam', 'use_high_res_features_in_sam', 'use_mask_input_as_output_without_sam']:
            if hasattr(vp, flag) and hasattr(self.sam2_model, flag):
                try:
                    setattr(vp, flag, getattr(self.sam2_model, flag))
                except Exception:
                    pass

        if not copied_any:
            try:
                missing2, unexpected2 = vp.load_state_dict(state, strict=False)
                logger.info(
                    "Attempted strict=False load into video_predictor; "
                    f"missing={len(missing2)}, unexpected={len(unexpected2)}"
                )
            except Exception as e:
                logger.warning(
                    "Could not load fine-tuned weights into video predictor. "
                    "Propagation may use original weights. Error: %s" % e
                )
    
    # ========================================================================
    # OC-CCL Fine-tuning (Video Tracker Backpropagation)
    # ========================================================================
    
    def _preprocess_image_for_video(self, image_rgb: np.ndarray) -> 'torch.Tensor':
        """Preprocess image for video predictor (same as SAM2 internal)."""
        import torch
        
        # Resize to model's expected size
        img_size = self.video_predictor.image_size
        h, w = image_rgb.shape[:2]
        
        # Resize maintaining aspect ratio
        scale = img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Pad to square
        padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # To tensor and normalize (SAM2 normalization)
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        
        # SAM2 normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    def _get_vision_features(self, image_tensor: 'torch.Tensor'):
        """
        Extract vision features from an image tensor.
        This bypasses inference_mode by calling forward_image directly.
        
        Args:
            image_tensor: [1, C, H, W] preprocessed image
            
        Returns:
            (vision_feats, vision_pos_embeds, feat_sizes) matching what track_step expects
        """
        import torch
        
        # forward_image is in SAM2Base and supports gradients
        backbone_out = self.video_predictor.forward_image(image_tensor)
        
        # _prepare_backbone_features processes the multi-scale features
        features = self.video_predictor._prepare_backbone_features(backbone_out)
        
        # Handle variable return format
        if len(features) == 4:
            _, vision_feats, vision_pos_embeds, feat_sizes = features
        elif len(features) == 3:
            vision_feats, vision_pos_embeds, feat_sizes = features
        else:
            raise ValueError(f"Unexpected features tuple length: {len(features)}")
        
        return vision_feats, vision_pos_embeds, feat_sizes
    
    def _occcl_forward(self, images_tensor: 'torch.Tensor', gt_mask: 'torch.Tensor',
                       compute_loss_fn) -> 'torch.Tensor':
        """
        Run OC-CCL forward pass per the paper (arxiv.org/abs/2501.06749).
        
        CORRECT 4-FRAME PALINDROME:
            Frame 0: x0 with mask prompt y0 (opening)
            Frame 1: x1 (unlabeled) â†' predict Å·1
            *** RESET MEMORY BANK ***
            Frame 2: x1â€  (duplicate of x1) â†' use Å·1 as DIFFERENTIABLE prompt
            Frame 3: x0â€  (closing) â†' predict, compute loss vs y0
        
        The key insight from the paper:
        - Resetting memory after frame 1 prevents the model from "cheating" by
          just remembering y0 through to frame 3
        - Using Å·1 as a differentiable prompt for frame 2 forces the model to
          learn good intermediate predictions
        - Gradient flows through Å·1, teaching accurate tracking
        
        images_tensor: [x0, x1, x1â€ , x0â€ ] - 4 frames
        """
        import torch
        
        img_size = self.video_predictor.image_size
        num_frames = 4  # Always 4 for OC-CCL
        
        # Prepare mask input for frame 0 (opening)
        mask_input = gt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        if mask_input.shape[-2:] != (img_size, img_size):
            mask_input = torch.nn.functional.interpolate(
                mask_input, size=(img_size, img_size),
                mode='bilinear', align_corners=False
            )
        mask_input = (mask_input >= 0.5).float()
        
        # Initialize output dict for tracking memory (Phase 1: frames 0-1)
        output_dict_phase1 = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # ==========================================
        # PHASE 1: Opening sequence (frames 0-1)
        # ==========================================
        
        # Frame 0: x0 with mask prompt y0
        image_0 = images_tensor[0].unsqueeze(0)
        vision_feats_0, vision_pos_embeds_0, feat_sizes_0 = self._get_vision_features(image_0)
        
        current_out_0 = self.video_predictor.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats_0,
            current_vision_pos_embeds=vision_pos_embeds_0,
            feat_sizes=feat_sizes_0,
            point_inputs=None,
            mask_inputs=mask_input,
            output_dict=output_dict_phase1,
            num_frames=2,  # Phase 1 has 2 frames
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        
        output_dict_phase1["cond_frame_outputs"][0] = {
            "maskmem_features": current_out_0["maskmem_features"],
            "maskmem_pos_enc": current_out_0["maskmem_pos_enc"],
            "pred_masks": current_out_0["pred_masks"],
            "obj_ptr": current_out_0["obj_ptr"],
        }
        
        # Frame 1: x1 (unlabeled) - predict Å·1
        image_1 = images_tensor[1].unsqueeze(0)
        vision_feats_1, vision_pos_embeds_1, feat_sizes_1 = self._get_vision_features(image_1)
        
        current_out_1 = self.video_predictor.track_step(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats_1,
            current_vision_pos_embeds=vision_pos_embeds_1,
            feat_sizes=feat_sizes_1,
            point_inputs=None,
            mask_inputs=None,  # No prompt - uses memory from frame 0
            output_dict=output_dict_phase1,
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        
        # Get predicted mask Å·1 (KEEP DIFFERENTIABLE!)
        pred_mask_1 = current_out_1["pred_masks"]  # This is the key differentiable tensor
        
        # ==========================================
        # MEMORY RESET (per paper Section 3.2)
        # ==========================================
        # Create fresh output_dict for phase 2
        # This prevents the model from carrying y0 memory to frame 3
        
        output_dict_phase2 = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # ==========================================
        # PHASE 2: Closing sequence (frames 2-3)
        # ==========================================
        
        # Frame 2: x1â€  (duplicate of x1) with Å·1 as DIFFERENTIABLE prompt
        # Re-use the same image features from frame 1 (it's a duplicate)
        
        # Prepare Å·1 as mask input (differentiable!)
        pred_mask_1_for_prompt = pred_mask_1  # Keep gradients flowing!
        if pred_mask_1_for_prompt.shape[-2:] != (img_size, img_size):
            pred_mask_1_for_prompt = torch.nn.functional.interpolate(
                pred_mask_1_for_prompt, size=(img_size, img_size),
                mode='bilinear', align_corners=False
            )
        # Apply sigmoid to get probability, then threshold (but keep differentiable)
        pred_mask_1_prob = torch.sigmoid(pred_mask_1_for_prompt)
        
        current_out_2 = self.video_predictor.track_step(
            frame_idx=0,  # Reset frame index for phase 2
            is_init_cond_frame=True,  # This is now the "init" frame for phase 2
            current_vision_feats=vision_feats_1,  # Same features as frame 1 (x1â€ )
            current_vision_pos_embeds=vision_pos_embeds_1,
            feat_sizes=feat_sizes_1,
            point_inputs=None,
            mask_inputs=pred_mask_1_prob,  # Å·1 as differentiable prompt!
            output_dict=output_dict_phase2,
            num_frames=2,  # Phase 2 has 2 frames
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        
        output_dict_phase2["cond_frame_outputs"][0] = {
            "maskmem_features": current_out_2["maskmem_features"],
            "maskmem_pos_enc": current_out_2["maskmem_pos_enc"],
            "pred_masks": current_out_2["pred_masks"],
            "obj_ptr": current_out_2["obj_ptr"],
        }
        
        # Frame 3: x0â€  (closing) - predict from memory of Å·1
        # Re-use features from frame 0 (it's a duplicate)
        
        current_out_3 = self.video_predictor.track_step(
            frame_idx=1,  # Second frame in phase 2
            is_init_cond_frame=False,
            current_vision_feats=vision_feats_0,  # Same features as frame 0 (x0â€ )
            current_vision_pos_embeds=vision_pos_embeds_0,
            feat_sizes=feat_sizes_0,
            point_inputs=None,
            mask_inputs=None,  # No prompt - predict from memory
            output_dict=output_dict_phase2,
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=False,  # Last frame, no need to encode
            prev_sam_mask_logits=None,
        )
        
        # ==========================================
        # Compute OC-CCL loss: Å·0â€  vs y0
        # ==========================================
        
        pred_masks_closing = current_out_3["pred_masks"]
        
        if pred_masks_closing.shape[-2:] != gt_mask.shape[-2:]:
            pred_masks_closing = torch.nn.functional.interpolate(
                pred_masks_closing, size=gt_mask.shape[-2:],
                mode='bilinear', align_corners=False
            )
        
        pred_mask_final = pred_masks_closing.squeeze()
        loss = compute_loss_fn(pred_mask_final, gt_mask)
        
        return loss
    
    def finetune_pal(self, template_image_path: str, template_mask: np.ndarray,
                     target_image_paths: List[str], output_checkpoint: str,
                     num_epochs: int = 50, learning_rate: float = 1e-5,
                     max_images_per_epoch: int = 10,
                     additional_templates: Optional[List[Dict]] = None,
                     use_lora: bool = False, lora_rank: int = 16,
                     flip_augment: str = "none"):
        """
        PAL (Palindrome) fine-tuning with correct OC-CCL per arxiv.org/abs/2501.06749.
        
        Creates 4-frame palindromes: {x0, x1, x1â€ , x0â€ }
        
        Phase 1 (frames 0-1):
            x0: Template with mask prompt y0
            x1: Unlabeled â†' predict Å·1
            
        MEMORY RESET (key insight from paper!)
        
        Phase 2 (frames 2-3):
            x1â€ : Duplicate of x1 with Å·1 as DIFFERENTIABLE prompt
            x0â€ : Closing frame â†' predict, compute loss vs y0
        
        The memory reset prevents "cheating" by carrying y0 memory through.
        Using Å·1 as a differentiable prompt forces accurate intermediate predictions.
        
        Args:
            template_image_path: Path to primary template image
            template_mask: Primary template's ground truth mask (numpy)
            target_image_paths: List of unlabeled image paths
            output_checkpoint: Where to save fine-tuned weights
            num_epochs: Training epochs (paper uses 25 with LoRA)
            learning_rate: Learning rate (paper uses 1e-4 with LoRA)
            max_images_per_epoch: Max unlabeled images per epoch (paper uses 100)
            additional_templates: List of additional templates from load_training_templates()
            use_lora: Whether to use LoRA fine-tuning (per paper)
            lora_rank: LoRA rank (default 16)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Build list of all templates
        all_templates = []
        
        # Add primary template
        all_templates.append({
            'image_path': template_image_path,
            'mask': template_mask,
            'category_id': 1
        })
        
        # Add additional templates if provided
        if additional_templates:
            all_templates.extend(additional_templates)

        # Flip augmentation of the LABELLED templates only (unlabelled targets stay as they are),
        # so the model learns to propagate from a mirrored template to an unmirrored specimen.
        # h = left-right, v = top-bottom, hv = both (= 180 deg); 'all' gives 4x: original, h, v, hv.
        _flip_modes = {"none": [], "h": ["h"], "v": ["v"], "hv": ["h", "v"], "all": ["h", "v", "hv"]}
        if flip_augment not in _flip_modes:
            raise ValueError(f"flip_augment must be one of {list(_flip_modes)}, got {flip_augment!r}")
        if _flip_modes[flip_augment]:
            _base = list(all_templates)
            all_templates.extend(dict(t, flip=f) for t in _base for f in _flip_modes[flip_augment])
            logger.info(f"Flip augmentation '{flip_augment}': {len(_base)} labelled templates -> "
                        f"{len(all_templates)} (added {', '.join(_flip_modes[flip_augment])} copies)")
        
        logger.info("="*60)
        if use_lora:
            logger.info("PAL Fine-tuning with LoRA (per paper)")
        else:
            logger.info("PAL Fine-tuning (full fine-tune mode)")
        logger.info(f"Training templates: {len(all_templates)}")
        for i, t in enumerate(all_templates[:5]):  # Show first 5
            logger.info(f"  Template {i+1}: {os.path.basename(t['image_path'])}")
        if len(all_templates) > 5:
            logger.info(f"  ... and {len(all_templates) - 5} more")
        logger.info(f"4-frame palindrome: {{x0, x1, x1â€ , x0â€ }} with MEMORY RESET")
        logger.info(f"Epochs: {num_epochs}, LR: {learning_rate}")
        if use_lora:
            logger.info(f"LoRA rank: {lora_rank}")
        logger.info(f"Unlabeled images available: {len(target_image_paths)}")
        logger.info(f"Images per epoch: {min(max_images_per_epoch, len(target_image_paths))}")
        logger.info("="*60)
        
        # Verify video predictor has required methods
        required_methods = ['forward_image', '_prepare_backbone_features', 'track_step']
        for method in required_methods:
            if not hasattr(self.video_predictor, method):
                raise RuntimeError(f"VideoPredictor missing required method: {method}")
        logger.info("âœ“ VideoPredictor has all required methods for PAL fine-tuning")
        
        # Preprocess all templates
        img_size = self.video_predictor.image_size
        preprocessed_templates = []
        
        for template in all_templates:
            img = cv2.imread(template['image_path'])
            if img is None:
                logger.warning(f"Could not load: {template['image_path']}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _flip_code = {"h": 1, "v": 0, "hv": -1}.get(template.get('flip'))
            if _flip_code is not None:
                img_rgb = np.ascontiguousarray(cv2.flip(img_rgb, _flip_code))
            
            orig_h, orig_w = img_rgb.shape[:2]
            scale = img_size / max(orig_h, orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            # Resize mask
            mask = template['mask']
            if mask.shape[:2] != (orig_h, orig_w):
                mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if _flip_code is not None:
                mask = np.ascontiguousarray(cv2.flip(mask.astype(np.uint8), _flip_code))
            
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Pad to square
            gt_mask_padded = np.zeros((img_size, img_size), dtype=np.float32)
            gt_mask_padded[:new_h, :new_w] = mask_resized
            gt_mask_tensor = torch.from_numpy(gt_mask_padded)

            # Preprocess image — keep on CPU, move to GPU on-demand during training
            img_tensor = self._preprocess_image_for_video(img_rgb)
            
            preprocessed_templates.append({
                'image_tensor': img_tensor,
                'mask_tensor': gt_mask_tensor,
                'orig_size': (orig_h, orig_w),
                'name': os.path.basename(template['image_path'])
                        + (f" [{template['flip']}-flip]" if template.get('flip') else "")
            })
        
        if not preprocessed_templates:
            raise ValueError("No valid templates after preprocessing!")
        
        logger.info(f"Successfully preprocessed {len(preprocessed_templates)} templates")
        
        # ============================================
        # Set up training mode (with optional LoRA)
        # ============================================

        # Freeze image encoder (saves ~70% memory)
        for param in self.video_predictor.image_encoder.parameters():
            param.requires_grad = False
        self.video_predictor.image_encoder.eval()

        # Modules the PAL paper trains: decoder + memory encoder/attention
        lora_targets = []
        target_names = ['sam_mask_decoder', 'sam_prompt_encoder']
        for attr in ['memory_encoder', 'memory_attention']:
            if hasattr(self.video_predictor, attr):
                target_names.append(attr)

        for attr in target_names:
            mod = getattr(self.video_predictor, attr, None)
            if mod is not None:
                lora_targets.append((attr, mod))

        if use_lora:
            # Freeze ALL base weights first
            for _, mod in lora_targets:
                for p in mod.parameters():
                    p.requires_grad = False

            if hasattr(self.video_predictor, 'obj_ptr_proj'):
                for p in self.video_predictor.obj_ptr_proj.parameters():
                    p.requires_grad = False

            # Apply LoRA adapters (only these will be trainable)
            total_replaced = 0
            for attr, mod in lora_targets:
                n = apply_lora_to_module(mod, rank=lora_rank, alpha=lora_rank * 2.0)
                total_replaced += n
                logger.info(f"  LoRA applied to {attr}: {n} Linear layers wrapped (rank={lora_rank})")

            logger.info(f"Total LoRA adapters: {total_replaced}")
            self._lora_modules = {attr: mod for attr, mod in lora_targets}
        else:
            self._lora_modules = None

        # Set target modules to train mode
        for _, mod in lora_targets:
            mod.train(True)

        # Collect trainable parameters
        trainable_params = []
        for _, mod in lora_targets:
            for p in mod.parameters():
                if p.requires_grad:
                    trainable_params.append(p)

        if not use_lora and hasattr(self.video_predictor, 'obj_ptr_proj'):
            for p in self.video_predictor.obj_ptr_proj.parameters():
                if p.requires_grad:
                    trainable_params.append(p)

        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {num_trainable:,}")
        if use_lora:
            logger.info(f"(LoRA mode — paper recommends LR=1e-4, epochs=25)")
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Loss function
        def compute_loss(pred_mask, gt_mask):
            pred = torch.sigmoid(pred_mask)
            pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
            bce = -(gt_mask * torch.log(pred) + (1 - gt_mask) * torch.log(1 - pred)).mean()
            intersection = (pred * gt_mask).sum()
            dice = 1 - (2 * intersection + 1) / (pred.sum() + gt_mask.sum() + 1)
            return bce + dice
        
        best_loss = float('inf')
        
        # ============================================
        # Training loop
        # ============================================
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Sample unlabeled images for this epoch
            if len(target_image_paths) > max_images_per_epoch:
                epoch_images = random.sample(target_image_paths, max_images_per_epoch)
            else:
                epoch_images = target_image_paths.copy()
                random.shuffle(epoch_images)
            
            for img_path in tqdm(epoch_images, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                try:
                    # Randomly select a template (key for multi-template training!)
                    template = random.choice(preprocessed_templates)
                    template_tensor = template['image_tensor'].to(self.device)
                    gt_mask_tensor = template['mask_tensor'].to(self.device)
                    orig_h, orig_w = template['orig_size']
                    
                    # Load and preprocess unlabeled image
                    unlabeled_img = cv2.imread(img_path)
                    if unlabeled_img is None:
                        continue
                    unlabeled_rgb = cv2.cvtColor(unlabeled_img, cv2.COLOR_BGR2RGB)
                    unlabeled_rgb = cv2.resize(unlabeled_rgb, (orig_w, orig_h))
                    unlabeled_tensor = self._preprocess_image_for_video(unlabeled_rgb).to(self.device)
                    
                    # OC-CCL 4-frame palindrome: {x0, x1, x1â€ , x0â€ }
                    # We pass [template, unlabeled] and _occcl_forward reuses features
                    # for the duplicate frames internally
                    images_tensor = torch.stack([
                        template_tensor,   # x0 (opening)
                        unlabeled_tensor,  # x1 (unlabeled)
                    ])
                    
                    optimizer.zero_grad()
                    
                    loss = self._occcl_forward(
                        images_tensor=images_tensor,
                        gt_mask=gt_mask_tensor,
                        compute_loss_fn=compute_loss
                    )
                    
                    if loss is None:
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM on {os.path.basename(img_path)}, clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        logger.warning(f"Error on {os.path.basename(img_path)}: {e}")
                        continue
                
                except Exception as e:
                    logger.warning(f"Error processing {os.path.basename(img_path)}: {e}")
                    continue
                
                finally:
                    del template_tensor, gt_mask_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            scheduler.step()
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Images: {len(epoch_losses)}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    ckpt_data = {
                        'epoch': epoch + 1,
                        'loss': best_loss,
                        'num_templates': len(preprocessed_templates),
                        'use_lora': use_lora,
                    }
                    if use_lora and self._lora_modules:
                        lora_state = {}
                        for attr, mod in self._lora_modules.items():
                            lora_state[attr] = collect_lora_state_dict(mod)
                        ckpt_data['lora_state'] = lora_state
                        ckpt_data['lora_rank'] = lora_rank
                    else:
                        ckpt_data['model_state_dict'] = self.video_predictor.state_dict()
                    torch.save(ckpt_data, output_checkpoint)
                    logger.info(f"  â†' New best! Saved checkpoint")
            else:
                logger.warning(f"Epoch {epoch+1}: No successful training iterations")
        
        logger.info("="*60)
        logger.info("PAL Fine-tuning Complete!")
        logger.info(f"Best loss: {best_loss:.4f}")
        logger.info(f"Templates used: {len(preprocessed_templates)}")
        logger.info(f"Checkpoint: {output_checkpoint}")
        logger.info("="*60)
        
        # Load the best checkpoint (if one was saved)
        if os.path.exists(output_checkpoint):
            self._load_finetuned_weights(output_checkpoint)
        else:
            logger.warning("No checkpoint was saved (all training iterations may have failed)")

        return {'best_loss': best_loss, 'num_templates': len(preprocessed_templates)}
    
    # Backwards compatibility alias
    def finetune_occcl(self, *args, **kwargs):
        """Alias for finetune_pal (deprecated name)."""
        logger.warning("finetune_occcl is deprecated, use finetune_pal instead")
        return self.finetune_pal(*args, **kwargs)
    

    def save_prompt_visualization(self, image_bgr: np.ndarray, mask: np.ndarray,
                                  points_xy: Optional[np.ndarray],
                                  box_xyxy: Optional[np.ndarray],
                                  out_path: str) -> None:
        """Save a debug visualization of mask + prompts on an image."""
        import cv2
        vis = image_bgr.copy()
        h, w = vis.shape[:2]

        # draw mask overlay (red)
        if mask is not None:
            m = (mask > 0).astype(np.uint8)
            if m.ndim == 3:
                m = m.squeeze()
            overlay = vis.copy()
            overlay[m > 0] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        # draw box (yellow)
        if box_xyxy is not None:
            b = np.array(box_xyxy).reshape(-1)
            if b.size != 4:
                b = b.reshape(-1, 4)[0]
            x1, y1, x2, y2 = map(int, b.tolist())
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw points (green)
        if points_xy is not None:
            pts = np.array(points_xy).reshape(-1, 2)
            for (x, y) in pts:
                x_i, y_i = int(x), int(y)
                cv2.circle(vis, (x_i, y_i), 4, (0, 255, 0), -1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, vis)


    def predict_single_mask(self, image_rgb: np.ndarray,
                            point_coords: Optional[np.ndarray],
                            point_labels: Optional[np.ndarray],
                            box_xyxy: Optional[np.ndarray],
                            multimask: bool = True) -> Optional[np.ndarray]:
        """Run the (fine-tuned) image path to predict a mask on a single image.

        Returns a binary uint8 mask (H,W) in the image's current resolution.
        """
        import torch

        self.sam2_model.sam_mask_decoder.eval()
        self.sam2_model.sam_prompt_encoder.eval()

        with torch.inference_mode(), torch.amp.autocast('cuda'):
            self.image_predictor.set_image(image_rgb)

            mask_input, unnorm_coords, labels, unnorm_box = self.image_predictor._prep_prompts(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_xyxy,
                mask_logits=None,
                normalize_coords=True,
            )

            if unnorm_coords is None or labels is None:
                # If no points were provided, still allow box-only prompts
                pass

            # --- robust prompt tensor shaping ---
            device = self.image_predictor.device
            if unnorm_coords is not None:
                coords_t = torch.as_tensor(unnorm_coords, dtype=torch.float32, device=device)
                if coords_t.ndim == 2:
                    coords_t = coords_t.unsqueeze(0)  # [1, N, 2]
            else:
                coords_t = None

            if labels is not None:
                labels_t = torch.as_tensor(labels, dtype=torch.int64, device=device)
                if labels_t.ndim == 1:
                    labels_t = labels_t.unsqueeze(0)  # [1, N]
            else:
                labels_t = None

            boxes_t = None
            if unnorm_box is not None:
                boxes_t = torch.as_tensor(unnorm_box, dtype=torch.float32, device=device)
                # Accept [4] or [1,4] or [1,2,2]
                if boxes_t.ndim == 1 and boxes_t.numel() == 4:
                    boxes_t = boxes_t.view(1, 2, 2)  # [[x1,y1],[x2,y2]]
                elif boxes_t.ndim == 2 and boxes_t.shape[-1] == 4:
                    boxes_t = boxes_t.view(-1, 2, 2)
                elif boxes_t.ndim == 3 and boxes_t.shape[-2:] == (2, 2):
                    pass
                else:
                    # best effort reshape
                    boxes_t = boxes_t.view(-1, 2, 2)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=(coords_t, labels_t) if (coords_t is not None and labels_t is not None) else None,
                boxes=boxes_t,
                masks=mask_input,
            )

            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in self.image_predictor._features["high_res_feats"]
            ]
            low_res_masks, prd_scores, _, _ = self.sam2_model.sam_mask_decoder(
                image_embeddings=self.image_predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            prd_masks = self.image_predictor._transforms.postprocess_masks(
                low_res_masks, self.image_predictor._orig_hw[-1]
            )

            # pick best mask by score
            scores = prd_scores[0].float()
            best_idx = int(torch.argmax(scores).item()) if scores.numel() > 0 else 0
            mask_prob = torch.sigmoid(prd_masks[0, best_idx])
            mask_bin = (mask_prob > 0.5).to(torch.uint8).cpu().numpy()

        return mask_bin


    def prepare_pseudo_video(self, template_image_path: str, target_image_paths: List[str],
                             temp_dir: str, interleave_template: bool = False,
                             palindrome: bool = False) -> str:
        """Create pseudo-video directory with template as frame 0."""
        frames_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Frame 0: Template  (v21: canonical frame - letterboxed square, optional rembg foreground)
        template_img, _tlb = self._canon_image(template_image_path)
        target_size = template_img.size
        template_img.save(os.path.join(frames_dir, '00000.jpg'), 'JPEG', quality=95)

        # Frames 1-N: Target images (optionally interleave template between each)
        frame_idx = 1
        for img_path in target_image_paths:
            if palindrome:
                # PAL palindrome {x0, x1, x1', x0'} per target
                img, _lb = self._canon_image(img_path)          # v21: same canonical frame as the template
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                # x1
                img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1
                # x1' (duplicate)
                img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1
                # x0' (template)
                template_img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1
            elif interleave_template:
                img, _lb = self._canon_image(img_path)          # v21: same canonical frame as the template
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                img.save(os.path.join(frames_dir, f'{frame_idx:05d}.jpg'), 'JPEG', quality=95)
                frame_idx += 1

        total_frames = frame_idx
        logger.info(f"Created pseudo-video with {total_frames} frames")
        return frames_dir
    
    def propagate_masks(self, frames_dir: str, template_masks: List[Dict],
                        multi_mask: bool = False,
                        interleave_template: bool = False,
                        reanchor_every: int = 0,
                        area_growth_limit: float = 0.0,
                        area_clamp_pad: int = 10) -> Dict[int, List[Dict]]:
        """Propagate masks through pseudo-video using SAM2 tracking.
        
        V18 FIX: Uses CPU offloading to prevent OOM on large pseudo-videos.
        V18 FIX: Removed bulk pre-anchoring that caused OOM in preflight.
        """
        import torch
        import gc
        
        # === V18 FIX: CPU offloading prevents OOM ===
        # Training uses manual track_step on 2 frames at a time (low memory).
        # Inference via init_state loads ALL frames; CPU offloading keeps them
        # in RAM and moves to GPU one at a time during propagation.
        try:
            inference_state = self.video_predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False
            )
            logger.info("Using CPU-offloaded inference (memory-efficient)")
        except TypeError:
            # Older SAM2 versions may not support offloading kwargs
            logger.warning("CPU offloading not available; falling back to standard init_state")
            inference_state = self.video_predictor.init_state(video_path=frames_dir)
        
        # Add template masks on frame 0
        # === V18.2 FIX: Build explicit obj_id → category_id map ===
        # We assign obj_id=1,2,3... and record what category each maps to.
        # Do NOT interpret add_new_mask's return value for remapping — it
        # returns ALL obj_ids on the frame, not just the one we added,
        # which caused V18.1 to overwrite earlier entries.
        obj_id_to_category = {}
        for obj_id, mask_data in enumerate(template_masks, start=1):
            mask = mask_data['mask']
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask
            )
            cat_id = mask_data['category_id']
            obj_id_to_category[obj_id] = cat_id
            cat_name = self.categories.get(cat_id, {}).get('name', f'cat_{cat_id}')
            logger.info(f"Added object {obj_id} mask to frame 0 → category {cat_id} ({cat_name})")
        
        logger.info(f"obj_id → category mapping: {obj_id_to_category}")
        
        # === V18 FIX: Do NOT bulk pre-anchor all template frames ===
        # In v17, when interleave_template=True, we pre-anchored masks at every
        # even frame (0, 2, 4, ...). This caused propagate_in_video_preflight()
        # to call _consolidate_temp_output_across_obj() which interpolates ALL
        # pre-anchored masks to high-res on GPU simultaneously → OOM.
        #
        # Instead, we only anchor frame 0. The interleaved template frames
        # (identical image content) naturally provide tracking context through
        # SAM2's memory mechanism. Re-anchoring is done lazily during
        # propagation via reanchor_every (if enabled).

        # Propagate
        results = {}
        prev_area = {}  # obj_id -> area
        prev_box = {}   # obj_id -> box_xyxy

        with torch.inference_mode():
            for frame_idx, obj_ids, masks in tqdm(
                self.video_predictor.propagate_in_video(inference_state),
                desc="Propagating masks"
            ):
                frame_masks = []
                
                # On first frame: detect obj_id scheme and build definitive mapping
                if not results:
                    raw_ids = [int(x) for x in obj_ids] if hasattr(obj_ids, '__iter__') else [int(obj_ids)]
                    logger.info(f"First propagation frame {frame_idx}: SAM2 obj_ids={raw_ids}")
                    logger.info(f"Our registered mapping: {obj_id_to_category}")
                    
                    # Check if ALL returned ids exist in our map
                    all_match = all(int(rid) in obj_id_to_category for rid in raw_ids)
                    
                    if not all_match:
                        # SAM2 renumbered our obj_ids (e.g. 0-indexed vs 1-indexed).
                        # SAM2 always returns objects in registration order, so use
                        # POSITIONAL mapping: output position i → template_masks[i]
                        logger.warning(
                            f"SAM2 obj_ids {raw_ids} don't match our map keys "
                            f"{list(obj_id_to_category.keys())} — rebuilding positional map"
                        )
                        obj_id_to_category = {}
                        for pos, rid in enumerate(raw_ids):
                            if pos < len(template_masks):
                                obj_id_to_category[int(rid)] = template_masks[pos]['category_id']
                        logger.info(f"Rebuilt obj_id → category mapping: {obj_id_to_category}")
                    else:
                        logger.info(f"obj_ids match — using registered mapping")
                
                for i, obj_id in enumerate(obj_ids):
                    oid = int(obj_id)  # Ensure Python int, not torch.Tensor
                    mask = (masks[i] > 0).cpu().numpy().squeeze().astype(np.uint8)
                    _lg = masks[i].detach().float(); _pos = _lg > 0   # [v20] SAM2's own confidence: mean prob inside the predicted mask
                    logit_conf = float(torch.sigmoid(_lg[_pos]).mean()) if bool(_pos.any()) else 0.0

                    # Drift control: clamp explosive growth to previous box (+pad)
                    area = int(mask.sum())
                    if oid in prev_area and area_growth_limit and prev_area[oid] > 0:
                        if area > area_growth_limit * prev_area[oid]:
                            box = prev_box.get(oid)
                            if box is None:
                                box = get_box_from_mask(mask, pad=area_clamp_pad)
                            if box is not None:
                                mask = clamp_mask_to_box(mask, box)
                                area = int(mask.sum())

                    # Update prev stats (skip template frames if interleaved)
                    is_template_frame = interleave_template and (frame_idx % 2 == 0)
                    if area > 0 and not is_template_frame:
                        prev_area[oid] = area
                        b = get_box_from_mask(mask, pad=area_clamp_pad)
                        if b is not None:
                            prev_box[oid] = b

                    # === V18.1 FIX: Use explicit dict, not template_masks[obj_id-1] ===
                    category_id = obj_id_to_category.get(oid, 1)
                    frame_masks.append({
                        'mask': mask,
                        'category_id': category_id,
                        'obj_id': oid,
                        'logit_conf': logit_conf   # [v20]
                    })

                results[frame_idx] = frame_masks

                # Re-anchor at template frames during propagation (lazy approach)
                # This is much more memory-friendly than bulk pre-anchoring
                should_reanchor = False
                if interleave_template and is_template_frame and frame_idx > 0:
                    should_reanchor = True
                elif reanchor_every and frame_idx > 0 and (frame_idx % reanchor_every == 0):
                    should_reanchor = True
                
                if should_reanchor:
                    for obj_id_r, mask_data in enumerate(template_masks, start=1):
                        try:
                            self.video_predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id_r,
                                mask=mask_data['mask']
                            )
                        except Exception as e:
                            logger.debug(f"Re-anchor failed at frame {frame_idx}: {e}")

        # Reset
        self.video_predictor.reset_state(inference_state)
        
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return results
    
    def propagate_masks_cycle_consistent(self, frames_dir: str,
                                          template_masks: List[Dict],
                                          multi_mask: bool = False,
                                          interleave_template: bool = False,
                                          palindrome: bool = False,
                                          area_growth_limit: float = 0.0,
                                          area_clamp_pad: int = 10,
                                          iou_threshold: float = 0.5) -> Tuple[Dict[int, List[Dict]], Dict[int, float]]:
        """
        Palindrome round-trip cycle consistency.

        Single forward pass through {x0, x1, x1', x0'} palindrome frames.
        The model propagates template masks forward; at the final frame (x0')
        we compare the predicted mask with the known template mask.  High IoU
        means the model successfully round-tripped — the target prediction at
        frame 1 is reliable.

        Per-object IoU at the round-trip frame is averaged to give a single
        confidence score for each target image in the chunk.

        Returns:
            (results_dict, confidence_dict)
            - results_dict: frame_idx -> list of mask dicts
            - confidence_dict: frame_idx -> round-trip IoU score
        """
        import torch
        import gc

        # ====================================================================
        # Single forward pass through palindrome
        # ====================================================================
        logger.info("Palindrome round-trip: forward propagation...")

        try:
            inference_state = self.video_predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False
            )
        except TypeError:
            inference_state = self.video_predictor.init_state(video_path=frames_dir)

        # Add template masks on frame 0
        obj_id_to_category = {}
        for obj_id, mask_data in enumerate(template_masks, start=1):
            self.video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=mask_data['mask']
            )
            obj_id_to_category[obj_id] = mask_data['category_id']

        # Build per-object ground truth for round-trip comparison
        gt_combined = None
        for md in template_masks:
            if gt_combined is None:
                gt_combined = md['mask'].copy()
            else:
                gt_combined = np.maximum(gt_combined, md['mask'])

        forward_results = {}
        roundtrip_iou = {}  # target_frame_idx -> round-trip IoU
        self._objiou = {}   # [v20] target_frame_idx -> {obj_id: per-object round-trip IoU}

        prev_area = {}
        prev_box = {}

        with torch.inference_mode():
            for frame_idx, obj_ids, masks in tqdm(
                self.video_predictor.propagate_in_video(inference_state),
                desc="Forward pass"
            ):
                frame_masks = []

                if not forward_results:
                    raw_ids = [int(x) for x in obj_ids] if hasattr(obj_ids, '__iter__') else [int(obj_ids)]
                    all_match = all(int(rid) in obj_id_to_category for rid in raw_ids)
                    if not all_match:
                        logger.warning(f"Palindrome: SAM2 obj_ids {raw_ids} don't match map — rebuilding positional")
                        obj_id_to_category = {}
                        for pos, rid in enumerate(raw_ids):
                            if pos < len(template_masks):
                                obj_id_to_category[int(rid)] = template_masks[pos]['category_id']

                combined_mask = None
                for i, obj_id in enumerate(obj_ids):
                    oid = int(obj_id)
                    mask = (masks[i] > 0).cpu().numpy().squeeze().astype(np.uint8)
                    _lg = masks[i].detach().float(); _pos = _lg > 0   # [v20] SAM2's own confidence: mean prob inside the predicted mask
                    logit_conf = float(torch.sigmoid(_lg[_pos]).mean()) if bool(_pos.any()) else 0.0

                    area = int(mask.sum())
                    if oid in prev_area and area_growth_limit and prev_area[oid] > 0:
                        if area > area_growth_limit * prev_area[oid]:
                            box = prev_box.get(oid)
                            if box is None:
                                box = get_box_from_mask(mask, pad=area_clamp_pad)
                            if box is not None:
                                mask = clamp_mask_to_box(mask, box)
                                area = int(mask.sum())

                    if palindrome:
                        is_template_frame = (frame_idx % 3 == 0)
                    else:
                        is_template_frame = interleave_template and (frame_idx % 2 == 0)
                    if area > 0 and not is_template_frame:
                        prev_area[oid] = area
                        b = get_box_from_mask(mask, pad=area_clamp_pad)
                        if b is not None:
                            prev_box[oid] = b

                    category_id = obj_id_to_category.get(oid, 1)
                    frame_masks.append({
                        'mask': mask,
                        'category_id': category_id,
                        'obj_id': oid,
                        'logit_conf': logit_conf   # [v20]
                    })

                    if combined_mask is None:
                        combined_mask = mask.copy()
                    else:
                        combined_mask = np.maximum(combined_mask, mask)

                forward_results[frame_idx] = frame_masks

                # Round-trip check: x0' frames are at 3, 6, 9, ...
                if palindrome and frame_idx > 0 and frame_idx % 3 == 0:
                    target_frame = frame_idx - 2  # frame 3 → target at frame 1
                    if combined_mask is not None and gt_combined is not None:
                        gt_mask = gt_combined
                        if combined_mask.shape != gt_mask.shape:
                            gt_mask = cv2.resize(gt_mask, (combined_mask.shape[1], combined_mask.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                        intersection = np.sum((combined_mask > 0) & (gt_mask > 0))
                        union = np.sum((combined_mask > 0) | (gt_mask > 0))
                        iou = float(intersection) / (float(union) + 1e-6)
                        roundtrip_iou[target_frame] = iou
                        # [v20] per-object round-trip IoU: an object that was hallucinated on the target
                        # (e.g. a scrobe on a scrobe-less head) does not return onto its own template mask.
                        per_obj = {}
                        for fm in frame_masks:
                            oid_ = fm.get('obj_id')
                            if oid_ is None or not (0 < oid_ <= len(template_masks)):
                                continue
                            tm = template_masks[oid_ - 1]['mask']; pm = fm['mask']
                            if pm.shape != tm.shape:
                                tm = cv2.resize(tm, (pm.shape[1], pm.shape[0]), interpolation=cv2.INTER_NEAREST)
                            u_ = float(np.sum((pm > 0) | (tm > 0)))
                            per_obj[oid_] = float(np.sum((pm > 0) & (tm > 0))) / (u_ + 1e-6)
                        self._objiou[target_frame] = per_obj

        # Cleanup GPU
        self.video_predictor.reset_state(inference_state)
        del inference_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ====================================================================
        # Build confidence scores from round-trip IoU
        # ====================================================================
        confidence = {}
        low_iou_count = 0

        for target_frame, iou in roundtrip_iou.items():
            confidence[target_frame] = iou
            if iou < iou_threshold:
                low_iou_count += 1

        logger.info(f"Palindrome round-trip complete: {low_iou_count} targets below IoU threshold ({iou_threshold})")
        if low_iou_count > 0:
            worst = sorted(confidence.items(), key=lambda x: x[1])[:5]
            for fi, iou in worst:
                logger.warning(f"  Low confidence target frame {fi}: round-trip IoU={iou:.3f}")

        return forward_results, confidence
    
    def propagate_masks_chunked(self, template_image_path: str,
                                 target_image_paths: List[str],
                                 template_masks: List[Dict],
                                 chunk_size: int = 50,
                                 interleave_template: bool = False,
                                 area_growth_limit: float = 0.0,
                                 area_clamp_pad: int = 10,
                                 cycle_consistency: bool = False,
                                 on_image_result=None) -> Tuple[Dict[int, List[Dict]], Dict]:
        """
        Process images in chunks to bound memory usage for very large datasets.

        Each chunk creates its own pseudo-video, propagates, and resets.
        When cycle_consistency=True, each chunk uses forward+backward propagation
        to filter out inconsistent predictions.

        Returns:
            (combined_results, confidence_scores) tuple
        """
        import torch
        import gc

        all_results = {}
        all_confidence = {}
        total = len(target_image_paths)
        n_chunks = (total + chunk_size - 1) // chunk_size
        use_palindrome = cycle_consistency
        processed_count = 0

        mode_str = "palindrome cycle-consistent" if use_palindrome else ("cycle-consistent chunked" if cycle_consistency else "chunked")
        logger.info(f"{mode_str} inference: {total} images in {n_chunks} chunks of ≤{chunk_size}")

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total)
            chunk_paths = target_image_paths[start:end]

            logger.info(f"Chunk {chunk_idx+1}/{n_chunks}: images {start+1}-{end}")

            chunk_temp = tempfile.mkdtemp(prefix=f'pal_chunk{chunk_idx}_')
            try:
                frames_dir = self.prepare_pseudo_video(
                    template_image_path, chunk_paths, chunk_temp,
                    interleave_template=interleave_template,
                    palindrome=use_palindrome
                )

                if cycle_consistency:
                    chunk_results, chunk_conf = self.propagate_masks_cycle_consistent(
                        frames_dir, template_masks,
                        interleave_template=interleave_template,
                        palindrome=use_palindrome
                    )
                else:
                    chunk_results = self.propagate_masks(
                        frames_dir, template_masks,
                        interleave_template=interleave_template,
                        area_growth_limit=area_growth_limit,
                        area_clamp_pad=area_clamp_pad
                    )
                    chunk_conf = {}

                for img_idx_in_chunk, img_path in enumerate(chunk_paths):
                    global_img_idx = start + img_idx_in_chunk
                    if use_palindrome:
                        video_frame_idx = 3 * img_idx_in_chunk + 1
                    elif interleave_template:
                        video_frame_idx = 2 * img_idx_in_chunk + 1
                    else:
                        video_frame_idx = img_idx_in_chunk + 1

                    frame_masks = chunk_results.get(video_frame_idx, [])
                    conf_val = chunk_conf.get(video_frame_idx)
                    for _md in frame_masks:   # [v20] attach the per-object round-trip IoU when available
                        _oi = getattr(self, '_objiou', {}).get(video_frame_idx, {}).get(_md.get('obj_id'))
                        if _oi is not None:
                            _md['obj_iou'] = _oi

                    if on_image_result is not None:
                        on_image_result(global_img_idx, img_path, frame_masks, conf_val)
                        processed_count += 1
                    else:
                        if frame_masks:
                            all_results[global_img_idx] = frame_masks
                        if conf_val is not None:
                            all_confidence[global_img_idx] = conf_val

                del chunk_results
                del chunk_conf

            finally:
                shutil.rmtree(chunk_temp, ignore_errors=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                gc.collect()

        if on_image_result is not None:
            logger.info(f"{mode_str} inference complete: {processed_count}/{total} images processed (streamed)")
            return {}, {}
        else:
            logger.info(f"{mode_str} inference complete: {len(all_results)}/{total} images processed")
            return all_results, all_confidence
    
    def mask_to_coco_annotation(self, mask: np.ndarray, annotation_id: int,
                                 image_id: int, category_id: int) -> Optional[Dict]:
        """Convert binary mask to COCO annotation."""
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        if area < 10:
            return None
        
        segmentation = contour.flatten().tolist()
        if len(segmentation) < 6:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        
        return {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': [segmentation],
            'area': float(area),
            'bbox': [float(x), float(y), float(w), float(h)],
            'iscrowd': 0,
            'score': 1.0
        }
    
    def process_batch(self, template_image_path: str, template_masks: List[Dict],
                      target_image_paths: List[str], output_dir: str,
                      save_masks: bool = True, save_vis: bool = False,
                      multi_mask: bool = False,
                      interleave_template: bool = False,
                      reanchor_every: int = 0,
                      area_growth_limit: float = 0.0,
                      area_clamp_pad: int = 10,
                      num_points: int = 30,
                      output_timestamp: bool = False,
                      output_category_prefix: bool = False,
                      cycle_consistency: bool = False,
                      chunk_size: int = 0,
                      iou_threshold: float = 0.5) -> Dict:
        """Process batch of images using video propagation.
        
        V18: Added cycle_consistency, chunk_size, iou_threshold parameters.
        """
        import torch
        from datetime import datetime
        
        # Generate timestamp for output naming
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") if output_timestamp else ""
        
        os.makedirs(output_dir, exist_ok=True)

        # v21: the template masks go through the SAME letterbox geometry as the template image, so the
        # masks SAM2 receives and the frames SAM2 sees are in one coordinate frame. `template_masks`
        # (native size) is still used for the visualisations drawn on the native template image.
        _canvas_side = int(getattr(self, "canvas_side", 0) or 0)
        template_masks_canvas = ([{**m, 'mask': _letterbox_mask(m['mask'], _canvas_side)} for m in template_masks]
                                 if _canvas_side else template_masks)
        if _canvas_side:
            self._canon_image(template_image_path)
            logger.info(f"v21 canonical frame: {_canvas_side}x{_canvas_side} letterbox"
                        f"{' + rembg foreground' if getattr(self, 'fg_session', None) is not None else ''}")

        if save_masks:
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
        
        
        if save_vis:
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            # Save template visualization once (mask + prompts)
            try:
                timg = cv2.imread(template_image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                #timg = cv2.imread(template_image_path)
                if timg is not None and len(template_masks) > 0:
                    tmask = template_masks[0]['mask']
                    if tmask.shape[:2] != timg.shape[:2]:
                        tmask = cv2.resize(tmask, (timg.shape[1], timg.shape[0]), interpolation=cv2.INTER_NEAREST)

                    tbox = get_box_from_mask(tmask, padding=5)
                    tpts = get_points_from_mask(tmask, int(num_points))
                    self.save_prompt_visualization(
                        timg, tmask, tpts, tbox,
                        os.path.join(vis_dir, 'vis_TEMPLATE_prompts.png')
                    )
                    cv2.imwrite(
                        os.path.join(vis_dir, 'vis_TEMPLATE_mask.png'),
                        overlay_mask(timg, tmask, color=(0,255,0), alpha=0.5),
                    )
            except Exception as e:
                logger.debug(f'Could not save template vis: {e}')

        # Precompute fallback prompts (used if tracker returns no mask for a frame)
        fallback_box = None
        fallback_points = None
        fallback_labels = None
        if len(template_masks_canvas) > 0:   # v21: prompts in canvas coordinates
            fallback_box = get_box_from_mask(template_masks_canvas[0]['mask'], padding=5)
            if fallback_box is not None:
                fallback_points = get_points_from_box(fallback_box, int(num_points))
            if fallback_points is None:
                fallback_points = get_points_from_mask(template_masks_canvas[0]['mask'], int(num_points))
            if fallback_points is not None:
                fallback_labels = np.ones(len(fallback_points), dtype=np.int32)

        temp_dir = tempfile.mkdtemp(prefix='pal_')
        confidence_scores = {}  # frame_idx -> IoU (only populated if cycle_consistency)
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # =================================================================
            # V18: Choose propagation strategy
            # =================================================================
            
            if chunk_size > 0:
                # --- Chunked mode: stream results, process each image immediately ---
                mode_str = "cycle-consistent chunked" if cycle_consistency else "chunked"
                logger.info(f"Using {mode_str} inference (chunk_size={chunk_size})")

                coco_output = {
                    'images': [],
                    'annotations': [],
                    'categories': list(self.categories.values())
                }
                stats = {'success': 0, 'failed': 0, 'total': len(target_image_paths), 'total_masks': 0}
                annotation_id_box = [1]  # mutable counter for closure

                def _on_image_result(global_idx, img_path, frame_masks, conf_val):
                    image_id = global_idx + 1
                    img_filename = os.path.basename(img_path)
                    orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    if orig_img is None:
                        stats['failed'] += 1
                        return
                    h, w = orig_img.shape[:2]

                    img_entry = {'id': image_id, 'file_name': img_filename, 'height': h, 'width': w}
                    if conf_val is not None:
                        img_entry['cycle_iou'] = round(conf_val, 4)
                        confidence_scores[global_idx] = conf_val
                    coco_output['images'].append(img_entry)

                    if not frame_masks:
                        try:
                            # v21: the fallback box is in canvas coordinates -> predict on the canvas image
                            img_rgb = (np.asarray(self._canon_image(img_path)[0]) if _canvas_side
                                       else cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                            with torch.inference_mode():
                                self.image_predictor.set_image(img_rgb)
                                masks_np, scores, _ = self.image_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=fallback_box,
                                    multimask_output=True,
                                )
                            if masks_np is not None and len(masks_np) > 0:
                                best_idx = int(np.argmax(scores))
                                pred = (masks_np[best_idx] > 0).astype(np.uint8)
                                if pred.sum() > 0:
                                    pred = postprocess_mask(pred)
                                    frame_masks = [{"obj_id": 1,
                                        "category_id": template_masks[0]["category_id"] if template_masks else 1,
                                        "mask": pred}]
                                    logger.warning(f"Image {img_filename}: tracker empty; used fallback.")
                                else:
                                    logger.warning(f"No result for image {img_filename}")
                                    stats['failed'] += 1
                                    return
                            else:
                                logger.warning(f"No result for image {img_filename}")
                                stats['failed'] += 1
                                return
                        except Exception as e:
                            logger.warning(f"No result for image {img_filename} - fallback failed: {e}")
                            stats['failed'] += 1
                            return

                    _lb = getattr(self, "_canon", {}).get(img_path)      # v21
                    if _canvas_side and _lb is not None:
                        # v21: canvas -> the target's native grid, IN PLACE, so every consumer below
                        # (mask PNGs, COCO JSON, vis_ overlay, pair_ side-by-side) sees native masks.
                        # (First cut only fixed a local copy -> the overlays stretched the square
                        # canvas onto the plate and looked shrunken/misplaced while the saved masks
                        # were already right.)
                        for _md in frame_masks:
                            _m = _md['mask']
                            if _m.shape[:2] == (_canvas_side, _canvas_side):
                                _md['mask'] = _unletterbox_mask(_m, _lb)
                    for mask_data in frame_masks:
                        mask = mask_data['mask']
                        category_id = mask_data['category_id']
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        ann = self.mask_to_coco_annotation(mask, annotation_id_box[0], image_id, category_id)
                        if ann:
                            if conf_val is not None:
                                ann['score'] = round(conf_val, 4)
                            if mask_data.get('obj_iou') is not None:   # [v20]
                                ann['obj_score'] = round(mask_data['obj_iou'], 4)
                            if mask_data.get('logit_conf') is not None:   # [v20]
                                ann['obj_conf'] = round(mask_data['logit_conf'], 4)
                            coco_output['annotations'].append(ann)
                            stats['total_masks'] += 1
                            annotation_id_box[0] += 1
                            if save_masks:
                                cat_name = self.categories.get(category_id, {}).get('name', f'cat_{category_id}')
                                base_name = os.path.splitext(img_filename)[0]
                                mask_filename = f"{cat_name}_{base_name}_mask.png" if output_category_prefix else f"{base_name}_{cat_name}_mask.png"
                                if timestamp_str:
                                    mask_filename = f"{timestamp_str}_{mask_filename}"
                                cv2.imwrite(os.path.join(masks_dir, mask_filename), mask * 255)

                    if save_vis and frame_masks:
                        vis_img = orig_img.copy()
                        _labels = []                                     # v21: fixed colour per category
                        for i, mask_data in enumerate(frame_masks):
                            mask = mask_data['mask']
                            if mask.shape[:2] != (h, w):
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            _cid = mask_data['category_id']
                            _labels.append((self.categories.get(_cid, {}).get('name', f'cat_{_cid}'), _cid))
                            color = _v21_colour_bgr(_cid)
                            vis_img[mask > 0] = (vis_img[mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                        vis_img = np.vstack([vis_img, _v21_legend(vis_img.shape[1],
                                                                  [(nm, _v21_colour_bgr(cid)) for nm, cid in _labels])])
                        cv2.imwrite(os.path.join(vis_dir, f"vis_{img_filename}"), vis_img)
                        try:
                            timg = cv2.imread(template_image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                            if timg is not None:
                                tmask = template_masks[0]['mask']
                                if tmask.shape[:2] != timg.shape[:2]:
                                    tmask = cv2.resize(tmask, (timg.shape[1], timg.shape[0]), interpolation=cv2.INTER_NEAREST)
                                current_masks = []
                                for md in frame_masks:
                                    m = md['mask']
                                    if m.shape[:2] != (h, w):
                                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                                    current_masks.append(m)
                                save_side_by_side(timg, tmask, orig_img, current_masks,
                                                  os.path.join(vis_dir, f"pair_{img_filename}"), labels=_labels)   # v21
                        except Exception as e:
                            logger.debug(f'Could not save side-by-side: {e}')
                    stats['success'] += 1

                self.propagate_masks_chunked(
                    template_image_path, target_image_paths, template_masks_canvas,   # v21
                    chunk_size=chunk_size,
                    interleave_template=interleave_template,
                    area_growth_limit=area_growth_limit,
                    area_clamp_pad=area_clamp_pad,
                    cycle_consistency=cycle_consistency,
                    on_image_result=_on_image_result
                )
                chunked_mode = True

            else:
                # --- Standard mode: single pseudo-video ---
                chunked_mode = False
                
                logger.info("Creating pseudo-video...")
                frames_dir = self.prepare_pseudo_video(
                    template_image_path, target_image_paths, temp_dir,
                    interleave_template=interleave_template
                )
                
                if cycle_consistency:
                    logger.info("Using 4-stage cycle-consistent inference...")
                    frame_results, confidence_scores = self.propagate_masks_cycle_consistent(
                        frames_dir, template_masks_canvas,   # v21
                        multi_mask=multi_mask,
                        interleave_template=interleave_template,
                        area_growth_limit=area_growth_limit,
                        area_clamp_pad=area_clamp_pad,
                        iou_threshold=iou_threshold
                    )
                else:
                    logger.info("Propagating masks through pseudo-video...")
                    frame_results = self.propagate_masks(
                        frames_dir, template_masks_canvas, multi_mask,   # v21
                        interleave_template=interleave_template,
                        reanchor_every=reanchor_every,
                        area_growth_limit=area_growth_limit,
                        area_clamp_pad=area_clamp_pad
                    )
            
            if not chunked_mode:
                # Non-chunked: convert to COCO format in a single pass
                coco_output = {
                    'images': [],
                    'annotations': [],
                    'categories': list(self.categories.values())
                }
                annotation_id = 1
                stats = {'success': 0, 'failed': 0, 'total': len(target_image_paths), 'total_masks': 0}

            # Process results (skip frame 0) — only for non-chunked mode
            # Chunked mode already processed via on_image_result callback above
            for img_idx, img_path in enumerate(target_image_paths):
                if chunked_mode:
                    break
                image_id = img_idx + 1
                img_filename = os.path.basename(img_path)

                # Standard mode uses pseudo-video frame index
                result_key = (2 * img_idx + 1) if interleave_template else image_id
                
                orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                #orig_img = cv2.imread(img_path)
                if orig_img is None:
                    stats['failed'] += 1
                    continue
                
                h, w = orig_img.shape[:2]
                
                # Build image entry
                img_entry = {
                    'id': image_id,
                    'file_name': img_filename,
                    'height': h,
                    'width': w
                }
                # Attach confidence score if available
                if result_key in confidence_scores:
                    img_entry['cycle_iou'] = round(confidence_scores[result_key], 4)
                
                coco_output['images'].append(img_entry)

                frame_masks = frame_results.get(result_key, [])
                if not frame_masks:
                    # Tracker failed for this frame -> fall back to single-image prompting
                    try:
                        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                        with torch.inference_mode():
                            self.image_predictor.set_image(img_rgb)
                            masks_np, scores, _ = self.image_predictor.predict(
                                point_coords=fallback_points,
                                point_labels=fallback_labels,
                                box=fallback_box,
                                multimask_output=True,
                            )
                        if masks_np is not None and len(masks_np) > 0:
                            best_idx = int(np.argmax(scores))
                            pred = (masks_np[best_idx] > 0).astype(np.uint8)
                            if pred.sum() > 0:
                                pred = postprocess_mask(pred)
                                frame_masks = [{
                                    "obj_id": 1,
                                    "category_id": template_masks[0]["category_id"] if len(template_masks) > 0 else 1,
                                    "mask": pred,
                                }]
                                logger.warning(
                                    f"Frame {result_key} (image {img_filename}): tracker returned no mask; used image-path fallback."
                                )
                            else:
                                logger.warning(f"No result for frame {result_key} (image {img_filename})")
                                stats["failed"] += 1
                                continue
                        else:
                            logger.warning(f"No result for frame {result_key} (image {img_filename})")
                            stats["failed"] += 1
                            continue
                    except Exception as e:
                        logger.warning(f"No result for frame {result_key} (image {img_filename}) - fallback failed: {e}")
                        stats["failed"] += 1
                        continue
                for mask_data in frame_masks:
                    mask = mask_data['mask']
                    category_id = mask_data['category_id']
                    
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    ann = self.mask_to_coco_annotation(mask, annotation_id, image_id, category_id)
                    
                    if ann:
                        # Attach per-annotation confidence if available
                        if result_key in confidence_scores:
                            ann['score'] = round(confidence_scores[result_key], 4)
                        coco_output['annotations'].append(ann)
                        stats['total_masks'] += 1
                        annotation_id += 1
                        
                        if save_masks:
                            cat_name = self.categories.get(category_id, {}).get('name', f'cat_{category_id}')
                            base_name = os.path.splitext(img_filename)[0]
                            if output_category_prefix:
                                mask_filename = f"{cat_name}_{base_name}_mask.png"
                            else:
                                mask_filename = f"{base_name}_{cat_name}_mask.png"
                            if timestamp_str:
                                mask_filename = f"{timestamp_str}_{mask_filename}"
                            cv2.imwrite(os.path.join(masks_dir, mask_filename), mask * 255)
                
                if save_vis and frame_masks:
                    vis_img = orig_img.copy()
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    for i, mask_data in enumerate(frame_masks):
                        mask = mask_data['mask']
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        color = colors[i % len(colors)]
                        vis_img[mask > 0] = (vis_img[mask > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                    cv2.imwrite(os.path.join(vis_dir, f"vis_{img_filename}"), vis_img)
                    # Side-by-side with template for drift debugging
                    try:
                        timg = cv2.imread(template_image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        #timg = cv2.imread(template_image_path)
                        if timg is not None:
                            tmask = template_masks[0]['mask']
                            if tmask.shape[:2] != timg.shape[:2]:
                                tmask = cv2.resize(tmask, (timg.shape[1], timg.shape[0]), interpolation=cv2.INTER_NEAREST)
                            current_masks = []
                            for md in frame_masks:
                                m = md['mask']
                                if m.shape[:2] != (h, w):
                                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                                current_masks.append(m)
                            save_side_by_side(timg, tmask, orig_img, current_masks, os.path.join(vis_dir, f"pair_{img_filename}"))
                    except Exception as e:
                        logger.debug(f'Could not save side-by-side: {e}')
                
                stats['success'] += 1
            
            # Save COCO JSON with optional timestamp
            if timestamp_str:
                output_json_path = os.path.join(output_dir, f'pal_predictions_{timestamp_str}.json')
            else:
                output_json_path = os.path.join(output_dir, 'pal_predictions.json')
            with open(output_json_path, 'w') as f:
                json.dump(coco_output, f, indent=2)
            
            # Save confidence report if cycle consistency was used
            if confidence_scores:
                conf_path = os.path.join(output_dir, 'cycle_consistency_report.json')
                report = {
                    'iou_threshold': iou_threshold,
                    'total_frames': len(confidence_scores),
                    'low_confidence_count': sum(1 for v in confidence_scores.values() if v < iou_threshold),
                    'mean_iou': round(np.mean(list(confidence_scores.values())), 4) if confidence_scores else 0,
                    'per_frame': {str(k): round(v, 4) for k, v in sorted(confidence_scores.items())}
                }
                with open(conf_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Cycle consistency report: {conf_path}")
            
            logger.info(f"Processing complete!")
            logger.info(f"  Success: {stats['success']}/{stats['total']}")
            logger.info(f"  Total masks: {stats['total_masks']}")
            if confidence_scores:
                mean_iou = np.mean(list(confidence_scores.values()))
                logger.info(f"  Mean cycle IoU: {mean_iou:.4f}")
            
            return {
                'coco_output': coco_output,
                'output_path': output_json_path,
                'stats': stats,
                'confidence': confidence_scores
            }
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

# =============================================================================
# Orientation search (opt-in, off by default)
# SAM2-PAL is anchored to the template's orientation: on 14 held-out ant heads mean IoU fell from 0.78 upright
# to 0.35 upside-down and 0.14 sideways, and flip-augmented training did not fix it. Orientation search runs
# every target at 0/90/180/270 degrees, keeps the rotation with the highest mean SAM2 object confidence
# (obj_conf; picked the right rotation 14/14, 14/14, 28/28 on that test and restored sideways heads to 0.78)
# and rotates the masks back onto the original image. The palindrome cycle IoU is NOT used to choose: it
# was 0.996 for every orientation. Rotations never change handedness, so left_/right_ labels stay valid; a
# specimen photographed MIRRORED cannot be detected this way - image in the template's orientation.
# =============================================================================
_ORIENT_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_ORIENT_UNROT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def run_orientation_search(pal, template_image_path: str, template_masks: List[Dict],
                           target_image_paths: List[str], output_dir: str, save_masks: bool = True,
                           save_vis: bool = False, output_timestamp: bool = False,
                           output_category_prefix: bool = False, **batch_kwargs) -> Dict:
    """Predict every target at 0/90/180/270 deg (clockwise), keep the most confident rotation per image and
    write its masks, rotated back, as a normal pal_predictions JSON for the ORIGINAL images."""
    from datetime import datetime
    work = os.path.join(output_dir, 'orientation_search')
    var_dir, raw_dir = os.path.join(work, 'variants'), os.path.join(work, 'raw_predictions')
    os.makedirs(var_dir, exist_ok=True)
    variants, meta = [], {}
    for p in target_image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            logger.warning(f"Orientation search: cannot read {p}; skipped")
            continue
        for k in (0, 90, 180, 270):
            vname = f"{os.path.basename(p)}__o{k:03d}.png"
            vpath = os.path.join(var_dir, vname)
            if not os.path.exists(vpath):
                cv2.imwrite(vpath, img if k == 0 else cv2.rotate(img, _ORIENT_ROT[k]))
            variants.append(vpath)
            meta[vname] = (p, k)
    logger.info(f"Orientation search: {len(target_image_paths)} images x 4 rotations = {len(variants)} predictions")

    raw = pal.process_batch(template_image_path=template_image_path, template_masks=template_masks,
                            target_image_paths=variants, output_dir=raw_dir, save_masks=False, save_vis=False,
                            output_timestamp=False, output_category_prefix=output_category_prefix, **batch_kwargs)
    rc = raw['coco_output']
    anns_by_img = {}
    for a in rc['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
    per_orig = {}
    for im in rc['images']:
        p, k = meta[im['file_name']]
        anns = anns_by_img.get(im['id'], [])
        conf = [a.get('obj_conf', a.get('score', 0.0)) for a in anns]
        per_orig.setdefault(p, {})[k] = (float(np.mean(conf)) if conf else -1.0, im, anns)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") if output_timestamp else ""
    coco_output = {'images': [], 'annotations': [], 'categories': rc['categories']}
    stats = {'success': 0, 'failed': 0, 'total': len(target_image_paths), 'total_masks': 0}
    confidence, report, ann_id = {}, {}, 1
    if save_masks:
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    if save_vis:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    for idx, p in enumerate(target_image_paths):
        if p not in per_orig:
            stats['failed'] += 1
            continue
        scores = per_orig[p]
        best = max(scores, key=lambda r: scores[r][0])
        _, vim, anns = scores[best]
        orig = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h, w = orig.shape[:2]
        fname = os.path.basename(p)
        entry = {'id': idx + 1, 'file_name': fname, 'height': h, 'width': w, 'orientation_used': best,
                 'orientation_confidence': {str(r): round(scores[r][0], 4) for r in sorted(scores)}}
        if 'cycle_iou' in vim:
            entry['cycle_iou'] = vim['cycle_iou']
            confidence[idx] = vim['cycle_iou']
        coco_output['images'].append(entry)
        report[fname] = {'orientation_used': best, **entry['orientation_confidence']}
        if best != 0:
            logger.info(f"Orientation search: {fname} predicted at {best} deg clockwise (masks rotated back)")
        vis_masks = []
        for a in anns:
            m = np.zeros((vim['height'], vim['width']), np.uint8)
            for poly in a['segmentation']:
                cv2.fillPoly(m, [np.array(poly, np.int32).reshape(-1, 2)], 1)
            if best:
                m = cv2.rotate(m, _ORIENT_UNROT[best])
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            new = pal.mask_to_coco_annotation(m, ann_id, idx + 1, a['category_id'])
            if not new:
                continue
            for key in ('score', 'obj_score', 'obj_conf'):
                if key in a:
                    new[key] = a[key]
            coco_output['annotations'].append(new)
            stats['total_masks'] += 1
            ann_id += 1
            cat_name = pal.categories.get(a['category_id'], {}).get('name', f"cat_{a['category_id']}")
            vis_masks.append((m, cat_name, a['category_id']))
            if save_masks:
                base = os.path.splitext(fname)[0]
                mf = f"{cat_name}_{base}_mask.png" if output_category_prefix else f"{base}_{cat_name}_mask.png"
                cv2.imwrite(os.path.join(output_dir, 'masks', f"{timestamp_str}_{mf}" if timestamp_str else mf), m * 255)
        if save_vis and vis_masks:
            vis = orig.copy()
            for m, _, cid in vis_masks:
                vis[m > 0] = (vis[m > 0] * 0.5 + np.array(_v21_colour_bgr(cid)) * 0.5).astype(np.uint8)
            vis = np.vstack([vis, _v21_legend(vis.shape[1], [(nm, _v21_colour_bgr(cid)) for _, nm, cid in vis_masks])])
            cv2.imwrite(os.path.join(output_dir, 'visualizations', f"vis_{fname}"), vis)
        stats['success'] += 1

    out = os.path.join(output_dir, f'pal_predictions_{timestamp_str}.json' if timestamp_str else 'pal_predictions.json')
    with open(out, 'w') as f:
        json.dump(coco_output, f, indent=2)
    with open(os.path.join(output_dir, 'orientation_search_report.json'), 'w') as f:
        json.dump({'selector': 'mean obj_conf', 'rotations_cw': [0, 90, 180, 270], 'per_image': report}, f, indent=2)
    turned = sum(1 for r in report.values() if r['orientation_used'] != 0)
    logger.info(f"Orientation search: {turned}/{len(report)} images predicted in a rotated orientation; "
                f"report -> orientation_search_report.json")
    return {'coco_output': coco_output, 'output_path': out, 'stats': stats, 'confidence': confidence}


def main():
    parser = argparse.ArgumentParser(
        description='SAM2-PAL: Palindrome-based Mask Propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
=== HOW IT WORKS ===

PSEUDO-VIDEO APPROACH:
  Frame 0: Template + mask â†' stored in SAM2's memory
  Frame 1-N: Target images â†' masks propagated via tracking
  
FINE-TUNING:
  Trains mask_decoder and prompt_encoder using augmented template.
  Uses SAM2ImagePredictor internals (which support gradients).

Examples:
  # Basic inference (try first!)
  python sam2_pal_batch.py --template_mask mask.png \\
                           --template_image template.jpg \\
                           --image_dir ./images \\
                           --output_dir ./output \\
                           --save_vis

  # With fine-tuning (for challenging structures)
  python sam2_pal_batch.py --template_mask mask.png \\
                           --template_image template.jpg \\
                           --image_dir ./images \\
                           --output_dir ./output \\
                           --finetune --num_epochs 200 --save_vis
        """
    )
    
    # Template source
    template_group = parser.add_argument_group('Template Source')
    template_group.add_argument('--template_mask', help='Binary mask PNG')
    template_group.add_argument('--template_json', help='COCO JSON')
    template_group.add_argument('--template_image', required=True, help='Template image')
    template_group.add_argument('--category_name', default='object', help='Category name')
    
    # Input/Output
    parser.add_argument('--image_dir', required=True, help='Target images directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    # SAM2
    parser.add_argument('--sam2_checkpoint', required=True, help='SAM2 checkpoint')
    parser.add_argument('--sam2_config', required=True, help='SAM2 config YAML')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    
    # Output
    parser.add_argument('--save_masks', action='store_true', default=True)
    parser.add_argument('--no_save_masks', action='store_false', dest='save_masks')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--multi_mask', action='store_true', help='Enable multi-instance output')

    # Drift control / stabilization
    parser.add_argument('--interleave_template', action='store_true',
                        help='Use [T, U1, T, U2, ...] frame sequence to reduce drift')
    parser.add_argument('--reanchor_every', type=int, default=0,
                        help='(Best-effort) Re-add predicted mask every N frames during propagation (0=off)')
    parser.add_argument('--area_growth_limit', type=float, default=2.0,
                        help='Clamp masks if area grows > limit * previous area (0 disables)')
    parser.add_argument('--area_clamp_pad', type=int, default=10,
                        help='Padding (pixels) around previous bbox used for area clamp')
    
    # Fine-tuning
    finetune_group = parser.add_argument_group('Fine-tuning')
    finetune_group.add_argument('--finetune', action='store_true', help='Enable fine-tuning (augmentation-based)')
    finetune_group.add_argument('--pal_finetuning', action='store_true', 
                                help='Enable PAL fine-tuning (video tracker backprop - most powerful)')
    finetune_group.add_argument('--finetune_occcl', action='store_true', 
                                help='Alias for --pal_finetuning (deprecated name)')
    finetune_group.add_argument('--flip_augment', default='none', choices=['none', 'h', 'v', 'hv', 'all'],
                                help='PAL fine-tuning: add flipped copies of the labelled training images '
                                     '(h = left-right, v = top-bottom, hv = both, all = original+h+v+hv = 4x). '
                                     'For specimens photographed as mirror images. Default none.')
    finetune_group.add_argument('--use_lora', action='store_true',
                                help='Use LoRA fine-tuning (requires peft library, per paper)')
    finetune_group.add_argument('--lora_rank', type=int, default=16,
                                help='LoRA rank (default 16, paper does not specify)')
    finetune_group.add_argument('--num_epochs', type=int, default=200, help='Training epochs (paper uses 25 with LoRA)')
    finetune_group.add_argument('--learning_rate', type=float, default=None,
                                help='Learning rate (default: 1e-4 for LoRA, 1e-5 for full fine-tune)')
    finetune_group.add_argument('--num_points', type=int, default=3, help='Points per sample')
    finetune_group.add_argument('--finetune_checkpoint', help='Checkpoint path')
    finetune_group.add_argument('--max_images_per_epoch', type=int, default=10,
                                help='Max unlabeled images per epoch for PAL fine-tuning')
    
    # Multi-template training (NEW in v13)
    training_group = parser.add_argument_group('Multi-template Training (for PAL fine-tuning)')
    training_group.add_argument('--training_json', help='COCO JSON with multiple annotated training images')
    training_group.add_argument('--training_images_dir', help='Directory containing training images')
    training_group.add_argument('--training_masks_dir', help='Directory containing binary mask PNGs (alternative to JSON)')
    
    # Output naming options (NEW in v14)
    naming_group = parser.add_argument_group('Output Naming (NEW in v14)')
    naming_group.add_argument('--output_timestamp', action='store_true',
                              help='Add timestamp to output filenames (e.g., pal_predictions_20260113_143022.json)')
    naming_group.add_argument('--output_category_prefix', action='store_true',
                              help='Include category name in mask filenames')
    
    # V18: Inference strategy
    inference_group = parser.add_argument_group('Inference Strategy (NEW in v18)')
    inference_group.add_argument('--cycle_consistency', action='store_true',
                                 help='Use 4-stage cycle-consistent inference (forward+backward+IoU)')
    inference_group.add_argument('--chunk_size', type=int, default=0,
                                 help='Process images in chunks of N (0=disabled, all at once with CPU offloading)')
    inference_group.add_argument('--iou_threshold', type=float, default=0.5,
                                 help='IoU threshold for cycle consistency flagging (default 0.5)')
    inference_group.add_argument('--orientation_search', default='none', choices=['none', 'rot4'],
                                 help='OFF by default. rot4: predict each target at 0/90/180/270 deg, keep the most '
                                      'confident rotation and map its masks back (4x slower). A fallback for '
                                      'specimens imaged turned relative to the template; cannot correct mirrored '
                                      'imaging. Best practice: image every specimen in the template orientation.')
    inference_group.add_argument('--load_checkpoint',
                                 help='Load an existing fine-tuned .pt checkpoint for inference-only '
                                      '(skip training, use this model for propagation)')

    # v21: canonical frame (ON by default - this is the ossified fix for the recurring template/target
    # dimension mismatch: every image is foreground-extracted and letterboxed into one square)
    canon_group = parser.add_argument_group('Canonical frame (v21, on by default)')
    canon_group.add_argument('--canvas_side', type=int, default=2048,
                             help='side of the square canvas every image (template + targets) is '
                                  'letterboxed into, aspect preserved, Lanczos (default 2048)')
    canon_group.add_argument('--fg_mask', choices=['rembg', 'none'], default='rembg',
                             help='foreground extraction before letterboxing (default rembg)')
    canon_group.add_argument('--rembg_model', default='isnet-general-use')
    canon_group.add_argument('--fg_fill', default='255,255,255',
                             help='R,G,B fill for background and letterbox padding (default white)')
    canon_group.add_argument('--no_canon', action='store_true',
                             help='pre-v21 behaviour: stretch targets to the template size, no foreground '
                                  'extraction. Only for reproducing old runs.')

    args = parser.parse_args()
    
    # Validate
    if not args.template_mask and not args.template_json:
        parser.error("Must provide --template_mask or --template_json")
    
    # Set default learning rate based on fine-tuning method
    if args.learning_rate is None:
        if args.use_lora:
            args.learning_rate = 1e-4  # Paper's default for LoRA
            logger.info(f"Using LoRA default learning rate: {args.learning_rate}")
        else:
            args.learning_rate = 1e-5  # Conservative default for full fine-tuning
            logger.info(f"Using full fine-tune default learning rate: {args.learning_rate}")
    
    # Get target images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_dir = Path(args.image_dir)
    target_images = [
        str(f) for f in sorted(image_dir.iterdir())
        if f.suffix.lower() in image_extensions
        and not f.name.startswith('._')
    ]
    
    # Exclude template
    template_name = os.path.basename(args.template_image)
    target_images = [p for p in target_images if os.path.basename(p) != template_name]
    
    if not target_images:
        logger.error("No target images found!")
        sys.exit(1)
    
    logger.info(f"Found {len(target_images)} target images")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize
    try:
        pal = SAM2PAL(
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize SAM2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # v21: wire the canonical frame into the model object (prepare_pseudo_video / process_batch read it)
    pal.canvas_side = 0 if args.no_canon else int(args.canvas_side)
    pal.fg_fill = tuple(int(v) for v in args.fg_fill.split(','))
    pal.canon_cache_dir = os.path.join(args.output_dir, 'canon_cache')
    pal.fg_session = None
    if pal.canvas_side and args.fg_mask == 'rembg':
        try:
            pal.fg_session = _rembg_session(args.rembg_model)
            logger.info(f"v21: rembg foreground extraction ON ({args.rembg_model}), cache -> {pal.canon_cache_dir}")
        except Exception as e:
            logger.warning(f"v21: rembg unavailable ({e}); continuing WITHOUT foreground extraction")
    if pal.canvas_side and int(getattr(args, 'chunk_size', 0) or 0) == 0:
        args.chunk_size = 1
        logger.info("v21: canonical frame needs the chunked path; --chunk_size set to 1")
    logger.info(f"v21: canonical frame {'OFF (--no_canon)' if not pal.canvas_side else f'{pal.canvas_side}x{pal.canvas_side}, fill {pal.fg_fill}'}")

    # Load template
    if args.template_mask:
        mask = pal.load_template_mask(args.template_mask, args.category_name)
        template_masks = [{'mask': mask, 'category_id': 1}]
    else:
        template_masks = pal.load_template_from_coco(args.template_json, args.template_image)
    
    if not template_masks:
        logger.error("No template masks!")
        sys.exit(1)
    
    
    # Optional: save a debug visualization of the template prompt (points + box)
    if getattr(args, "save_vis", False):
        try:
            if getattr(pal, "template_mask", None) is not None:
                tmask = (pal.template_mask > 0).astype(np.uint8)
                # Use a modest padding for visualization; this is not the area clamp pad
                tbox = get_box_from_mask(tmask, padding=15)
                tpts = get_points_from_mask(tmask, args.num_points)  # Use standalone function
                dbg_path = os.path.join(args.output_dir, "template_prompt_debug.png")
                pal.save_prompt_visualization(
                    template_image=template_image,
                    mask=tmask,
                    points=tpts,
                    box=tbox,
                    output_path=dbg_path,
                    title=f"Template prompt (N={args.num_points} pts + box)",
                )
                logger.info(f"Saved template prompt debug: {dbg_path}")
        except Exception as e:
            logger.warning(f"Could not save template prompt debug: {e}")

# Fine-tune
    # Handle both --pal_finetuning and --finetune_occcl (alias)
    use_pal_finetuning = args.pal_finetuning or args.finetune_occcl
    
    if use_pal_finetuning:
        # === FIX v17: Allow COCO JSON OR binary mask for PAL fine-tuning ===
        if not args.template_mask and len(template_masks) == 0:
            logger.error("PAL fine-tuning requires --template_mask OR --template_json with masks")
            sys.exit(1)
        
        if len(template_masks) == 0 and args.template_mask:
            logger.error("No masks loaded. Check your template_mask file.")
            sys.exit(1)
        
        logger.info(f"PAL fine-tuning with {len(template_masks)} template mask(s)")
        
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        
        # Load additional training templates if provided
        additional_templates = None
        if args.training_json or args.training_masks_dir:
            additional_templates = pal.load_training_templates(
                training_json=args.training_json,
                training_images_dir=args.training_images_dir,
                training_masks_dir=args.training_masks_dir
            )
            if additional_templates:
                logger.info(f"Loaded {len(additional_templates)} additional training templates from training JSON")
        
        # === FIX v17: Include ALL template masks in training, not just first ===
        # If template_masks has multiple masks (e.g., from COCO JSON with scape, antenna, eye),
        # add masks 1+ to additional_templates so they all get trained
        if len(template_masks) > 1:
            logger.info(f"Multi-mask template detected: {len(template_masks)} masks will be used for training")
            if additional_templates is None:
                additional_templates = []
            # Add masks 1, 2, ... to additional templates (mask 0 is the primary)
            for i, mask_data in enumerate(template_masks[1:], start=2):
                additional_templates.append({
                    'image_path': args.template_image,
                    'mask': mask_data['mask'],
                    'category_id': mask_data.get('category_id', i)
                })
                cat_name = pal.categories.get(mask_data.get('category_id', i), {}).get('name', f'mask_{i}')
                logger.info(f"  Added template mask {i}: {cat_name}")
        
        pal.finetune_pal(
            template_image_path=args.template_image,
            template_mask=template_masks[0]['mask'],
            target_image_paths=target_images,
            output_checkpoint=ckpt,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_images_per_epoch=args.max_images_per_epoch,
            additional_templates=additional_templates,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            flip_augment=args.flip_augment
        )
    
    elif args.finetune:
        if not args.template_mask and len(template_masks) == 0:
            logger.error("Fine-tuning requires --template_mask or --template_json with masks")
            sys.exit(1)
        
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        
        # Log multi-mask info
        if len(template_masks) > 1:
            logger.info(f"Note: Legacy fine-tune mode uses first mask only. Use --pal_finetuning for multi-mask training.")
        
        pal.finetune(
            template_image_path=args.template_image,
            template_mask=template_masks[0]['mask'],
            output_checkpoint=ckpt,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            num_points=args.num_points
        )
    
    # Process
    # === V18: Load existing checkpoint for inference-only mode ===
    # If no training was performed but a checkpoint was specified, load it now.
    # This allows: train once → save .pt → reuse for inference on new images
    use_pal_finetuning_done = use_pal_finetuning  # Was training done above?
    finetune_done = args.finetune
    
    if not use_pal_finetuning_done and not finetune_done:
        # No training was run — check if user wants to load an existing checkpoint
        load_ckpt = args.load_checkpoint or args.finetune_checkpoint
        if load_ckpt and os.path.exists(load_ckpt):
            logger.info(f"Loading existing checkpoint for inference: {load_ckpt}")
            pal._load_finetuned_weights(load_ckpt)
            logger.info("Fine-tuned weights loaded successfully")
        elif load_ckpt:
            logger.warning(f"Checkpoint not found: {load_ckpt} — using original SAM2 weights")
        else:
            logger.info("No fine-tuning or checkpoint specified — using original SAM2 weights")

    if getattr(args, 'orientation_search', 'none') == 'rot4':
        results = run_orientation_search(
            pal, args.template_image, template_masks, target_images, args.output_dir,
            save_masks=args.save_masks, save_vis=args.save_vis, output_timestamp=args.output_timestamp,
            output_category_prefix=args.output_category_prefix,
            multi_mask=args.multi_mask, interleave_template=args.interleave_template,
            reanchor_every=args.reanchor_every, area_growth_limit=args.area_growth_limit,
            area_clamp_pad=args.area_clamp_pad, num_points=args.num_points,
            cycle_consistency=args.cycle_consistency, chunk_size=args.chunk_size,
            iou_threshold=args.iou_threshold)
    else:
      results = pal.process_batch(
        template_image_path=args.template_image,
        template_masks=template_masks,
        target_image_paths=target_images,
        output_dir=args.output_dir,
        save_masks=args.save_masks,
        save_vis=args.save_vis,
        multi_mask=args.multi_mask,
        interleave_template=args.interleave_template,
        reanchor_every=args.reanchor_every,
        area_growth_limit=args.area_growth_limit,
        area_clamp_pad=args.area_clamp_pad,
        num_points=args.num_points,
        output_timestamp=args.output_timestamp,
        output_category_prefix=args.output_category_prefix,
        cycle_consistency=args.cycle_consistency,
        chunk_size=args.chunk_size,
        iou_threshold=args.iou_threshold
    )
    
    # Summary
    stats = results['stats']
    print("\n" + "="*60)
    print("SAM2-PAL v19 Processing Complete!")
    print("="*60)
    print(f"Images processed: {stats['success']}/{stats['total']}")
    print(f"Total masks: {stats['total_masks']}")
    print(f"Output: {results['output_path']}")
    if args.cycle_consistency:
        conf = results.get('confidence', {})
        if conf:
            mean_iou = np.mean(list(conf.values()))
            low = sum(1 for v in conf.values() if v < args.iou_threshold)
            print(f"Cycle consistency: mean IoU={mean_iou:.4f}, low-confidence frames={low}")
    if use_pal_finetuning:
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        print(f"Checkpoint (PAL): {ckpt}")
    elif args.finetune:
        ckpt = args.finetune_checkpoint or os.path.join(args.output_dir, 'finetuned_sam2_pal.pt')
        print(f"Checkpoint: {ckpt}")
    elif args.load_checkpoint or (args.finetune_checkpoint and not use_pal_finetuning and not args.finetune):
        loaded = args.load_checkpoint or args.finetune_checkpoint
        if os.path.exists(loaded):
            print(f"Loaded checkpoint: {loaded}")
        else:
            print(f"Checkpoint not found (used original weights): {loaded}")
    print("="*60)
    print("\nUse 'View Predictions' in Descriptron to review results")


if __name__ == '__main__':
    main()
