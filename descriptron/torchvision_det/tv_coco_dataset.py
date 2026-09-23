#!/usr/bin/env python3
"""
tv_coco_dataset.py — COCO JSON into the targets torchvision's detectors expect
=============================================================================

Detectron2 read your COCO file for you (`register_coco_instances`). torchvision
hands you the model and nothing else, so this module is the part that was in the
box before: it turns the same COCO JSON into the dicts that
`maskrcnn_resnet50_fpn_v2` and `keypointrcnn_resnet50_fpn` take, and it applies
the same augmentations the Detectron2 recipe used.

Two deliberate choices, both worth knowing:

* **The geometry is written by hand.** torchvision 0.19's `transforms.v2` can
  carry images, masks and boxes through a rotation, but there is no `KeyPoints`
  tv_tensor before 0.22, so keypoints would be left behind. Every geometric
  operation here therefore builds one affine matrix and applies it to the image,
  the masks, the boxes and the keypoints together. Nothing can fall out of step.
* **No horizontal flip.** The Detectron2 recipe did not flip either, and that is
  correct for this data rather than an oversight: categories such as left_scape
  and right_mandible are sides, and a flipped wing is a different hand.

Mosaic augmentation is included (`--mosaic_p`). It could not be done in
Detectron2 without replacing its data loader; here it is ordinary Dataset code.
It is off by default because it is not obviously right for this material — see
`MosaicMixer` for what it costs.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

# ── reading the file ──────────────────────────────────────────────────────────


def load_coco(path):
    """Read a COCO JSON and index it by image."""
    with open(path) as fh:
        d = json.load(fh)
    by_image = defaultdict(list)
    for a in d.get("annotations", []):
        by_image[a["image_id"]].append(a)
    images = {im["id"]: im for im in d.get("images", [])}
    cats = sorted(d.get("categories", []), key=lambda c: c["id"])
    return d, images, by_image, cats


def category_maps(cats):
    """COCO category ids are arbitrary; the model wants 1..C with 0 = background."""
    coco_to_contig = {c["id"]: i + 1 for i, c in enumerate(cats)}
    contig_to_name = {i + 1: c["name"] for i, c in enumerate(cats)}
    return coco_to_contig, contig_to_name


def ann_to_mask(ann, height, width):
    """Polygon, uncompressed RLE or compressed RLE -> uint8 HxW. None if unusable."""
    seg = ann.get("segmentation")
    if not seg:
        return None
    try:
        if isinstance(seg, list):
            if not any(len(p) >= 6 for p in seg):
                return None
            rles = mask_utils.frPyObjects([p for p in seg if len(p) >= 6], height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, height, width)
        else:
            rle = seg
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m.max(axis=2)
        return m.astype(np.uint8)
    except Exception:          # a malformed polygon is skipped, not fatal
        return None


def boxes_from_masks(masks):
    """Tight xyxy box of each mask. Empty masks give a degenerate box, caller drops it."""
    out = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
        else:
            out.append([float(xs.min()), float(ys.min()),
                        float(xs.max()) + 1.0, float(ys.max()) + 1.0])
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def boxes_from_keypoints(kps, width, height, pad_frac=0.10, min_pad=8.0):
    """
    Keypoint R-CNN is a detector: it needs a box per instance before it can place
    keypoints in it. `diaphorina_forewing_keypoints.json` carries no boxes at all
    (97 annotations, keypoints only), so the box is derived from the extent of the
    visible keypoints, padded, and clipped to the image.
    """
    out = []
    for kp in kps:
        vis = kp[kp[:, 2] > 0]
        if vis.shape[0] == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        x0, y0 = vis[:, 0].min(), vis[:, 1].min()
        x1, y1 = vis[:, 0].max(), vis[:, 1].max()
        px = max(min_pad, (x1 - x0) * pad_frac)
        py = max(min_pad, (y1 - y0) * pad_frac)
        out.append([max(0.0, x0 - px), max(0.0, y0 - py),
                    min(float(width), x1 + px), min(float(height), y1 + py)])
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


# ── geometry: one matrix, applied to everything ───────────────────────────────


def rotation_matrix(width, height, angle_deg, expand=True):
    """
    2x3 affine for a rotation about the image centre. `expand=True` grows the
    canvas so no content is cut, which is what Detectron2's RandomRotation does
    by default — matching it keeps the two arms of the comparison honest.
    """
    a = math.radians(angle_deg)
    cos_a, sin_a = abs(math.cos(a)), abs(math.sin(a))
    cx, cy = width / 2.0, height / 2.0
    m = np.array([[math.cos(a), math.sin(a), 0.0],
                  [-math.sin(a), math.cos(a), 0.0]], dtype=np.float64)
    if expand:
        new_w = int(round(height * sin_a + width * cos_a))
        new_h = int(round(height * cos_a + width * sin_a))
    else:
        new_w, new_h = width, height
    # send the old centre to the new centre
    m[0, 2] = new_w / 2.0 - (m[0, 0] * cx + m[0, 1] * cy)
    m[1, 2] = new_h / 2.0 - (m[1, 0] * cx + m[1, 1] * cy)
    return m.astype(np.float32), new_w, new_h


def apply_affine_points(pts, m):
    """pts (...,2) -> affine -> (...,2)."""
    p = np.asarray(pts, dtype=np.float32)
    shape = p.shape
    flat = p.reshape(-1, 2)
    out = flat @ m[:, :2].T + m[:, 2]
    return out.reshape(shape)


def warp(image, masks, kps, m, out_w, out_h):
    """Apply one affine to the image, every mask and every keypoint set."""
    import cv2
    img = cv2.warpAffine(image, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    new_masks = None
    if masks is not None and len(masks):
        new_masks = np.stack([
            cv2.warpAffine(mk, m, (out_w, out_h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for mk in masks]).astype(np.uint8)
    new_kps = None
    if kps is not None and len(kps):
        new_kps = kps.copy()
        new_kps[:, :, :2] = apply_affine_points(kps[:, :, :2], m)
        outside = ((new_kps[:, :, 0] < 0) | (new_kps[:, :, 0] >= out_w) |
                   (new_kps[:, :, 1] < 0) | (new_kps[:, :, 1] >= out_h))
        new_kps[outside, 2] = 0          # rotated out of frame -> not visible
    return img, new_masks, new_kps


def colour_jitter(image, brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                  saturation=(0.8, 1.2), rng=random):
    """
    The photometric half of the Detectron2 recipe. Applied to pixels only, so no
    geometry can drift; the order matches detectron2.data.transforms.
    """
    import cv2
    img = image.astype(np.float32)
    img *= rng.uniform(*brightness)
    mean = img.mean()
    img = (img - mean) * rng.uniform(*contrast) + mean
    grey = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    grey = grey[:, :, None].astype(np.float32)
    img = grey + (img - grey) * rng.uniform(*saturation)
    return np.clip(img, 0, 255).astype(np.uint8)


# ── the dataset ───────────────────────────────────────────────────────────────


class CocoInstanceDataset(Dataset):
    """
    One COCO JSON -> the (image, target) pairs torchvision's detectors take.

    task="masks"      target: boxes, labels, masks
    task="keypoints"  target: boxes, labels, keypoints  (boxes derived if absent)

    Images whose annotations are all unusable are dropped at construction, which
    is what Detectron2's DATALOADER.FILTER_EMPTY_ANNOTATIONS = True did.
    """

    def __init__(self, coco_json, img_dir, task="masks", image_ids=None,
                 augment=False, mosaic_p=0.0, mosaic_size=512, rotation=45.0,
                 kp_box_pad=0.10, seed=0, category_ids=None):
        self.img_dir = Path(img_dir)
        self.task = task
        self.augment = augment
        self.mosaic_p = float(mosaic_p)
        self.mosaic_size = int(mosaic_size)
        self.rotation = float(rotation)
        self.kp_box_pad = float(kp_box_pad)
        self.rng = random.Random(seed)

        raw, self.images, by_image, cats = load_coco(coco_json)
        if category_ids is not None:
            cats = [c for c in cats if c["id"] in set(category_ids)]
        self.categories = cats
        self.coco_to_contig, self.contig_to_name = category_maps(cats)
        self.num_classes = len(cats) + 1          # + background

        keep = set(self.coco_to_contig)
        wanted = set(image_ids) if image_ids is not None else None
        self.samples = []
        for image_id, anns in by_image.items():
            if wanted is not None and image_id not in wanted:
                continue
            if image_id not in self.images:
                continue
            usable = [a for a in anns
                      if a.get("category_id") in keep and not a.get("iscrowd", 0)
                      and (a.get("segmentation") if task == "masks" else a.get("keypoints"))]
            if usable:
                self.samples.append((image_id, usable))
        self.samples.sort(key=lambda s: s[0])

        # how many keypoints the model must predict, from the category definition
        self.num_keypoints = 0
        if task == "keypoints":
            counts = {len(c.get("keypoints", []) or []) for c in cats}
            counts.discard(0)
            if len(counts) > 1:
                raise ValueError(
                    f"categories declare different keypoint counts {sorted(counts)}; "
                    "Keypoint R-CNN has one keypoint head, so train them separately")
            self.num_keypoints = counts.pop() if counts else 0

    def __len__(self):
        return len(self.samples)

    # ---- raw decoding, before any augmentation ----

    def _load_raw(self, i):
        image_id, anns = self.samples[i]
        meta = self.images[image_id]
        path = self.img_dir / Path(meta["file_name"]).name
        if not path.exists():                       # COCO file_name may carry a subpath
            alt = self.img_dir / meta["file_name"]
            path = alt if alt.exists() else path
        img = np.asarray(Image.open(path).convert("RGB"))
        h, w = img.shape[:2]

        labels, masks, kps = [], [], []
        for a in anns:
            if self.task == "masks":
                m = ann_to_mask(a, h, w)
                if m is None or m.sum() == 0:
                    continue
                masks.append(m)
            else:
                kp = np.asarray(a["keypoints"], dtype=np.float32).reshape(-1, 3)
                if self.num_keypoints and kp.shape[0] != self.num_keypoints:
                    continue
                if (kp[:, 2] > 0).sum() == 0:
                    continue
                kps.append(kp)
            labels.append(self.coco_to_contig[a["category_id"]])

        masks = np.stack(masks) if masks else np.zeros((0, h, w), np.uint8)
        kps = np.stack(kps) if kps else np.zeros((0, max(self.num_keypoints, 1), 3), np.float32)
        return img, np.asarray(labels, np.int64), masks, kps, image_id

    # ---- mosaic ----

    def _mosaic(self, i):
        """
        Four images tiled into one canvas at a random split, annotations carried
        across. Worth knowing what it costs here: each specimen ends up at about
        half its linear size, and half a wing can end up beside half a head, so
        the model sees scales and adjacencies that never occur at inference. That
        is why it is off by default and why the trainer closes it for the last
        stretch of training.
        """
        import cv2
        s = self.mosaic_size
        canvas = np.zeros((s * 2, s * 2, 3), np.uint8)
        xc = int(self.rng.uniform(s * 0.5, s * 1.5))
        yc = int(self.rng.uniform(s * 0.5, s * 1.5))
        picks = [i] + [self.rng.randrange(len(self.samples)) for _ in range(3)]

        all_labels, all_masks, all_kps = [], [], []
        for q, idx in enumerate(picks):
            img, labels, masks, kps, _ = self._load_raw(idx)
            h0, w0 = img.shape[:2]
            scale = s / max(h0, w0)
            w, h = max(1, int(w0 * scale)), max(1, int(h0 * scale))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

            if q == 0:      # top-left
                xa1, ya1, xa2, ya2 = max(xc - w, 0), max(yc - h, 0), xc, yc
                xb1, yb1 = w - (xa2 - xa1), h - (ya2 - ya1)
            elif q == 1:    # top-right
                xa1, ya1, xa2, ya2 = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                xb1, yb1 = 0, h - (ya2 - ya1)
            elif q == 2:    # bottom-left
                xa1, ya1, xa2, ya2 = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                xb1, yb1 = w - (xa2 - xa1), 0
            else:           # bottom-right
                xa1, ya1, xa2, ya2 = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                xb1, yb1 = 0, 0
            xb2, yb2 = xb1 + (xa2 - xa1), yb1 + (ya2 - ya1)
            if xa2 <= xa1 or ya2 <= ya1:
                continue
            canvas[ya1:ya2, xa1:xa2] = img[yb1:yb2, xb1:xb2]
            padw, padh = xa1 - xb1, ya1 - yb1

            if self.task == "masks" and len(masks):
                for mk in masks:
                    mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
                    big = np.zeros((s * 2, s * 2), np.uint8)
                    big[ya1:ya2, xa1:xa2] = mk[yb1:yb2, xb1:xb2]
                    all_masks.append(big)
            if self.task == "keypoints" and len(kps):
                k = kps.copy()
                k[:, :, 0] = k[:, :, 0] * scale + padw
                k[:, :, 1] = k[:, :, 1] * scale + padh
                out = ((k[:, :, 0] < 0) | (k[:, :, 0] >= s * 2) |
                       (k[:, :, 1] < 0) | (k[:, :, 1] >= s * 2))
                k[out, 2] = 0
                all_kps.append(k)
            all_labels.append(labels)

        labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), np.int64)
        masks = (np.stack(all_masks) if all_masks
                 else np.zeros((0, s * 2, s * 2), np.uint8))
        kps = (np.concatenate(all_kps) if all_kps
               else np.zeros((0, max(self.num_keypoints, 1), 3), np.float32))
        return canvas, labels, masks, kps

    # ---- the item ----

    def __getitem__(self, i):
        use_mosaic = self.augment and self.mosaic_p > 0 and self.rng.random() < self.mosaic_p
        if use_mosaic:
            img, labels, masks, kps = self._mosaic(i)
            image_id = self.samples[i][0]
        else:
            img, labels, masks, kps, image_id = self._load_raw(i)
            if self.augment:
                angle = self.rng.uniform(-self.rotation, self.rotation)
                m, ow, oh = rotation_matrix(img.shape[1], img.shape[0], angle, expand=True)
                img, masks_w, kps_w = warp(img, masks if len(masks) else None,
                                           kps if len(kps) else None, m, ow, oh)
                if masks_w is not None:
                    masks = masks_w
                if kps_w is not None:
                    kps = kps_w
                img = colour_jitter(img, rng=self.rng)

        h, w = img.shape[:2]
        if self.task == "masks":
            boxes = boxes_from_masks(masks)
            good = ((boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
                    if len(boxes) else np.zeros((0,), bool))
        else:
            boxes = boxes_from_keypoints(kps, w, h, self.kp_box_pad)
            visible = (kps[:, :, 2] > 0).sum(axis=1) if len(kps) else np.zeros((0,), int)
            good = ((boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
                    & (visible > 0) if len(boxes) else np.zeros((0,), bool))

        boxes, labels = boxes[good], labels[good]
        target = {
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes)).float().reshape(-1, 4),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)).long(),
            "image_id": torch.tensor([int(image_id)]),
        }
        if self.task == "masks":
            target["masks"] = torch.from_numpy(
                np.ascontiguousarray(masks[good])).to(torch.uint8).reshape(-1, h, w)
        else:
            target["keypoints"] = torch.from_numpy(
                np.ascontiguousarray(kps[good])).float().reshape(-1, max(self.num_keypoints, 1), 3)
        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return image, target


def collate_fn(batch):
    """Detection models take lists, not stacked tensors — images differ in size."""
    return tuple(zip(*batch))
