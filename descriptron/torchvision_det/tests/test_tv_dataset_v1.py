#!/usr/bin/env python3
"""
Unit tests for tv_coco_dataset — the geometry, the mosaic and the box derivation.

These matter more than usual because the module transforms masks, boxes and
keypoints with one matrix by hand: if they ever fall out of step the model still
trains, silently, on wrong targets. Every test here builds its own tiny COCO in a
temp directory, so nothing depends on the real data.

    python tests/test_tv_dataset_v1.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tv_coco_dataset as D


def tiny_coco(tmp, n_images=4, task="masks", size=(120, 160)):
    """A square blob at a known place in each image, or four keypoints on it."""
    h, w = size
    img_dir = Path(tmp) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for i in range(n_images):
        arr = np.zeros((h, w, 3), np.uint8)
        x0, y0, x1, y1 = 40, 30, 90, 70
        arr[y0:y1, x0:x1] = 200
        Image.fromarray(arr).save(img_dir / f"im{i}.png")
        images.append({"id": i, "file_name": f"im{i}.png", "width": w, "height": h})
        a = {"id": i + 1, "image_id": i, "category_id": 1, "iscrowd": 0,
             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0)}
        if task == "masks":
            a["segmentation"] = [[x0, y0, x1, y0, x1, y1, x0, y1]]
        else:
            a["keypoints"] = [x0, y0, 2, x1, y0, 2, x1, y1, 2, x0, y1, 2]
            a["num_keypoints"] = 4
        anns.append(a)
    cat = {"id": 1, "name": "blob"}
    if task == "keypoints":
        cat["keypoints"] = ["tl", "tr", "br", "bl"]
    coco = {"images": images, "annotations": anns, "categories": [cat]}
    p = Path(tmp) / "coco.json"
    p.write_text(json.dumps(coco))
    return str(p), str(img_dir)


class TestGeometry(unittest.TestCase):
    def test_zero_rotation_is_identity(self):
        m, w, h = D.rotation_matrix(160, 120, 0.0)
        self.assertEqual((w, h), (160, 120))
        pts = np.array([[10.0, 20.0], [100.0, 90.0]])
        np.testing.assert_allclose(D.apply_affine_points(pts, m), pts, atol=1e-4)

    def test_expand_grows_canvas_like_detectron2(self):
        _, w, h = D.rotation_matrix(100, 100, 45.0, expand=True)
        self.assertGreater(w, 100)          # a rotated square needs a bigger canvas
        self.assertAlmostEqual(w, h)

    def test_mask_and_keypoints_stay_together_through_rotation(self):
        """The property that matters: a rotated keypoint must land on its mask."""
        h, w = 120, 160
        mask = np.zeros((h, w), np.uint8)
        mask[30:70, 40:90] = 1
        corners = np.array([[[40.0, 30.0, 2.0], [89.0, 30.0, 2.0],
                             [89.0, 69.0, 2.0], [40.0, 69.0, 2.0]]], np.float32)
        img = np.zeros((h, w, 3), np.uint8)
        for angle in (-37.0, 12.0, 45.0):
            m, ow, oh = D.rotation_matrix(w, h, angle, expand=True)
            _, masks, kps = D.warp(img, mask[None], corners, m, ow, oh)
            inside = 0
            for (x, y, v) in kps[0]:
                if v == 0:
                    continue
                xi, yi = int(round(x)), int(round(y))
                # allow a one-pixel slack: the corner sits on the mask boundary
                win = masks[0][max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]
                inside += int(win.any())
            self.assertEqual(inside, 4, f"keypoints left the mask at {angle} deg")

    def test_colour_jitter_keeps_shape_and_range(self):
        img = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
        out = D.colour_jitter(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)


class TestBoxes(unittest.TestCase):
    def test_boxes_from_masks_are_tight(self):
        m = np.zeros((50, 60), np.uint8)
        m[10:20, 5:35] = 1
        np.testing.assert_allclose(D.boxes_from_masks([m])[0], [5, 10, 35, 20])

    def test_boxes_from_keypoints_pad_and_clip(self):
        kp = np.array([[[10.0, 10.0, 2.0], [30.0, 40.0, 2.0]]], np.float32)
        box = D.boxes_from_keypoints(kp, 100, 100, pad_frac=0.1, min_pad=5.0)[0]
        self.assertLess(box[0], 10.0)                 # padded outwards
        self.assertGreater(box[2], 30.0)
        self.assertGreaterEqual(box[0], 0.0)          # clipped to the image
        self.assertLessEqual(box[3], 100.0)

    def test_invisible_keypoints_are_ignored_for_the_box(self):
        kp = np.array([[[10.0, 10.0, 2.0], [90.0, 90.0, 0.0]]], np.float32)
        box = D.boxes_from_keypoints(kp, 100, 100)[0]
        self.assertLess(box[2], 50.0)                 # the invisible point is not in it


class TestDataset(unittest.TestCase):
    def test_masks_target_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 3, "masks")
            ds = D.CocoInstanceDataset(j, d, task="masks", augment=False)
            self.assertEqual(len(ds), 3)
            self.assertEqual(ds.num_classes, 2)       # blob + background
            img, t = ds[0]
            self.assertEqual(img.shape[0], 3)
            self.assertEqual(t["masks"].shape[0], t["boxes"].shape[0])
            self.assertEqual(t["labels"].tolist(), [1])
            np.testing.assert_allclose(t["boxes"][0].numpy(), [40, 30, 90, 70], atol=1.0)

    def test_keypoints_target_and_derived_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 3, "keypoints")
            ds = D.CocoInstanceDataset(j, d, task="keypoints", augment=False)
            self.assertEqual(ds.num_keypoints, 4)
            _, t = ds[0]
            self.assertEqual(tuple(t["keypoints"].shape[1:]), (4, 3))
            self.assertEqual(t["boxes"].shape[0], 1)  # a box was derived, none existed
            self.assertGreater(float(t["boxes"][0, 2] - t["boxes"][0, 0]), 0.0)

    def test_augmentation_keeps_targets_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 3, "masks")
            ds = D.CocoInstanceDataset(j, d, task="masks", augment=True, seed=3)
            for i in range(3):
                img, t = ds[i]
                self.assertEqual(t["masks"].shape[-2:], img.shape[-2:],
                                 "mask canvas must follow the rotated image")
                b = t["boxes"][0].numpy()
                self.assertTrue(b[2] > b[0] and b[3] > b[1])

    def test_mosaic_composes_four_and_keeps_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 6, "masks")
            ds = D.CocoInstanceDataset(j, d, task="masks", augment=True,
                                       mosaic_p=1.0, mosaic_size=128, seed=1)
            img, t = ds[0]
            self.assertEqual(img.shape[-2:], (256, 256))         # 2 x mosaic_size
            self.assertGreaterEqual(t["boxes"].shape[0], 2,
                                    "a mosaic should carry instances from several images")
            self.assertEqual(t["masks"].shape[-2:], (256, 256))
            self.assertEqual(t["masks"].shape[0], t["boxes"].shape[0])

    def test_mosaic_works_for_keypoints_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 6, "keypoints")
            ds = D.CocoInstanceDataset(j, d, task="keypoints", augment=True,
                                       mosaic_p=1.0, mosaic_size=128, seed=2)
            _, t = ds[0]
            self.assertGreaterEqual(t["keypoints"].shape[0], 2)
            self.assertEqual(t["keypoints"].shape[0], t["boxes"].shape[0])

    def test_image_id_subset_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 5, "masks")
            ds = D.CocoInstanceDataset(j, d, task="masks", image_ids=[1, 3])
            self.assertEqual(len(ds), 2)
            self.assertEqual([s[0] for s in ds.samples], [1, 3])

    def test_empty_segmentation_images_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, d = tiny_coco(tmp, 3, "masks")
            coco = json.load(open(j))
            coco["annotations"][0]["segmentation"] = []       # unusable
            Path(j).write_text(json.dumps(coco))
            ds = D.CocoInstanceDataset(j, d, task="masks")
            self.assertEqual(len(ds), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
