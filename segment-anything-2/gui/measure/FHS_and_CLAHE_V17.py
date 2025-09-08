#!/usr/bin/env python3
"""
FHS_and_CLAHE_V14.py   (2025‑06‑24)
----------------------------------
Felzenszwalb → CLAHE → second‑iteration LAB averaging workflow
with optional shine removal, colour‑true dendrograms, ΔE sweep,
and background‑masking via the original binary mask.
"""
import os
import argparse
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
from skimage import io, color
from skimage.segmentation import felzenszwalb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa

###############################################################################
# helper – LAB → RGB (float 0‑1)
###############################################################################

def _lab2rgb_pts(lab_arr: np.ndarray) -> np.ndarray:
    rgb = color.lab2rgb(lab_arr.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.clip(rgb, 0, 1)

###############################################################################
# 1) Felzenszwalb segmentation (first pass)
###############################################################################

def fhs_segmentation(img_p: str, mask_p: str, out_p: str) -> None:
    rgb = io.imread(img_p)
    mask = io.imread(mask_p)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if rgb.shape[:2] != mask.shape:
        print("[WARN] shape mismatch – skipping", img_p)
        return
    fg = np.zeros_like(rgb)
    fg[mask > 0] = rgb[mask > 0]
    seg = felzenszwalb(fg, scale=100, sigma=0.5, min_size=50)
    io.imsave(out_p, color.label2rgb(seg, fg, kind="avg", bg_label=0))

###############################################################################
# 2) CLAHE luminance normalisation
###############################################################################

def normalize_color(src: str, dst: str) -> None:
    img = cv2.imread(src); assert img is not None, src
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
    cv2.imwrite(dst, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

###############################################################################
# 0) shine / dust removal
###############################################################################

def remove_shine_inpaint(src: str, dst: str, thresh_v: int = 240) -> None:
    img = cv2.imread(src)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 2], thresh_v, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                            iterations=2)
    cv2.imwrite(dst, cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA))

###############################################################################
# 3) LAB‑space averaging + coloured dendrogram / scatters
###############################################################################

def lab_average_clustering(seg_p: str,
                           save_p: str,
                           threshold: float = 25.0,
                           mask_path: Optional[str] = None,
                           dendro_path: Optional[str] = None,
                           scatter_before: Optional[str] = None,
                           scatter_after: Optional[str] = None) -> None:
    rgb = io.imread(seg_p)

    # mask background ---------------------------------------------
    if mask_path is not None and os.path.exists(mask_path):
        mask = io.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        fg = mask > 0
    else:  # fallback: drop strictly black pixels
        fg = ~((rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0))

    lab_fg = color.rgb2lab(rgb[fg]).reshape(-1, 3)

    uniq, inverse = np.unique(lab_fg, axis=0, return_inverse=True)
    Z = linkage(uniq, method="average", metric="euclidean")
    clusters = fcluster(Z, t=threshold, criterion="distance")
    means = {c: uniq[clusters == c].mean(axis=0) for c in np.unique(clusters)}

    lab_out = np.vstack([means[c] for c in clusters])[inverse]
    rgb_out = np.zeros_like(rgb, dtype=np.float64)
    rgb_out[fg] = color.lab2rgb(lab_out)
    io.imsave(save_p, (rgb_out * 255).astype(np.uint8))

    # ---------- extra outputs: grayscale & binary ----------
    gray_full = cv2.cvtColor((rgb_out * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)

    gray_canvas = np.zeros_like(gray_full)
    gray_canvas[fg] = gray_full[fg]
    io.imsave(save_p.replace(".png", "_gray.png"), gray_canvas)

    # map cluster label back to each foreground pixel
    pixel_labels = clusters[inverse]               # length == #fg pixels
    label_map = np.zeros_like(gray_full, dtype=np.int32)
    label_map[fg] = pixel_labels

    dominant = max(means, key=lambda c: (pixel_labels == c).sum())
    binary = np.zeros_like(gray_full, dtype=np.uint8)
    binary[label_map == dominant] = 255
    io.imsave(save_p.replace(".png", "_binary.png"), binary)

    # true‑colour dendrogram --------------------------------------
    if dendro_path:
        n_leaves = uniq.shape[0]
        node_lab: Dict[int, np.ndarray] = {i: uniq[i] for i in range(n_leaves)}
        for i, (l, r, *_rest) in enumerate(Z):
            node_lab[n_leaves + i] = (node_lab[int(l)] + node_lab[int(r)]) / 2

        def _link_color(nid: int):
            return mcolors.to_hex(_lab2rgb_pts(node_lab[nid][None, :])[0])

        fig, ax = plt.subplots(figsize=(10, 5))
        dn = dendrogram(Z, ax=ax, link_color_func=_link_color, no_labels=True)
        xs = ax.get_xticks(); leaves = dn["leaves"]
        for x, leaf_idx in zip(xs, leaves):
            c = _lab2rgb_pts(uniq[leaf_idx][None, :])[0]
            ax.scatter(x, -2, s=120, c=[c], marker="s", clip_on=False)
        ax.set_title(f"Colour dendrogram  (ΔE ≤ {threshold})")
        plt.tight_layout(); plt.savefig(dendro_path, dpi=180); plt.close()

    # 3‑D scatters -------------------------------------------------
    def _scatter(pts: np.ndarray, fn: Optional[str]):
        if fn:
            fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pts[:, 1], pts[:, 2], pts[:, 0], s=40,
                       c=_lab2rgb_pts(pts))
            ax.set_xlabel("a"); ax.set_ylabel("b"); ax.set_zlabel("L")
            plt.tight_layout(); plt.savefig(fn, dpi=180); plt.close()

    _scatter(uniq, scatter_before)
    _scatter(np.array(list(means.values())), scatter_after)

###############################################################################
# 4) CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Felzenszwalb + CLAHE + LAB averaging")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_segmentation", action="store_true")
    p.add_argument("--run_normalization", action="store_true")
    p.add_argument("--remove_shine", action="store_true")
    p.add_argument("--run_lab_average", action="store_true")
    p.add_argument("--lab_threshold", type=float, default=25.0)
    p.add_argument("--lab_sweep", type=str,
                   help="Comma‑sep list of ΔE thresholds, e.g. 15,20,25")
    p.add_argument("--visualize", action="store_true")
    return p.parse_args()

###############################################################################
# 5) main
###############################################################################

def main():
    args = parse_args(); os.makedirs(args.output_dir, exist_ok=True)
    thresholds = ([float(t) for t in args.lab_sweep.split(",")]
                  if args.lab_sweep else [args.lab_threshold])

    all_seg_pairs: List[Tuple[str, str]] = []  # (seg_path, matching_mask)
    viz_done = False

    # build bin/fg pairs --------------------------------------------------
    files = os.listdir(args.input_dir)
    bin_files = [f for f in files if "binary_mask_" in f]
    fg_files  = [f for f in files if "foreground_mask_" in f]
    pairs: Dict[str, Dict[str, str]] = {}
    for f in bin_files:
        key = f.replace("binary_mask_", "", 1)
        pairs.setdefault(key, {})["bin"] = f
    for f in fg_files:
        key = f.replace("foreground_mask_", "", 1)
        pairs.setdefault(key, {})["fg"] = f

    # iterate specimens ----------------------------------------------------
    for base, pair in pairs.items():
        if "bin" not in pair or "fg" not in pair:
            continue
        bin_p = os.path.join(args.input_dir, pair["bin"])
        fg_p  = os.path.join(args.input_dir, pair["fg"])

        fg_work = fg_p
        if args.remove_shine:
            shine_p = os.path.join(args.output_dir, f"shinefree_{pair['fg']}")
            remove_shine_inpaint(fg_p, shine_p)
            fg_work = shine_p

        if args.run_segmentation:
            seg_p = os.path.join(args.output_dir, f"fhs_{pair['fg']}")
            fhs_segmentation(fg_work, bin_p, seg_p)
            all_seg_pairs.append((seg_p, bin_p))

        if args.run_normalization:
            norm_p = os.path.join(args.output_dir, f"normalized_{pair['fg']}")
            normalize_color(fg_work, norm_p)
            if args.run_segmentation:
                seg_norm_p = os.path.join(args.output_dir,
                                          f"fhs_normalized_{pair['fg']}")
                fhs_segmentation(norm_p, bin_p, seg_norm_p)
                all_seg_pairs.append((seg_norm_p, bin_p))

    # second iteration LAB averaging --------------------------------------
    if args.run_lab_average:
        for seg_p, mask_p in all_seg_pairs:
            base = os.path.basename(seg_p)
            for t in thresholds:
                out_p = os.path.join(args.output_dir, f"labavg_T{t}_{base}")
                dendro = scatter_b = scatter_a = None
                if args.visualize and not viz_done:
                    dendro  = os.path.join(args.output_dir, 'dendrogram.png')
                    scatter_b = os.path.join(args.output_dir, 'lab_before.png')
                    scatter_a = os.path.join(args.output_dir, 'lab_after.png')
                lab_average_clustering(seg_p, out_p,
                                       threshold=t,
                                       mask_path=mask_p,
                                       dendro_path=dendro,
                                       scatter_before=scatter_b,
                                       scatter_after=scatter_a)
                if dendro:  # produced global viz once
                    viz_done = True

    print("Processing complete. Results saved in:", args.output_dir)

if __name__ == "__main__":
    main()
