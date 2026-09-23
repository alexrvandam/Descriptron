#!/usr/bin/env python3
"""
texture_phenomics_from_coco.py
==============================
Texture feature extraction + multivariate analysis per sclerite category.

Reads a COCO JSON with segmentation masks, extracts a rich texture feature
vector from each masked region (per category, per specimen), then runs
PCA / UMAP / hierarchical clustering for morphospace visualisation.

Output CSVs are formatted for direct use in phylogenetic comparative
analyses (e.g. phytools::contMap in R, or dendropy in Python).

Feature vector (~25 features per specimen per category):
  GLCM (5 metrics × 3 distances)     = 15 features
  LBP  (entropy, uniformity, mean, std) = 4 features
  FFT granularity (max_power, max_freq, sum_power) = 3 features
  LAB stats (mean, std per channel)   = 6 features

Usage:
  python texture_phenomics_from_coco.py \\
    --json annotations.json --image_dir ./images \\
    --output_dir ./texture_out

Optional:
  --category_name pronotum   (process one category; blank = all)
  --tree phylogeny.nwk       (Newick tree for trait mapping)
  --umap_n_neighbors 15
"""

import sys
import os
import json
import argparse
import logging
import numpy as np
import cv2
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from scipy.ndimage import uniform_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logging.warning("scikit-image not available — GLCM and LBP disabled")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from pycocotools import mask as maskUtils
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("texture_phenomics.log"),
              logging.StreamHandler(sys.stdout)]
)


###############################################################################
# COCO HELPERS
###############################################################################
def load_coco(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    cats = {c['id']: c['name'] for c in coco.get('categories', [])}
    imgs = {i['id']: i for i in coco.get('images', [])}
    anns_by_img = {}
    for a in coco.get('annotations', []):
        anns_by_img.setdefault(a['image_id'], []).append(a)
    return cats, imgs, anns_by_img


def decode_mask(ann, h, w):
    seg = ann.get('segmentation')
    if not seg:
        return np.zeros((h, w), np.uint8)
    try:
        if isinstance(seg, list):
            rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
        else:
            rle = seg
        m = maskUtils.decode(rle)
    except Exception:
        return np.zeros((h, w), np.uint8)
    return (m > 0).astype(np.uint8)


###############################################################################
# TEXTURE FEATURE EXTRACTION
###############################################################################

def glcm_features(gray, mask, distances=(1, 3, 5), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Extract GLCM texture features from masked region.
    Returns dict of contrast, dissimilarity, homogeneity, energy, correlation
    averaged over angles, per distance."""
    if not HAS_SKIMAGE:
        return {}

    # crop to bounding box for efficiency
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {}
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = gray[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    # set background to 0 to avoid spurious co-occurrences
    crop[crop_mask == 0] = 0

    # quantise to 64 levels for stable GLCM
    crop_q = (crop // 4).astype(np.uint8)

    glcm = graycomatrix(crop_q, distances=list(distances),
                        angles=list(angles), levels=64, symmetric=True, normed=True)

    features = {}
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in props:
        vals = graycoprops(glcm, prop)  # shape (n_distances, n_angles)
        for di, dist in enumerate(distances):
            # average over angles
            features[f'glcm_{prop}_d{dist}'] = float(vals[di, :].mean())

    return features


def lbp_features(gray, mask, radius=2, n_points=16):
    """Extract Local Binary Pattern features from masked region."""
    if not HAS_SKIMAGE:
        return {}

    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    fg_lbp = lbp[mask > 0]
    if len(fg_lbp) == 0:
        return {}

    n_bins = n_points + 2  # uniform LBP has P+2 bins
    hist, _ = np.histogram(fg_lbp, bins=n_bins, range=(0, n_bins), density=True)

    # entropy of LBP histogram
    hist_nz = hist[hist > 0]
    entropy = float(-np.sum(hist_nz * np.log2(hist_nz)))

    # uniformity = proportion of uniform patterns (all bins except last)
    uniformity = float(hist[:-1].sum())

    return {
        'lbp_entropy': entropy,
        'lbp_uniformity': uniformity,
        'lbp_mean': float(fg_lbp.mean()),
        'lbp_std': float(fg_lbp.std()),
    }


def granularity_features(gray, mask, n_bands=20, max_filter_px=80):
    """FFT band-pass granularity analysis (MICA-style).
    Returns max_power, max_freq, sum_power."""
    fg = mask > 0
    if fg.sum() == 0:
        return {}

    lum = gray.astype(np.float64)
    fg_mean = lum[fg].mean()
    lum_masked = lum.copy()
    lum_masked[~fg] = fg_mean

    filter_sizes = np.linspace(1, max_filter_px, n_bands + 1).astype(int)
    filter_sizes = np.unique(filter_sizes)

    energies = []
    prev_smooth = lum_masked.copy()
    for fs in filter_sizes:
        ksize = max(1, int(fs))
        smoothed = uniform_filter(lum_masked, size=ksize, mode='reflect')
        band = prev_smooth - smoothed
        energy = float(band[fg].std())
        energies.append(energy)
        prev_smooth = smoothed

    energies = np.array(energies)
    max_idx = int(np.argmax(energies))

    return {
        'gran_max_power': float(energies[max_idx]),
        'gran_max_freq': float(filter_sizes[max_idx]),
        'gran_sum_power': float(energies.sum()),
    }


def color_stats(img_bgr, mask):
    """Extract LAB color statistics from masked region."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    fg = mask > 0
    if fg.sum() == 0:
        return {}

    features = {}
    for ci, name in enumerate(['L', 'a', 'b']):
        vals = lab[:, :, ci][fg]
        features[f'{name}_mean'] = float(vals.mean())
        features[f'{name}_std'] = float(vals.std())

    return features


def extract_texture_vector(img_bgr, mask):
    """Extract full texture feature vector from one masked region.
    Returns OrderedDict of ~25 features."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    features = OrderedDict()

    # GLCM
    features.update(glcm_features(gray, mask))

    # LBP
    features.update(lbp_features(gray, mask))

    # Granularity
    features.update(granularity_features(gray, mask))

    # Color stats
    features.update(color_stats(img_bgr, mask))

    return features


###############################################################################
# ANALYSIS PIPELINE
###############################################################################

def mk_thumb(img, mask, sz=(80, 80)):
    """Create thumbnail with transparent background."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bm = (mask > 0).astype(np.uint8)
    fg = np.zeros_like(img)
    fg[bm == 1] = img[bm == 1]
    rgb = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    pi = Image.fromarray(rgb).convert("RGBA")
    pi.putalpha(Image.fromarray((bm * 255).astype(np.uint8)).convert("L"))
    pi.thumbnail(sz, Image.LANCZOS)
    ni = Image.new('RGBA', sz, (255, 255, 255, 0))
    ni.paste(pi, ((sz[0] - pi.size[0]) // 2, (sz[1] - pi.size[1]) // 2))
    return np.array(ni)


def plot_morphospace(scores, imgs, masks, fns, od, cat, method='PCA',
                     labels=None):
    """PCA/UMAP morphospace with specimen thumbnails."""
    if scores.shape[1] < 2:
        scores = np.hstack([scores, np.zeros((len(scores), 1))])
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(f"Texture {method} — {cat}")
    ax.set_xlabel(f"{method}1")
    ax.set_ylabel(f"{method}2")
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0)
    for i, (p, im, mk) in enumerate(zip(scores, imgs, masks)):
        t = mk_thumb(im, mk)
        if t is not None:
            ax.add_artist(AnnotationBbox(
                OffsetImage(t, zoom=0.6), (p[0], p[1]), frameon=False))
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"texture_{method.lower()}_{cat}.png"), dpi=300)
    plt.close()


def plot_feature_heatmap(df, od, cat):
    """Heatmap of standardised texture features across specimens."""
    import seaborn as sns
    feat_cols = [c for c in df.columns if c != 'filename']
    if not feat_cols:
        return
    vals = df[feat_cols].values
    sc = StandardScaler()
    vals_z = sc.fit_transform(vals)

    fig, ax = plt.subplots(figsize=(max(12, len(feat_cols) * 0.5),
                                    max(6, len(df) * 0.15)))
    sns.heatmap(vals_z, xticklabels=feat_cols,
                yticklabels=df['filename'].values,
                cmap='RdBu_r', center=0, ax=ax)
    ax.set_title(f"Texture features (z-scored) — {cat}")
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"texture_heatmap_{cat}.png"), dpi=200)
    plt.close()


def do_clustering(X, od, cat, scores_2d, imgs, masks, fns, nc=3):
    """Hierarchical clustering + dendrogram."""
    import seaborn as sns
    Z = linkage(X, 'ward', 'euclidean')
    labels = fcluster(Z, nc, 'maxclust')

    # dendrogram
    fig, ax = plt.subplots(figsize=(16, 6))
    dendrogram(Z, labels=fns, leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title(f"Texture dendrogram — {cat}")
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"texture_dendrogram_{cat}.png"), dpi=200)
    plt.close()

    # scatter with cluster colors
    fig, ax = plt.subplots(figsize=(12, 8))
    for lab, col in zip(np.unique(labels), sns.color_palette(None, len(np.unique(labels)))):
        m = labels == lab
        ax.scatter(scores_2d[m, 0], scores_2d[m, 1], c=[col],
                   label=f'Cluster {lab}', alpha=0.7, s=60)
    for i, (p, im, mk) in enumerate(zip(scores_2d, imgs, masks)):
        t = mk_thumb(im, mk, sz=(50, 50))
        if t is not None:
            ax.add_artist(AnnotationBbox(
                OffsetImage(t, zoom=0.5), (p[0], p[1]), frameon=False))
    ax.set_title(f"Texture clusters — {cat}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(od, f"texture_clusters_{cat}.png"), dpi=300)
    plt.close()

    return labels


###############################################################################
# PHYLOGENETIC TRAIT MAPPING (optional)
###############################################################################

def map_traits_to_tree(tree_path, trait_df, trait_cols, od, cat):
    """Basic trait-mapped phylogeny using Bio.Phylo or toytree."""
    try:
        from Bio import Phylo
        import matplotlib.colors as mcolors

        tree = Phylo.read(tree_path, 'newick')

        # try to match tip labels to filenames (species names)
        tip_names = [t.name for t in tree.get_terminals() if t.name]
        matched = trait_df[trait_df['filename'].str.contains(
            '|'.join(tip_names), case=False, na=False)]

        if len(matched) < 3:
            logging.warning(f"Only {len(matched)} tips matched — "
                            f"tree mapping skipped for {cat}")
            return

        # map first trait column (e.g. PC1) as color
        trait_col = trait_cols[0] if trait_cols else 'PC1'
        if trait_col not in matched.columns:
            logging.warning(f"Trait column '{trait_col}' not found")
            return

        logging.info(f"Tree trait mapping: {len(matched)} tips matched, "
                     f"trait='{trait_col}'")

        fig, ax = plt.subplots(figsize=(10, max(8, len(tip_names) * 0.2)))
        Phylo.draw(tree, axes=ax, do_show=False)
        ax.set_title(f"Phylogeny — {cat} — {trait_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(od, f"texture_phylo_{cat}.png"), dpi=200)
        plt.close()

        # export trait table matched to tips for R/phytools
        matched.to_csv(os.path.join(od, f"texture_traits_for_phylo_{cat}.csv"),
                       index=False)
        logging.info(f"Trait table for phylo saved — use with phytools::contMap() in R")

    except ImportError:
        logging.info("Bio.Phylo not available — exporting CSV for R/phytools instead")
        trait_df.to_csv(os.path.join(od, f"texture_traits_for_phylo_{cat}.csv"),
                        index=False)
    except Exception as e:
        logging.error(f"Tree mapping error: {e}")
        trait_df.to_csv(os.path.join(od, f"texture_traits_for_phylo_{cat}.csv"),
                        index=False)


###############################################################################
# MAIN
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description='Texture phenomics: GLCM + LBP + granularity + color '
                    'per category from COCO JSON')
    p.add_argument('--json', required=True)
    p.add_argument('--image_dir', required=True)
    p.add_argument('--output_dir', default='./texture_out')
    p.add_argument('--category_name', default='',
                   help='Category to process (blank = all)')
    p.add_argument('--tree', default='',
                   help='Newick tree file for phylogenetic trait mapping')

    # granularity
    p.add_argument('--granularity_bands', type=int, default=20)
    p.add_argument('--granularity_max_px', type=int, default=80)

    # UMAP
    p.add_argument('--umap_n_neighbors', type=int, default=15)
    p.add_argument('--umap_min_dist', type=float, default=0.1)

    # clustering
    p.add_argument('--n_clusters', type=int, default=3)

    # group labels
    p.add_argument('--group_labels', default='',
                   help='CSV/TSV with filename,group_label columns')

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cats, imgs_info, anns_by_img = load_coco(args.json)

    # filter categories
    kp_names = {'keypoints', 'line_keypoints'}
    if args.category_name.strip():
        seg_cats = [(cid, n) for cid, n in cats.items()
                    if n.lower() == args.category_name.lower()]
    else:
        seg_cats = [(cid, n) for cid, n in cats.items()
                    if n.lower() not in kp_names]
    if not seg_cats:
        logging.error("No categories found")
        return

    # group labels
    gl_map = {}
    if args.group_labels and os.path.isfile(args.group_labels):
        try:
            sep = '\t' if args.group_labels.endswith('.tsv') else ','
            gl_df = pd.read_csv(args.group_labels, sep=sep)
            gl_df.columns = [c.strip().lower() for c in gl_df.columns]
            if 'filename' in gl_df.columns and 'group_label' in gl_df.columns:
                gl_map = dict(zip(gl_df['filename'].astype(str),
                                  gl_df['group_label'].astype(str)))
        except Exception as e:
            logging.warning(f"Could not load group labels: {e}")

    for cat_id, cat_name in seg_cats:
        logging.info(f"=== TEXTURE: {cat_name} ===")
        od = os.path.join(args.output_dir, cat_name.replace(' ', '_'))
        os.makedirs(od, exist_ok=True)

        records = []
        all_imgs = []
        all_masks = []
        all_fns = []

        for iid, anns in anns_by_img.items():
            ii = imgs_info.get(iid)
            if not ii:
                continue
            fn = ii['file_name']
            bn = os.path.basename(fn)
            ip = os.path.join(args.image_dir, fn)
            img = cv2.imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]

            # find best annotation for this category
            cat_anns = [a for a in anns if a['category_id'] == cat_id]
            if not cat_anns:
                continue
            best = max(cat_anns, key=lambda a: a.get('area', 0))
            mask = decode_mask(best, h, w)
            if mask.sum() < 100:  # skip tiny masks
                continue

            # extract features
            feats = extract_texture_vector(img, mask)
            if not feats:
                continue

            rec = OrderedDict()
            rec['filename'] = bn
            rec['image_id'] = iid
            if bn in gl_map:
                rec['group_label'] = gl_map[bn]
            rec.update(feats)
            records.append(rec)

            all_imgs.append(img)
            all_masks.append(mask)
            all_fns.append(bn)

        if not records:
            logging.warning(f"No specimens for '{cat_name}'")
            continue

        logging.info(f"Extracted texture features for {len(records)} specimens")

        # save raw features
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(od, f"texture_features_{cat_name}.csv"),
                  index=False)

        # --- multivariate analysis ---
        feat_cols = [c for c in df.columns
                     if c not in ('filename', 'image_id', 'group_label')]
        X = df[feat_cols].values.astype(float)

        # handle NaN
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape[1]):
                X[nan_mask[:, j], j] = col_means[j]

        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)

        # PCA
        n_comp = min(X_scaled.shape[0], X_scaled.shape[1], 10)
        pca = PCA(n_components=n_comp)
        pc = pca.fit_transform(X_scaled)

        logging.info(f"PCA: {pca.explained_variance_ratio_[:3].round(3)} "
                     f"(first 3 components)")

        plot_morphospace(pc[:, :2], all_imgs, all_masks, all_fns,
                         od, cat_name, 'PCA')

        # add PC scores to dataframe
        for i in range(min(5, pc.shape[1])):
            df[f'PC{i+1}'] = pc[:, i]

        # UMAP
        if HAS_UMAP and len(records) > 5:
            nn = min(args.umap_n_neighbors, len(records) - 1)
            if nn > 1:
                um = umap.UMAP(n_components=2, random_state=42,
                               n_neighbors=nn, min_dist=args.umap_min_dist)
                umap_scores = um.fit_transform(X_scaled)
                plot_morphospace(umap_scores, all_imgs, all_masks, all_fns,
                                 od, cat_name, 'UMAP')
                df['UMAP1'] = umap_scores[:, 0]
                df['UMAP2'] = umap_scores[:, 1]

        # clustering
        labels = do_clustering(X_scaled, od, cat_name, pc[:, :2],
                               all_imgs, all_masks, all_fns,
                               nc=args.n_clusters)
        df['cluster'] = labels

        # feature heatmap (limit to first 50 specimens for readability)
        if len(df) <= 50:
            hm_df = df[['filename'] + feat_cols]
        else:
            hm_df = df[['filename'] + feat_cols].iloc[:50]
        plot_feature_heatmap(hm_df, od, cat_name)

        # save full results
        df.to_csv(os.path.join(od, f"texture_analysis_{cat_name}.csv"),
                  index=False)

        # variance explained
        var_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'variance_explained': pca.explained_variance_ratio_,
            'cumulative': np.cumsum(pca.explained_variance_ratio_)
        })
        var_df.to_csv(os.path.join(od, f"texture_pca_variance_{cat_name}.csv"),
                      index=False)

        # feature loadings
        load_df = pd.DataFrame(
            pca.components_[:min(5, n_comp)].T,
            columns=[f'PC{i+1}' for i in range(min(5, n_comp))],
            index=feat_cols
        )
        load_df.to_csv(os.path.join(od, f"texture_pca_loadings_{cat_name}.csv"))

        # phylogenetic trait mapping
        if args.tree and os.path.isfile(args.tree):
            trait_cols_for_phylo = [f'PC{i+1}' for i in range(min(3, pc.shape[1]))]
            map_traits_to_tree(args.tree, df, trait_cols_for_phylo, od, cat_name)
        else:
            # always save a phylo-ready CSV even without a tree
            phylo_cols = ['filename'] + feat_cols + [f'PC{i+1}' for i in range(min(5, pc.shape[1]))]
            phylo_cols = [c for c in phylo_cols if c in df.columns]
            df[phylo_cols].to_csv(
                os.path.join(od, f"texture_traits_for_phylo_{cat_name}.csv"),
                index=False)

        logging.info(f"Done: {cat_name} — {len(records)} specimens, "
                     f"{len(feat_cols)} features")

    logging.info("All categories complete.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
