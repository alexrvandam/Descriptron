import sys
import argparse
import json
import cv2
import numpy as np
import os
import traceback
from pycocotools import mask as maskUtils
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from scipy.cluster.hierarchy import dendrogram, linkage
import easyocr
import umap
import seaborn as sns
import re
from collections import defaultdict
from math import sqrt
import shutil
import logging

# DBSCAN
from sklearn.cluster import DBSCAN
#from pacmap import PaCMAP

###############################
# Configure Logging
###############################
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

###############################
# Argument Parsing
###############################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Perform semi-landmark Procrustes alignment using the previous (shift-based) method, plus separate keypoint alignment.'
    )
    parser.add_argument('--json', required=True, help='Path to the COCO JSON file')
    parser.add_argument('--image_dir', required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', default='./outputs', help='Directory to save outputs')
    parser.add_argument('--image_id', type=int, required=False, help='ID of the image to process (optional)')
    # Modified: now optional. If left blank, all segmentation categories are processed.
    parser.add_argument('--category_name', default="", 
                        help='Name of the segmentation category (e.g. entire_forewing) to process. If left blank, all categories are processed.')

    # Semi-landmarking
    parser.add_argument('--num_landmarks', type=int, default=100,
                        help='Total number of semi-landmarks (if auto_landmarks is not used)')
    parser.add_argument('--auto_landmarks', action='store_true',
                        help='Auto compute # of semi-landmarks based on relative bending energy')
    parser.add_argument('--min_auto_landmarks', type=int, default=50,
                        help='Minimum # of landmarks in auto mode')
    parser.add_argument('--auto_scale_factor', type=float, required=False,
                        help='Scale factor for auto-landmarks')

    # Keypoints
    parser.add_argument('--keypoints_indices', type=str, required=False,
                        help='Comma/dash list of keypoint indices (1-based) to keep')

    # If we only want images that have BOTH the segmentation and keypoints
    parser.add_argument('--require_keypoints_and_segmentation', action='store_true',
                        help='Only process images that have BOTH the specified segmentation category AND keypoints')

    # Group labels & MANOVA
    parser.add_argument('--group_labels', required=False,
                        help='Path to CSV/TSV file mapping image filenames to group labels (headers: filename,group_label)')
    parser.add_argument('--perform_manova', action='store_true',
                        help='Perform MANOVA analysis on PCA scores (requires group labels)')

    # UMAP
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                        help='n_neighbors for UMAP')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                        help='min_dist for UMAP')

    # Optionally allow reflection or not
    parser.add_argument('--alignment_method', choices=['without_reflection','with_reflection'],
                        default='without_reflection',
                        help='Whether to allow reflection in the shift-based alignment (default no reflection).')
    return parser.parse_args()

###############################
# load_annotations
###############################
def load_annotations(json_path, image_id=None):
    """
    Reads the entire COCO JSON, returns:
       all_anns_per_image: dict {image_id -> [annotations]}
       images_info: dict {image_id -> image_dict}
       categories: dict {cat_id -> cat_name}
    """
    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

    categories = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    images = {img['id']: img for img in coco.get('images', [])}

    if image_id is not None:
        image_ids = [image_id]
    else:
        image_ids = list(images.keys())

    all_anns_per_image = {}
    for img_id in image_ids:
        img = images.get(img_id)
        if not img:
            continue
        image_filename = img.get('file_name','')
        if '/' in image_filename or '\\' in image_filename:
            # skip subdirectories
            continue
        ann_list = [ann for ann in coco.get('annotations',[]) if ann['image_id']==img_id]
        if ann_list:
            all_anns_per_image[img_id] = ann_list

    return all_anns_per_image, images, categories

###############################
# Utilities
###############################
def clean_filename(s):
    s = s.replace(' ', '_')
    return "".join(c for c in s if c.isalnum() or c in ('_', '-')).rstrip()

def remove_duplicate_points(contour, tolerance=1e-3):
    if len(contour)==0:
        return contour
    unique = [contour[0]]
    for pt in contour[1:]:
        if np.linalg.norm(pt - unique[-1]) > tolerance:
            unique.append(pt)
    return np.array(unique)

# Modified: Added try/except to catch errors when processing segmentation
def create_mask_from_annotation(annotation, height, width):
    if 'segmentation' in annotation:
        seg = annotation['segmentation']
        if isinstance(seg, list):
            try:
                rles = maskUtils.frPyObjects(seg, height, width)
                rle = maskUtils.merge(rles)
            except Exception as e:
                logging.warning(f"Error processing segmentation for annotation: {e}")
                return np.zeros((height, width), dtype=np.uint8)
        else:
            rle = seg
        try:
            mask = maskUtils.decode(rle)
        except Exception as e:
            print(f"Error decoding mask: {e}")
            mask = np.zeros((height, width), dtype=np.uint8)
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
    binary = (mask>0).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary

def extract_foreground_mask(contour, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    c_int = contour.astype(np.int32)
    cv2.fillPoly(mask, [c_int], 1)
    return mask

def extract_annotation_keypoints(annotation):
    """Return keypoints from an annotation if present (v>0)."""
    kpts = annotation.get('keypoints', [])
    points = []
    if len(kpts) % 3 == 0:
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i:i+3]
            if v > 0:
                points.append((x,y))
    elif len(kpts) % 2 == 0:
        arr = np.array(kpts, dtype=np.float32).reshape(-1,2)
        points = [tuple(a) for a in arr]
    return points

def calculate_relative_bending_energy(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    diffs = np.diff(contour, axis=0)
    angles = []
    for i in range(1, len(diffs)):
        v1 = diffs[i-1]
        v2 = diffs[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1==0 or n2==0:
            angle=0
        else:
            c = np.dot(v1,v2)/(n1*n2)
            c = np.clip(c, -1,1)
            angle = np.arccos(c)
        angles.append(angle)
    bend = np.sum(np.array(angles)**2)
    total_len = np.sum(np.linalg.norm(diffs, axis=1))
    if total_len==0:
        return 0
    return bend/(total_len**2)

def resample_contour_evenly(contour, num_points):
    if len(contour)<2:
        raise ValueError("Contour must have at least two points for resampling.")
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    dists = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumdist = np.cumsum(dists)
    cumdist = np.insert(cumdist, 0, 0)
    total_len = cumdist[-1]
    from scipy.interpolate import interp1d
    fx = interp1d(cumdist, contour[:,0], kind='linear')
    fy = interp1d(cumdist, contour[:,1], kind='linear')
    new_dist = np.linspace(0, total_len, num_points)
    newx = fx(new_dist)
    newy = fy(new_dist)
    res = np.column_stack([newx, newy])
    if not np.array_equal(res[0], res[-1]):
        res = np.vstack([res, res[0]])
    return res

###############################
# Save Masks
###############################
def create_and_save_masks(contour, image, out_dir, cat_name, filename_prefix):
    """
    Save binary + foreground mask for the requested category.
    """
    bin_mask = extract_foreground_mask(contour, image.shape[:2])
    fore_mask = np.zeros_like(image)
    fore_mask[bin_mask==1] = image[bin_mask==1]
    cat_out = os.path.join(out_dir, cat_name)
    os.makedirs(cat_out, exist_ok=True)

    base_clean = clean_filename(filename_prefix)
    cat_clean = clean_filename(cat_name)

    bin_path = os.path.join(cat_out, f"{base_clean}_binary_mask_{cat_clean}.png")
    fore_path = os.path.join(cat_out, f"{base_clean}_foreground_mask_{cat_clean}.png")

    cv2.imwrite(bin_path, bin_mask*255)
    cv2.imwrite(fore_path, fore_mask)
    logging.info(f"Saved binary mask to: {bin_path}")
    logging.info(f"Saved foreground mask to: {fore_path}")

###############################
# Old SHIFT-BASED PROCRUSTES
###############################
def procrustes_analysis_no_reflection(X, Y):
    """
    The old method from V5: Standard Procrustes (no reflection).
    Both X,Y: (#points,2)
    """
    if X.shape[0] != Y.shape[0] or X.shape[1] != 2 or Y.shape[1] != 2:
        raise ValueError(f"X and Y must be (#points,2) with same #points. Shapes: {X.shape}, {Y.shape}")
    X0 = X - np.mean(X, axis=0)
    Y0 = Y - np.mean(Y, axis=0)
    ssX = np.sum(X0**2)
    ssY = np.sum(Y0**2)
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    A = np.dot(Y0.T, X0)
    U, s, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    # enforce no reflection
    if np.linalg.det(R)<0:
        Vt[-1,:]*=-1
        R = np.dot(U, Vt)
    Z = np.dot(Y0, R)
    d = np.sum((X0-Z)**2)
    return d, Z, None  # tform not needed here

def reflect_contour(contour, axis):
    centroid = np.mean(contour, axis=0)
    c = contour.copy()
    if axis=='horizontal':
        c[:,0] = 2*centroid[0] - c[:,0]
    elif axis=='vertical':
        c[:,1] = 2*centroid[1] - c[:,1]
    return c

def align_contour_with_shift(reference_contour, contour, method='without_reflection'):
    """
    Old approach: try all cyclic shifts, optionally reflection combos, pick best.
    """
    num_points = len(contour)
    min_d = np.inf
    best_aligned = None
    if method == 'with_reflection':
        refl_options = ['none','horizontal','vertical','both']
    else:
        refl_options = ['none']

    for shift in range(num_points):
        shifted = np.roll(contour, shift, axis=0)
        for refl in refl_options:
            if refl=='horizontal':
                test_contour = reflect_contour(shifted, 'horizontal')
            elif refl=='vertical':
                test_contour = reflect_contour(shifted, 'vertical')
            elif refl=='both':
                test_contour = reflect_contour(shifted, 'horizontal')
                test_contour = reflect_contour(test_contour, 'vertical')
            else:
                test_contour = shifted

            # if shapes differ, resample
            if reference_contour.shape != test_contour.shape:
                n_pts = min(reference_contour.shape[0], test_contour.shape[0])
                ref_res = resample_contour_evenly(reference_contour, n_pts)
                test_res = resample_contour_evenly(test_contour, n_pts)
            else:
                ref_res = reference_contour
                test_res = test_contour

            d, aligned, _ = procrustes_analysis_no_reflection(ref_res, test_res)
            if d < min_d:
                min_d = d
                best_aligned = aligned

    return best_aligned

def procrustes_alignment(reference_contour, contours, method='without_reflection'):
    """
    Align each shape in 'contours' to reference_contour using the old shift-based approach.
    """
    aligned = []
    for c in contours:
        try:
            a = align_contour_with_shift(reference_contour, c, method=method)
            aligned.append(a)
        except Exception as e:
            logging.error(f"Error in procrustes alignment: {e}")
            traceback.print_exc()
            aligned.append(c)
    return aligned


###############################
# Visualization
###############################
def label_semilandmarks(ax, shape, color='purple'):
    """
    Label a few points (1, 1/5, 2/5, 3/5, 4/5, last) in purple
    so we can see the alignment visually.
    """
    n = shape.shape[0]
    if n < 5:
        indices = [0, n-1]
    else:
        step = n // 5
        indices = [step*i for i in range(5)]
        indices.append(n-1)
        indices = sorted(set(indices))
    for i in indices:
        pt = shape[i]
        ax.text(pt[0], pt[1], f"{i+1}", color=color, fontsize=8)

def visualize_aligned_semilandmarks(shapes, filenames, output_dir, category_name):
    plt.figure(figsize=(10,10))
    for idx, shp in enumerate(shapes):
        closed = shp if np.array_equal(shp[0], shp[-1]) else np.vstack([shp, shp[0]])
        plt.plot(closed[:,0], closed[:,1], '-o', label=filenames[idx] if idx==0 else "")
        label_semilandmarks(plt.gca(), shp, color='purple')
    plt.title(f"Aligned Semi-landmarks for {category_name}")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal','box')
    out_fn = os.path.join(output_dir, f"aligned_semilandmarks_{clean_filename(category_name)}.png")
    plt.savefig(out_fn, dpi=300)
    plt.close()

def visualize_aligned_keypoints(shapes, filenames, output_dir, category_name):
    plt.figure(figsize=(10,10))
    for idx, kpts in enumerate(shapes):
        color = plt.cm.tab20(idx % 20)
        for i, pt in enumerate(kpts):
            plt.scatter(pt[0], pt[1], color=color, s=50)
            plt.text(pt[0], pt[1], f'{idx+1}-KP{i+1}', color='darkblue', fontsize=7)
    plt.title(f"Aligned Keypoints for {category_name}")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal','box')
    out_fn = os.path.join(output_dir, f"aligned_keypoints_{clean_filename(category_name)}.png")
    plt.savefig(out_fn, dpi=300)
    plt.close()

def save_keypoints_to_csv(aligned_keypoints, filenames, output_dir, category_name):
    rows = []
    for idx, kpts in enumerate(aligned_keypoints):
        for j, pt in enumerate(kpts):
            rows.append([filenames[idx], f'KP{j+1}', pt[0], pt[1]])
    df = pd.DataFrame(rows, columns=['filename','keypoint','x','y'])
    csv_path = os.path.join(output_dir, f"aligned_keypoints_{clean_filename(category_name)}.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Aligned keypoints CSV saved to: {csv_path}")

###############################
# PCA, UMAP, DBSCAN, etc.
###############################
def create_thumbnail(image, mask, thumbnail_size=(80,80)):
    if len(image.shape)==2 or image.shape[2]==1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=np.uint8)*255
    elif len(mask.shape)>2:
        mask = mask[:,:,0]
    binary_mask = (mask>0).astype(np.uint8)
    foreground = np.zeros_like(image)
    foreground[binary_mask==1] = image[binary_mask==1]
    if image.ndim==3 and image.shape[2]==3:
        foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    else:
        foreground_rgb = foreground
    pil_img = Image.fromarray(foreground_rgb).convert("RGBA")
    pil_mask = Image.fromarray((binary_mask*255).astype(np.uint8)).convert("L")
    pil_img.putalpha(pil_mask)
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)
    new_img = Image.new('RGBA', thumbnail_size, (255,255,255,0))
    offset = ((thumbnail_size[0]-pil_img.size[0])//2,
              (thumbnail_size[1]-pil_img.size[1])//2)
    new_img.paste(pil_img, offset)
    return np.array(new_img)

def perform_pca(contours, n_components=2):
    flattened = [c.flatten() for c in contours]
    data = np.array(flattened)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(data_scaled)
    logging.debug(f"PCA variance ratios: {pca.explained_variance_ratio_}")
    return pc, pca, data_scaled

def perform_umap(X, n_components=2, random_state=42, n_neighbors=15, min_dist=0.1):
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state,
                           n_neighbors=n_neighbors, min_dist=min_dist)
    umap_results = umap_model.fit_transform(X)
    logging.debug(f"UMAP completed with n_neighbors={n_neighbors}, min_dist={min_dist}.")
    return umap_results, umap_model

#def perform_pacmap(X, n_components=2, random_state=42, num_neighbors=None):
#    """
#    Performs PaCMAP on the given scaled data array X.
#    X should be shape (n_samples, n_features).
#
#    :param X: np.array of shape (n_samples, n_features)
#    :param n_components: int, number of dimensions in the embedding (usually 2 or 3)
#    :param random_state: int, for reproducibility
#    :param num_neighbors: int or None, if you wish to set n_neighbors for PaCMAP
#    :return: np.array of shape (n_samples, n_components)
#    """
#    logging.info(f"Performing PaCMAP with n_components={n_components} ...")
#    # Initialize PaCMAP
#    reducer = PaCMAP(
#        n_components=n_components,
#        MN_ratio=0.5,        # These are typical defaults; adjust if needed
#        FP_ratio=2.0,
#        random_state=random_state
#    )
    # If you want to set the number of neighbors, you can do:
    # reducer = PaCMAP(n_components=n_components, n_neighbors=num_neighbors, ...)
    
    # Fit and transform
#    pacmap_embedding = reducer.fit_transform(X, init="pca")
#    logging.info("PaCMAP completed.")
#    return pacmap_embedding


def plot_embedding(pc, images, masks, output_dir, category_name, method='PCA'):
    # new if we only got 1 principal component, pad a zero second dimension
    if pc.ndim == 2 and pc.shape[1] == 1:
        pc = np.hstack([pc, np.zeros((pc.shape[0],1), dtype=pc.dtype)])
        #new
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_title(f"{method} of {category_name}")
    ax.scatter(pc[:,0], pc[:,1], alpha=0.0)
    x_min, x_max = pc[:,0].min(), pc[:,0].max()
    y_min, y_max = pc[:,1].min(), pc[:,1].max()
    ax.set_xlim(x_min-(x_max-x_min)*0.1, x_max+(x_max-x_min)*0.1)
    ax.set_ylim(y_min-(y_max-y_min)*0.1, y_max+(y_max-y_min)*0.1)
    for i,(p, img, msk) in enumerate(zip(pc, images, masks)):
        thumb = create_thumbnail(img, msk)
        if thumb is None:
            continue
        ibox = OffsetImage(thumb, zoom=0.75)
        ab = AnnotationBbox(ibox, (p[0], p[1]), frameon=False, pad=0.0)
        ax.add_artist(ab)
    out_fn = os.path.join(output_dir, f"{method.lower()}_{clean_filename(category_name)}_with_thumbnails.png")
    plt.savefig(out_fn, dpi=300)
    plt.close()
    logging.debug(f"{method} plot saved to: {out_fn}")

def perform_hierarchical_clustering(X, output_dir, category_name, principal_components, images, masks,
                                    method='ward', metric='euclidean', num_clusters=3):
    Z = linkage(X, method=method, metric=metric)
    from scipy.cluster.hierarchy import fcluster
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    logging.debug(f"Hierarchical clustering formed {num_clusters} clusters using method '{method}' and metric '{metric}'.")
    plt.figure(figsize=(16,12))
    unique = np.unique(labels)
    colors = sns.color_palette(None, len(unique))
    for lab, color in zip(unique, colors):
        lab_name = f'Cluster {lab}'
        mask_lab = (labels == lab)
        xy = principal_components[mask_lab]
        plt.scatter(xy[:,0], xy[:,1], c=[color], label=lab_name, alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.title(f'Hierarchical Clustering of {category_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    for i,(pc_, img, msk, lab) in enumerate(zip(principal_components, images, masks, labels)):
        thumb = create_thumbnail(img, msk)
        if thumb is None:
            continue
        ibox = OffsetImage(thumb, zoom=0.5)
        ab = AnnotationBbox(ibox, (pc_[0], pc_[1]), frameon=False, pad=0.0)
        plt.gca().add_artist(ab)
    out_fn = os.path.join(output_dir, f"hierarchical_clustering_{clean_filename(category_name)}.png")
    plt.savefig(out_fn, dpi=300)
    plt.close()
    logging.debug(f"Hierarchical clustering plot saved to: {out_fn}")
    return labels

def perform_manova(pca_scores, group_labels, output_dir):
    try:
        n_components = pca_scores.shape[1]
        pc_cols = [f'PC{i+1}' for i in range(n_components)]
        df = pd.DataFrame(pca_scores, columns=pc_cols)
        df['group'] = group_labels
        valid_df = df[(df['group'] != 'Unknown') & (df['group'].notnull())]
        if valid_df.empty:
            print("Error: No valid group labels found. Cannot perform MANOVA.")
            return
        if len(valid_df['group'].unique()) < 2:
            print("Error: At least two groups are required to perform MANOVA.")
            return
        Y = valid_df[pc_cols]
        X = valid_df[['group']]
        X_encoded = pd.get_dummies(X['group'], drop_first=False)
        X_encoded = sm.add_constant(X_encoded)
        Y = Y.apply(pd.to_numeric)
        X_encoded = X_encoded.apply(pd.to_numeric)
        if Y.isnull().values.any() or X_encoded.isnull().values.any():
            print("Error: Missing values. Cannot perform MANOVA.")
            return
        manova = MANOVA(endog=Y, exog=X_encoded)
        results = manova.mv_test()
        stat_path = os.path.join(output_dir, 'manova_results.txt')
        with open(stat_path, 'w') as f:
            f.write(str(results))
        print(f"MANOVA results saved to: {stat_path}")
    except Exception as e:
        logging.error(f"Error performing MANOVA: {e}")
        traceback.print_exc()

###############################
# Main Pipeline
###############################
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # parse keypoints indices
    if args.keypoints_indices:
        if '-' in args.keypoints_indices:
            parts = args.keypoints_indices.split('-')
            kp_indices = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            kp_indices = [int(x.strip()) for x in args.keypoints_indices.split(',') if x.strip()]
        kp_indices = [i-1 for i in kp_indices]
    else:
        kp_indices = None

    # load annotations
    all_anns_per_image, images_info, categories = load_annotations(args.json, args.image_id)
    if not all_anns_per_image:
        print("No annotations found. Exiting.")
        return

    # identify keypoints category
    keypoints_cat_ids = [cid for cid, cname in categories.items() if cname.lower() == 'keypoints']

    # segmentation categories
    if args.category_name.strip() == '':
        segmentation_categories = [(cid, cname) for cid, cname in categories.items() if cname.lower() != 'keypoints']
    else:
        seg_cid = next((cid for cid, cname in categories.items() if cname.lower() == args.category_name.lower()), None)
        if seg_cid is None:
            print(f"No category matching '{args.category_name}'. Exiting.")
            return
        segmentation_categories = [(seg_cid, args.category_name)]

    for cat_id, cat_name in segmentation_categories:
        shapes_semiland, shapes_keypts = [], []
        filenames, images_list, masks_list = [], [], []

        out_dir = os.path.join(args.output_dir, clean_filename(cat_name))
        os.makedirs(out_dir, exist_ok=True)

        # 1) Gather and resample
        for img_id, ann_list in all_anns_per_image.items():
            img_info = images_info.get(img_id)
            if not img_info:
                continue
            img_fn = img_info['file_name']
            img = cv2.imread(os.path.join(args.image_dir, img_fn))
            if img is None:
                continue
            h, w = img.shape[:2]

            # best segmentation
            segs = [a for a in ann_list if a['category_id'] == cat_id]
            if not segs:
                continue
            best = max(segs, key=lambda a: a.get('area', 0))
            mask = create_mask_from_annotation(best, h, w)
            cts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cts:
                continue
            contour = remove_duplicate_points(cts[0].reshape(-1,2))
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])

            # keypoints
            kpts = []
            for a in ann_list:
                if a['category_id'] in keypoints_cat_ids:
                    kpts.extend(extract_annotation_keypoints(a))
            if kp_indices and kpts:
                kpts = [kpts[i] for i in kp_indices if i < len(kpts)]

            # resample
            res = resample_contour_evenly(contour, args.num_landmarks)

            shapes_semiland.append(res)
            shapes_keypts.append(np.array(kpts) if kpts else np.empty((0,2)))
            fname = f"{os.path.basename(img_fn)}_{img_id}"
            filenames.append(fname)
            images_list.append(img)
            masks_list.append(extract_foreground_mask(contour, (h, w)))

            create_and_save_masks(contour, img, out_dir, cat_name, fname)

        if not shapes_semiland:
            logging.warning(f"No shapes for '{cat_name}', skipping.")
            continue

        # 2) Align
        min_pts = min(s.shape[0] for s in shapes_semiland)
        eq_semiland = [resample_contour_evenly(s, min_pts) for s in shapes_semiland]
        aligned_semiland = procrustes_alignment(eq_semiland[0], eq_semiland, method=args.alignment_method)
        visualize_aligned_semilandmarks(aligned_semiland, filenames, out_dir, cat_name)

        # 3) Raw ordered output
        try:
            raw_coco = {'images': [], 'annotations': [], 'categories': [
                {'id':1,'name':'raw_ordered_semilandmarks'}
            ]}
            tps = []
            for i, fn in enumerate(filenames):
                lm = eq_semiland[i]
                xs = lm[:,0]; ys = lm[:,1]
                raw_coco['images'].append({'id':i+1,'file_name':fn})
                raw_coco['annotations'].append({
                    'id':i+1,'image_id':i+1,'category_id':1,
                    'segmentation':[lm.flatten().tolist()],
                    'bbox':[float(np.min(xs)),float(np.min(ys)),float(np.ptp(xs)),float(np.ptp(ys))],
                    'area':float(np.ptp(xs)*np.ptp(ys)),'iscrowd':0
                })
                tps.append(f"LM={lm.shape[0]}")
                for x,y in lm:
                    tps.append(f"{x} {y}")
                tps.append(f"IMAGE={fn}\n")
            with open(os.path.join(out_dir,'raw_ordered_coco.json'),'w') as f:
                json.dump(raw_coco,f,indent=2)
            with open(os.path.join(out_dir,'raw_ordered.TPS'),'w') as f:
                f.write("\n".join(tps))
            logging.info(f"Raw semilandmarks saved for '{cat_name}'")
            # plot raw
            fig, ax = plt.subplots(figsize=(10,10))
            for shp in eq_semiland:
                closed = shp if np.array_equal(shp[0], shp[-1]) else np.vstack([shp, shp[0]])
                ax.plot(closed[:,0], closed[:,1], '-o')
            ax.set_title(f"Raw ordered semi-landmarks for {cat_name}")
            ax.invert_yaxis(); ax.set_aspect('equal','box')
            plt.savefig(os.path.join(out_dir, f"raw_ordered_semilandmarks_plot.png"), dpi=300)
            plt.close()
        except Exception as e:
            logging.error(f"Raw output error: {e}")

        # helper for back-transform
        def compute_transformation_params(original, aligned):
            mean_o = np.mean(original, axis=0)
            Y0 = original - mean_o
            norm_Y = np.sqrt((Y0**2).sum())
            if norm_Y == 0:
                R = np.eye(2)
            else:
                Y0n = Y0 / norm_Y
                R, _, _, _ = np.linalg.lstsq(Y0n, aligned, rcond=None)
            return R, mean_o, norm_Y
        def back_transform(aligned_pts, R, mean_o, norm_Y):
            return aligned_pts.dot(R.T) * norm_Y + mean_o

        # 4) Back-transformed semilandmarks
        try:
            back_coco = {'images': [], 'annotations': [], 'categories': [
                {'id':1,'name':'back_transformed_semilandmarks'}
            ]}
            back_tps = []
            for i, fn in enumerate(filenames):
                orig = eq_semiland[i]; aligned = aligned_semiland[i]
                R, mo, nY = compute_transformation_params(orig, aligned)
                back = back_transform(aligned, R, mo, nY)
                xs = back[:,0]; ys = back[:,1]
                back_coco['images'].append({'id':i+1,'file_name':fn})
                back_coco['annotations'].append({
                    'id':i+1,'image_id':i+1,'category_id':1,
                    'segmentation':[back.flatten().tolist()],
                    'bbox':[float(np.min(xs)),float(np.min(ys)),float(np.ptp(xs)),float(np.ptp(ys))],
                    'area':float(np.ptp(xs)*np.ptp(ys)),'iscrowd':0
                })
                back_tps.append(f"LM={back.shape[0]}")
                for x,y in back:
                    back_tps.append(f"{x} {y}")
                back_tps.append(f"IMAGE={fn}\n")
            with open(os.path.join(out_dir,'back_transformed_coco.json'),'w') as f:
                json.dump(back_coco,f,indent=2)
            with open(os.path.join(out_dir,'back_transformed.TPS'),'w') as f:
                f.write("\n".join(back_tps))
            logging.info(f"Back-transformed semilandmarks saved for '{cat_name}'")
        except Exception as e:
            logging.error(f"Back-transformed output error: {e}")

        # 5) Downstream analysis
        n = len(aligned_semiland)
        maxc = n-1 if n>1 else 1
        pca_res, _, data_scaled = perform_pca(aligned_semiland, n_components=maxc)
        if pca_res.ndim==2 and pca_res.shape[1]==1:
            pca_plot = np.hstack([pca_res, np.zeros((n,1))])
        else:
            pca_plot = pca_res[:,:2]
        plot_embedding(pca_plot, images_list, masks_list, out_dir, cat_name, method='PCA')

        if n>=5:
            n_nb = min(args.umap_n_neighbors, n-1)
            if n_nb>1:
                umap_res,_ = perform_umap(data_scaled,2,n_neighbors=n_nb,min_dist=args.umap_min_dist)
                plot_embedding(umap_res, images_list, masks_list, out_dir, cat_name, method='UMAP')
            db = DBSCAN(eps=0.5, min_samples=5)
            db_labels = db.fit_predict(data_scaled)
        else:
            logging.warning(f"Skipping UMAP & DBSCAN for '{cat_name}' (need≥5, got {n})")
            umap_res = np.zeros((n,2)); db_labels = np.full(n,-1)

        if pca_plot.shape[1]==1:
            hc_plot = np.hstack([pca_plot, np.zeros((n,1))])
        else:
            hc_plot = pca_plot
        hier = perform_hierarchical_clustering(data_scaled, out_dir, cat_name, hc_plot, images_list, masks_list)

        df = pd.DataFrame({
            'filename':filenames,
            'PCA1':pca_plot[:,0],'PCA2':pca_plot[:,1],
            'UMAP1':umap_res[:,0],'UMAP2':umap_res[:,1],
            'DBSCAN':db_labels,'Hierarchical':hier
        })
        df.to_csv(os.path.join(out_dir,f"{clean_filename(cat_name)}_analysis_results.csv"), index=False)
        logging.info(f"Results CSV saved for '{cat_name}'")

    logging.info("All done.")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

