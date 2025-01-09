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
from scipy.spatial import procrustes
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA

def parse_args():
    parser = argparse.ArgumentParser(description='Perform semi-landmarking and Procrustes transformation on object contours in images.')
    parser.add_argument('--json', required=True, help='Path to the COCO JSON file')
    parser.add_argument('--image_dir', required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', required=False, help='Directory to save outputs', default='./outputs')
    parser.add_argument('--image_id', type=int, required=False, help='ID of the image to process (optional)')
    parser.add_argument('--category_name', required=False, help='Name of the category to process', default=None)
    # Parameters for semi-landmarking
    parser.add_argument('--num_landmarks', type=int, default=100, help='Number of semi-landmarks to resample the contour to')
    # Mode selection
    parser.add_argument('--mode', choices=['2D', '3D'], default='2D', help='Mode of alignment: 2D or 3D')
    # Optional group labels mapping
    parser.add_argument('--group_labels', required=False, help='Path to CSV/TSV file mapping image filenames to group labels, e.g., headers (filename,group_label) (optional)')
    # Optional flag to perform MANOVA
    parser.add_argument('--perform_manova', action='store_true', help='Perform MANOVA analysis on PCA scores (requires group labels)')
    # Add alignment method argument
    parser.add_argument('--alignment_method', choices=['without_reflection', 'with_reflection'], default='without_reflection', help='Method for Procrustes alignment')
    return parser.parse_args()

def clean_filename(s):
    # Replace spaces with underscores and remove any characters that are not alphanumeric, underscore, or hyphen
    s = s.replace(' ', '_')
    return "".join(c for c in s if c.isalnum() or c in ('_', '-')).rstrip()

def load_annotations(json_path, image_id=None, category_name=None):
    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)
    
    # Create mappings
    categories = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    images = {img['id']: img for img in coco.get('images', [])}
    
    # Determine which image IDs to process
    if image_id is not None:
        image_ids = [image_id]
    else:
        image_ids = list(images.keys())
    
    # Collect annotations for the selected image IDs
    annotations_per_image = {}
    for img_id in image_ids:
        img = images.get(img_id)
        if img is None:
            continue
        
        image_filename = img.get('file_name', '')
        
        # Skip images in subdirectories
        if '/' in image_filename or '\\' in image_filename:
            print(f"Skipping image '{image_filename}' as it is in a subdirectory.")
            continue
        
        anns = [ann for ann in coco.get('annotations', []) if ann['image_id'] == img_id]
        if category_name:
            category_ids = [cat_id for cat_id, name in categories.items() if name == category_name]
            anns = [ann for ann in anns if ann['category_id'] in category_ids]
        if anns:
            annotations_per_image[img_id] = anns
    
    return annotations_per_image, images, categories

def create_mask_from_annotation(annotation, height, width):
    # COCO annotations can be in segmentation or RLE format
    if 'segmentation' in annotation:
        # Handle both polygon and RLE segmentations
        if isinstance(annotation['segmentation'], list):
            # Polygon format
            rles = maskUtils.frPyObjects(annotation['segmentation'], height, width)
            rle = maskUtils.merge(rles)
        else:
            # RLE format
            rle = annotation['segmentation']
        try:
            mask = maskUtils.decode(rle)
        except Exception as e:
            print(f"Error decoding mask: {e}")
            mask = np.zeros((height, width), dtype=np.uint8)
    else:
        # Handle other types of annotations if necessary
        mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert to binary mask (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Clean the binary mask
    kernel_size = 5  # Adjust kernel size as needed
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Fill small holes and gaps
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # Remove small noise
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask

def resample_contour_evenly(contour, num_points):
    """
    Resamples the contour to have a fixed number of points, evenly spaced based on arc length.
    
    Parameters:
        contour (numpy.ndarray): Contour points (N, 2).
        num_points (int): Number of points to resample to.
    
    Returns:
        resampled_contour (numpy.ndarray): Resampled contour points (num_points, 2).
    """
    if len(contour) < 2:
        raise ValueError("Contour must have at least two points for resampling.")
    
    # Ensure the contour is closed
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    
    # Compute cumulative arc length
    distances = np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1))
    cumulative_dist = np.cumsum(distances)
    cumulative_dist = np.insert(cumulative_dist, 0, 0)
    
    # Generate equally spaced distances
    even_distances = np.linspace(0, cumulative_dist[-1], num_points)
    
    # Interpolate new points using linear interpolation
    fx = interpolate.interp1d(cumulative_dist, contour[:, 0], kind='linear')
    fy = interpolate.interp1d(cumulative_dist, contour[:, 1], kind='linear')
    new_x = fx(even_distances)
    new_y = fy(even_distances)
    
    resampled_contour = np.column_stack((new_x, new_y))
    
    # Ensure the resampled contour is closed
    if not np.array_equal(resampled_contour[0], resampled_contour[-1]):
        resampled_contour = np.vstack([resampled_contour, resampled_contour[0]])
    
    return resampled_contour

def remove_duplicate_points(contour, tolerance=1e-3):
    """
    Removes duplicate or nearly duplicate points from the contour.
    
    Parameters:
        contour (numpy.ndarray): Contour points (N, 2).
        tolerance (float): Distance threshold to consider points as duplicates.
    
    Returns:
        unique_contour (numpy.ndarray): Contour without duplicates.
    """
    if len(contour) == 0:
        return contour
    unique_contour = [contour[0]]
    for point in contour[1:]:
        if np.linalg.norm(point - unique_contour[-1]) > tolerance:
            unique_contour.append(point)
    return np.array(unique_contour)

def create_and_save_masks(contour, image, filename, output_dir, category_name):
    """
    Creates binary and foreground masks and saves them to file.

    Parameters:
        contour (numpy.ndarray): Contour points (N, 2).
        image (numpy.ndarray): Original image.
        filename (str): Base filename for saving masks.
        output_dir (str): Directory to save the masks.
        category_name (str): Category name for organizing masks.

    Returns:
        binary_mask (numpy.ndarray): Binary mask.
        foreground_mask (numpy.ndarray): Foreground mask.
    """
    # Create binary mask
    mask_shape = image.shape[:2]
    binary_mask = extract_foreground_mask(contour, mask_shape)
    
    # Create foreground mask
    foreground_mask = np.zeros_like(image)
    foreground_mask[binary_mask == 1] = image[binary_mask == 1]
    
    # Define category-specific output directory
    category_output_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_output_dir, exist_ok=True)
    
    # Clean the category name for filename use
    cleaned_category_name = clean_filename(category_name)
    
    # Define mask filenames with unique identifiers and add category name as a suffix
    binary_mask_filename = f"binary_mask_{filename}.png"
    binary_mask_filename = add_suffix_before_extension(binary_mask_filename, cleaned_category_name)

    foreground_mask_filename = f"foreground_mask_{filename}.png"
    foreground_mask_filename = add_suffix_before_extension(foreground_mask_filename, cleaned_category_name)
    
    # Save masks
    binary_mask_path = os.path.join(category_output_dir, binary_mask_filename)
    foreground_mask_path = os.path.join(category_output_dir, foreground_mask_filename)
    
    cv2.imwrite(binary_mask_path, binary_mask * 255)  # Convert to 0-255
    cv2.imwrite(foreground_mask_path, foreground_mask)
    
    print(f"Saved binary mask to: {binary_mask_path}")
    print(f"Saved foreground mask to: {foreground_mask_path}")
    
    return binary_mask, foreground_mask

def extract_foreground_mask(contour, image_shape):
    """
    Creates a binary mask with pixels inside the contour set to 1 and outside set to 0.
    
    Parameters:
        contour (numpy.ndarray): Contour points (N, 2).
        image_shape (tuple): Shape of the image (height, width).
    
    Returns:
        mask (numpy.ndarray): Binary mask.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    contour_int = contour.astype(np.int32)
    cv2.fillPoly(mask, [contour_int], 1)
    return mask

def add_suffix_before_extension(filename, suffix):
    """
    Adds a suffix before the file extension in a filename.
    
    Parameters:
        filename (str): Original filename.
        suffix (str): Suffix to add.
    
    Returns:
        new_filename (str): Filename with suffix added before the extension.
    """
    base_name, ext = os.path.splitext(filename)
    return f"{base_name}_{suffix}{ext}"

def order_contour_points(contour):
    """
    Orders contour points in a consistent manner based on minimal distance between consecutive points.

    Parameters:
        contour (numpy.ndarray): Contour points (N, 2).

    Returns:
        ordered_contour (numpy.ndarray): Ordered contour points.
    """
    # Start with the first point
    ordered_contour = [contour[0]]
    remaining_points = list(contour[1:])
    
    while remaining_points:
        last_point = ordered_contour[-1]
        # Find the closest point to the last point
        distances = np.linalg.norm(remaining_points - last_point, axis=1)
        min_index = np.argmin(distances)
        ordered_contour.append(remaining_points.pop(min_index))
    
    return np.array(ordered_contour)

def reorder_landmarks_to_match_reference(contour, reference_contour):
    """
    Reorders the landmarks in 'contour' to match the ordering of 'reference_contour'.
    This is done by finding the index that minimizes the sum of distances to the reference contour when rotated.

    Parameters:
        contour (numpy.ndarray): Contour points to reorder (N, 2).
        reference_contour (numpy.ndarray): Reference contour points (N, 2).

    Returns:
        reordered_contour (numpy.ndarray): Reordered contour points (N, 2).
    """
    num_points = len(contour)
    min_distance = np.inf
    best_shift = 0

    for shift in range(num_points):
        shifted_contour = np.roll(contour, -shift, axis=0)
        distance = np.sum(np.linalg.norm(shifted_contour - reference_contour, axis=1))
        if distance < min_distance:
            min_distance = distance
            best_shift = shift

    reordered_contour = np.roll(contour, -best_shift, axis=0)

    return reordered_contour

def procrustes_analysis_no_reflection(X, Y):
    """
    Performs Procrustes analysis to align Y to X without allowing reflections.
    This function is size invariant.

    Parameters:
        X (numpy.ndarray): Reference shape (N, K).
        Y (numpy.ndarray): Target shape to align (N, K).

    Returns:
        d (float): Squared error after transformation.
        Z (numpy.ndarray): Transformed Y.
        tform (dict): Dictionary of the transformation values.
    """
    # Center the data
    X0 = X - np.mean(X, 0)
    Y0 = Y - np.mean(Y, 0)

    # Compute scaling factors
    ssX = np.sum(X0 ** 2)
    ssY = np.sum(Y0 ** 2)

    # Normalize the shapes to remove scaling
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY

    # Compute the optimal rotation matrix using SVD
    A = np.dot(Y0.T, X0)
    U, s, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)

    # Ensure no reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)

    # Apply the rotation to Y0
    Z = np.dot(Y0, R)

    # Compute the residual error
    d = np.sum((X0 - Z) ** 2)

    # Since we normalized the shapes, scaling factor is 1
    tform = {'rotation': R, 'scale': 1.0, 'translation': np.mean(X, 0) - np.mean(Y, 0).dot(R)}

    # Apply the translation to the transformed Y
    Z = normX * Z + np.mean(X, 0)

    return d, Z, tform

def align_contour_with_shift(reference_contour, contour, method='without_reflection'):
    """
    Aligns a contour to the reference contour by cyclically shifting its points to minimize
    the Procrustes distance.

    Parameters:
        reference_contour (numpy.ndarray): Reference contour (N, 2).
        contour (numpy.ndarray): Contour to align (N, 2).
        method (str): Method of Procrustes alignment ('without_reflection', 'with_reflection').

    Returns:
        best_aligned_contour (numpy.ndarray): Aligned contour (N, 2).
    """
    num_points = len(contour)
    min_distance = np.inf
    best_aligned_contour = None

    for shift in range(num_points):
        shifted_contour = np.roll(contour, shift, axis=0)

        # Procrustes expects both matrices to have the same shape
        if reference_contour.shape != shifted_contour.shape:
            print("Warning: Reference and contour shapes do not match. Resampling to match.")
            num_points = min(reference_contour.shape[0], shifted_contour.shape[0])
            reference_resampled = resample_contour_evenly(reference_contour, num_points)
            contour_resampled = resample_contour_evenly(shifted_contour, num_points)
        else:
            reference_resampled = reference_contour
            contour_resampled = shifted_contour

        if method == 'without_reflection':
            d, aligned_contour, _ = procrustes_analysis_no_reflection(reference_resampled, contour_resampled)
        elif method == 'with_reflection':
            # Use scipy's procrustes method (allows reflection)
            mtx1, mtx2, d = procrustes(reference_resampled, contour_resampled)
            aligned_contour = mtx2
        else:
            raise ValueError(f"Unknown method '{method}' for Procrustes alignment.")

        if d < min_distance:
            min_distance = d
            best_aligned_contour = aligned_contour

    return best_aligned_contour

def procrustes_alignment(reference_contour, contours, method='without_reflection'):
    """
    Performs Procrustes analysis to align contours to the reference contour.

    Parameters:
        reference_contour (numpy.ndarray): Reference contour (N, 2).
        contours (list of numpy.ndarray): List of target contours to align.
        method (str): Method of alignment ('without_reflection', 'with_reflection').

    Returns:
        procrustes_contours (list of numpy.ndarray): List of Procrustes-aligned contours.
    """
    procrustes_contours = []
    for contour in contours:
        try:
            # Align contour with shift to minimize Procrustes distance
            aligned_contour = align_contour_with_shift(reference_contour, contour, method=method)
            procrustes_contours.append(aligned_contour)
        except Exception as e:
            print(f"Error during Procrustes analysis: {e}")
            traceback.print_exc()
            procrustes_contours.append(contour)  # Fallback to unaligned contour
    return procrustes_contours

def perform_pca(procrustes_contours, n_components=2):
    """
    Performs PCA on the flattened Procrustes-aligned contours.
    
    Parameters:
        procrustes_contours (list of numpy.ndarray): List of Procrustes-aligned contours (N, 2).
        n_components (int): Number of PCA components.
    
    Returns:
        principal_components (numpy.ndarray): PCA-transformed coordinates (N_samples, n_components).
        pca_model (sklearn.decomposition.PCA): Trained PCA model.
        data_scaled (numpy.ndarray): Scaled data used for PCA.
    """
    # Flatten the contours
    flattened_contours = [contour.flatten() for contour in procrustes_contours]
    data = np.array(flattened_contours)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_scaled)
    print(f"PCA completed. Explained variance ratios: {pca.explained_variance_ratio_}")
    return principal_components, pca, data_scaled

def perform_umap(X, n_components=2, random_state=42):
    """
    Performs UMAP dimensionality reduction.
    
    Parameters:
        X (numpy.ndarray): Scaled feature data.
        n_components (int): Number of dimensions for UMAP.
        random_state (int): Random state for reproducibility.
    
    Returns:
        umap_results (numpy.ndarray): UMAP-transformed coordinates.
        umap_model (umap.UMAP): Trained UMAP model.
    """
    import umap
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    umap_results = umap_model.fit_transform(X)
    print(f"UMAP completed with {n_components} components.")
    return umap_results, umap_model

def create_thumbnail(image, mask, thumbnail_size=(80, 80)):
    """
    Creates a thumbnail image from an image and a binary mask with a transparent background.
    Resizes the image while maintaining aspect ratio and adds padding to fit the thumbnail size.

    Parameters:
        image (numpy.ndarray): Original image (height, width, channels).
        mask (numpy.ndarray): Binary mask (height, width), where foreground pixels are 1.
        thumbnail_size (tuple): Size to resize the thumbnail (width, height).

    Returns:
        foreground_rgba (numpy.ndarray): RGBA thumbnail image with transparent background.
    """
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Create foreground by masking the image
    foreground = np.zeros_like(image)
    foreground[binary_mask == 1] = image[binary_mask == 1]

    # Convert BGR to RGB if image has 3 channels
    if foreground.shape[2] == 3:
        foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    else:
        foreground_rgb = foreground  # Assume already RGB

    # Convert to PIL Image
    pil_img = Image.fromarray(foreground_rgb)

    # Create alpha channel based on the mask
    pil_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))

    # Combine image and mask to create RGBA image
    pil_img.putalpha(pil_mask)

    # Resize while maintaining aspect ratio
    original_size = pil_img.size
    pil_img.thumbnail(thumbnail_size, Image.LANCZOS)  # Resizes in place

    # Create a new image with the desired thumbnail size and transparent background
    new_img = Image.new('RGBA', thumbnail_size, (255, 255, 255, 0))
    # Calculate position to center the thumbnail
    offset = ((thumbnail_size[0] - pil_img.size[0]) // 2,
              (thumbnail_size[1] - pil_img.size[1]) // 2)
    new_img.paste(pil_img, offset)

    # Convert back to NumPy array
    foreground_rgba = np.array(new_img)

    return foreground_rgba

def plot_pca_umap(principal_components, images, masks, output_dir, category_name, method='PCA', thumbnail_size=(80, 80)):
    """
    Plots PCA or UMAP analysis with thumbnails of the foreground masks placed at their reduced coordinates.

    Parameters:
        principal_components (numpy.ndarray): PCA or UMAP-transformed coordinates (N_samples, 2).
        images (list of numpy.ndarray): List of original images corresponding to masks.
        masks (list of numpy.ndarray): List of binary masks.
        output_dir (str): Directory to save the PCA/UMAP plot.
        category_name (str): Name of the category for labeling.
        method (str): 'PCA' or 'UMAP'.
        thumbnail_size (tuple): Size to resize thumbnails for plotting.

    Returns:
        None
    """
    print(f"Plotting {method} with Thumbnails for category '{category_name}'...")
    print("Components Shape:", principal_components.shape)
    print("First 5 Components:\n", principal_components[:5])

    # Check if data is valid
    if principal_components.size == 0:
        print(f"Error: {method} data is empty.")
        return

    # Verify that the number of data points matches
    if len(principal_components) != len(images) or len(images) != len(masks):
        print("Error: The lengths of principal_components, images, and masks do not match.")
        print(f"principal_components: {principal_components.shape}, images: {len(images)}, masks: {len(masks)}")
        return

    fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size to accommodate thumbnails
    ax.set_title(f'{method} of {category_name} Procrustes-Aligned Contours')
    ax.set_xlabel(f'{method} Component 1')
    ax.set_ylabel(f'{method} Component 2')

    # Plot invisible scatter points to set the axis limits
    ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.0)

    # Set axis limits based on coordinates
    x_min, x_max = principal_components[:, 0].min(), principal_components[:, 0].max()
    y_min, y_max = principal_components[:, 1].min(), principal_components[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    for i, (pc, mask, img) in enumerate(zip(principal_components, masks, images)):
        print(f"Processing thumbnail {i+1}/{len(images)} at {method} coordinates: {pc}")

        # Create thumbnail from image and mask
        thumbnail = create_thumbnail(img, mask, thumbnail_size)
        if thumbnail is None:
            print(f"Skipping thumbnail {i+1} due to creation failure.")
            continue

        # Create an OffsetImage with RGBA (transparent background)
        imagebox = OffsetImage(thumbnail, zoom=0.75)  # Adjust zoom as needed
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]),
                            frameon=False, pad=0.0)
        ax.add_artist(ab)

        # Optional: Save a sample thumbnail for verification
        if i == 0:
            sample_thumbnail_path = os.path.join(output_dir, 'sample_thumbnail.png')
            pil_img = Image.fromarray(thumbnail)
            pil_img.save(sample_thumbnail_path)
            print(f"Sample thumbnail saved to: {sample_thumbnail_path}")

    plt.grid(True)
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'{method.lower()}_{cleaned_category_name}_procrustes_contours_with_thumbnails.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"{method} plot with thumbnails saved to: {plot_path}")

def perform_dbscan(X, output_dir, category_name, principal_components, images, masks):
    """
    Performs DBSCAN clustering and plots the results with thumbnails.

    Parameters:
        X (numpy.ndarray): Scaled feature data.
        output_dir (str): Directory to save the DBSCAN plot.
        category_name (str): Name of the category for labeling.
        principal_components (numpy.ndarray): Coordinates for plotting.
        images (list of numpy.ndarray): List of original images corresponding to masks.
        masks (list of numpy.ndarray): List of binary masks.

    Returns:
        dbscan_labels (numpy.ndarray): Cluster labels from DBSCAN.
    """
    from sklearn.cluster import DBSCAN
    import seaborn as sns

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {num_clusters} clusters")
    
    # Add cluster labels to principal components for plotting
    plt.figure(figsize=(16, 12))
    unique_labels = set(dbscan_labels)
    colors = sns.color_palette(None, len(unique_labels))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black used for noise.
            color = (0, 0, 0, 1)
            label_name = 'Noise'
        else:
            label_name = f'Cluster {label+1}'
        class_member_mask = (dbscan_labels == label)
        xy = principal_components[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], label=label_name, alpha=0.6, edgecolors='w', linewidths=0.5)
    
    plt.title(f'DBSCAN Clustering of {category_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    
    # Optionally, add thumbnails
    for i, (pc, img, mask, label) in enumerate(zip(principal_components, images, masks, dbscan_labels)):
        if label == -1:
            continue  # Skip noise points
        thumbnail = create_thumbnail(img, mask)
        if thumbnail is None:
            continue
        imagebox = OffsetImage(thumbnail, zoom=0.5)
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]), frameon=False, pad=0.0)
        plt.gca().add_artist(ab)
    
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'dbscan_{cleaned_category_name}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"DBSCAN plot saved to: {plot_path}")

    return dbscan_labels

def perform_hierarchical_clustering(X, output_dir, category_name, principal_components, images, masks, method='ward', metric='euclidean', num_clusters=3):
    """
    Performs hierarchical clustering and plots the results with thumbnails.

    Parameters:
        X (numpy.ndarray): Scaled feature data.
        output_dir (str): Directory to save the hierarchical clustering plot.
        category_name (str): Name of the category for labeling.
        principal_components (numpy.ndarray): Coordinates for plotting.
        images (list of numpy.ndarray): List of original images corresponding to masks.
        masks (list of numpy.ndarray): List of binary masks.
        method (str): Linkage method to use ('ward', 'single', 'complete', 'average', etc.).
        metric (str): Distance metric to use ('euclidean', 'cityblock', etc.).
        num_clusters (int): Number of clusters to form.

    Returns:
        labels (numpy.ndarray): Cluster labels from hierarchical clustering.
    """
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    import seaborn as sns

    Z = linkage(X, method=method, metric=metric)
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    print(f"Hierarchical clustering formed {num_clusters} clusters using method '{method}' and metric '{metric}'.")

    # Plot dendrogram
    plt.figure(figsize=(16, 8))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(f'Hierarchical Clustering Dendrogram for {category_name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    dendrogram_plot_filename = f'hierarchical_clustering_dendrogram_{clean_filename(category_name)}.png'
    dendrogram_plot_path = os.path.join(output_dir, dendrogram_plot_filename)
    plt.savefig(dendrogram_plot_path, dpi=300)
    plt.close()
    print(f"Dendrogram plot saved to: {dendrogram_plot_path}")

    # Plot clusters with thumbnails
    plt.figure(figsize=(16, 12))
    unique_labels = np.unique(labels)
    colors = sns.color_palette(None, len(unique_labels))

    for label, color in zip(unique_labels, colors):
        label_name = f'Cluster {label}'
        class_member_mask = (labels == label)
        xy = principal_components[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[color], label=label_name, alpha=0.6, edgecolors='w', linewidths=0.5)

    plt.title(f'Hierarchical Clustering of {category_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()

    # Optionally, add thumbnails
    for i, (pc, img, mask, label) in enumerate(zip(principal_components, images, masks, labels)):
        thumbnail = create_thumbnail(img, mask)
        if thumbnail is None:
            continue
        imagebox = OffsetImage(thumbnail, zoom=0.5)
        ab = AnnotationBbox(imagebox, (pc[0], pc[1]), frameon=False, pad=0.0)
        plt.gca().add_artist(ab)

    plot_filename = f'hierarchical_clustering_{clean_filename(category_name)}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Hierarchical clustering plot saved to: {plot_path}")

    return labels

def visualize_ordered_contours(ordered_contours, filenames, output_dir, category_name):
    """
    Visualizes all ordered (resampled and evenly distributed) contours before Procrustes alignment.

    Parameters:
        ordered_contours (list of numpy.ndarray): List of ordered contours (N, 2).
        filenames (list of str): Corresponding filenames for each contour.
        output_dir (str): Directory to save the plot.
        category_name (str): Name of the category for labeling.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    
    for idx, (contour, filename) in enumerate(zip(ordered_contours, filenames)):
        # Ensure the contour is closed for plotting
        contour_closed = contour
        if not np.array_equal(contour[0], contour[-1]):
            contour_closed = np.vstack([contour, contour[0]])
        
        plt.plot(contour_closed[:, 0], contour_closed[:, 1], '-o', label=f'Contour {idx+1}' if idx == 0 else "")
        
        # Highlight first point
        plt.scatter(contour[0, 0], contour[0, 1], color='green', s=100, zorder=5)
        
        # Label subset of points
        num_landmarks = len(contour)
        landmark_indices = [0, num_landmarks//8, num_landmarks//4, 3*num_landmarks//8,
                            5*num_landmarks//8, 3*num_landmarks//4, 7*num_landmarks//8]
        for j in landmark_indices:
            if j < len(contour):
                plt.text(contour[j, 0], contour[j, 1], str(j + 1),
                         fontsize=8, color='purple', zorder=10)

    plt.title(f"Ordered {category_name} Contours (Before Procrustes Alignment)")
    plt.gca().invert_yaxis()  # Match image coordinate system
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    # Clean the category name for filename use
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'ordered_contours_{cleaned_category_name}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Ordered Contours plot saved to: {plot_path}")

def save_combined_csv(contours, filenames, output_dir, combined_csv_path):
    """
    Saves all contours into a single CSV file with an additional filename column.

    Parameters:
        contours (list of numpy.ndarray): List of contours.
        filenames (list of str): Corresponding filenames for each contour.
        output_dir (str): Directory to save the combined CSV.
        combined_csv_path (str): Path to the combined CSV file.
    """
    combined_data = []
    for contour, filename in zip(contours, filenames):
        # Flatten the contour and prepend the filename
        flat_contour = contour.flatten()
        combined_row = [filename] + flat_contour.tolist()
        combined_data.append(combined_row)

    # Determine the number of landmarks
    num_landmarks = len(contours[0])

    # Create header
    header = ['filename']
    for i in range(1, num_landmarks + 1):
        header.append(f'X{i}')
    for i in range(1, num_landmarks + 1):
        header.append(f'Y{i}')

    # Save to CSV
    try:
        with open(combined_csv_path, 'w') as f:
            f.write(','.join(header) + '\n')
            for row in combined_data:
                f.write(','.join(map(str, row)) + '\n')
        print(f"Combined CSV saved to: {combined_csv_path}")
    except Exception as e:
        print(f"Error saving combined CSV: {e}")
        traceback.print_exc()

def save_tps(reference_contour, target_contours, filenames, output_dir, combined_tps_path):
    """
    Saves contours into a TPS file, indicating the first point as a keypoint and the rest as semi-landmarks.

    Parameters:
        reference_contour (numpy.ndarray): Reference contour (N, 2).
        target_contours (list of numpy.ndarray): List of target contours (N, 2).
        filenames (list of str): Corresponding filenames for each contour.
        output_dir (str): Directory to save the TPS file.
        combined_tps_path (str): Path to the combined TPS file.
    """
    try:
        with open(combined_tps_path, 'w') as f:
            # Reference points
           # f.write("LM=" + str(len(reference_contour)) + "\n")
            # Write the keypoint (first point)
           # keypoint = reference_contour[0]
           # f.write(f"{keypoint[0]:.5f} {keypoint[1]:.5f}\n")
            # Indicate that this is a keypoint
           # f.write("KEYPOINT\n")
            # Write the remaining semi-landmarks
           # for point in reference_contour[1:]:
           #     f.write(f"{point[0]:.5f} {point[1]:.5f}\n")
           # f.write("\n")

            # Target configurations
            for contour, filename in zip(target_contours, filenames):
                f.write("LM=" + str(len(contour)) + "\n")
                # Write the keypoint (first point)
                keypoint = contour[0]
                f.write(f"{keypoint[0]:.5f} {keypoint[1]:.5f}\n")
                f.write("KEYPOINT\n")
                # Write the remaining semi-landmarks
                for point in contour[1:]:
                    f.write(f"{point[0]:.5f} {point[1]:.5f}\n")
                f.write("IMAGE=" + filename + "\n")
                f.write("\n")
        print(f"Combined TPS saved to: {combined_tps_path}")
    except Exception as e:
        print(f"Error saving TPS file: {e}")
        traceback.print_exc()

def visualize_all_procrustes_contours(procrustes_contours, reference_contour, num_landmarks, image_filenames, output_dir, category_name):
    """
    Visualizes all Procrustes-aligned contours together with highlighted first points and labeled landmarks.
    """
    plt.figure(figsize=(10, 10))
    
    for idx, (contour, filename) in enumerate(zip(procrustes_contours, image_filenames)):
        # Ensure the contour is closed for plotting
        contour_closed = contour
        if not np.array_equal(contour[0], contour[-1]):
            contour_closed = np.vstack([contour, contour[0]])
        
        plt.plot(contour_closed[:, 0], contour_closed[:, 1], '-o', label=f'Contour {idx+1}' if idx == 0 else "")
        
        # Highlight first point
        plt.scatter(contour[0, 0], contour[0, 1], color='red', s=100, zorder=5)
        
        # Label subset of points
        landmark_indices = [0, num_landmarks//8, num_landmarks//4, 3*num_landmarks//8,
                            5*num_landmarks//8, 3*num_landmarks//4, 7*num_landmarks//8]
        for j in landmark_indices:
            if j < len(contour):
                plt.text(contour[j, 0], contour[j, 1], str(j + 1),
                         fontsize=8, color='blue', zorder=10)
    
    plt.title(f"Procrustes Aligned {category_name} Contours")
    plt.gca().invert_yaxis()  # Match image coordinate system
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    # Clean the category name for filename use
    cleaned_category_name = clean_filename(category_name)
    plot_filename = f'procrustes_aligned_contours_{cleaned_category_name}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Procrustes Aligned Contours plot saved to: {plot_path}")

def load_group_labels_mapping(mapping_file_path):
    """
    Loads a CSV/TSV file mapping image filenames to group labels.

    Parameters:
        mapping_file_path (str): Path to the CSV/TSV mapping file.

    Returns:
        group_labels_dict (dict): Dictionary mapping image filenames to group labels.
    """
    import pandas as pd
    try:
        # Determine the separator based on file extension
        _, ext = os.path.splitext(mapping_file_path)
        sep = ',' if ext.lower() == '.csv' else '\t'

        mapping_df = pd.read_csv(mapping_file_path, sep=sep)
        # Assume the file has columns named 'filename' and 'group_label'
        if 'filename' not in mapping_df.columns or 'group_label' not in mapping_df.columns:
            raise ValueError("Mapping file must contain 'filename' and 'group_label' columns.")

        group_labels_dict = pd.Series(mapping_df.group_label.values, index=mapping_df.filename).to_dict()
        print(f"Group labels mapping loaded from: {mapping_file_path}")
        return group_labels_dict
    except Exception as e:
        print(f"Error reading group labels mapping file: {e}")
        traceback.print_exc()
        return None

def perform_manova(pca_scores, group_labels, output_dir):
    """
    Performs MANOVA on PCA scores and saves the results to a text file.

    Parameters:
        pca_scores (numpy.ndarray): PCA-transformed coordinates (N_samples, n_components).
        group_labels (list of str): List of group labels corresponding to each sample.
        output_dir (str): Directory to save the MANOVA results.

    Returns:
        None
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.multivariate.manova import MANOVA

        # Create a DataFrame with PCA scores and group labels
        n_components = pca_scores.shape[1]
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        data_df = pd.DataFrame(pca_scores, columns=pc_columns)
        data_df['group'] = group_labels

        # Remove samples with missing or invalid group labels
        valid_data_df = data_df[(data_df['group'] != 'Unknown') & (data_df['group'].notnull())]
        if valid_data_df.empty:
            print("Error: No valid group labels found. Cannot perform MANOVA.")
            return

        # Check if there are multiple groups
        unique_groups = valid_data_df['group'].unique()
        if len(unique_groups) < 2:
            print("Error: At least two groups are required to perform MANOVA.")
            return

        # Prepare dependent and independent variables
        Y = valid_data_df[pc_columns]  # Dependent variables (PCA scores)
        X = valid_data_df[['group']]   # Independent variable (group labels)

        # Encode the group labels
        X_encoded = pd.get_dummies(X['group'], drop_first=False)

        # Add a constant term to the independent variables
        X_encoded = sm.add_constant(X_encoded)

        # **Convert boolean columns to numeric types (int64)**
        for col in X_encoded.select_dtypes(include=['bool']).columns:
            X_encoded[col] = X_encoded[col].astype(int)

        # Convert data types explicitly
        Y = Y.apply(pd.to_numeric)
        X_encoded = X_encoded.apply(pd.to_numeric)

        # Check for missing values
        if Y.isnull().values.any() or X_encoded.isnull().values.any():
            print("Error: Missing values detected in the data. Cannot perform MANOVA.")
            return

        # Debug: Print data types and shapes
        print("Y dtypes:")
        print(Y.dtypes)
        print("X_encoded dtypes:")
        print(X_encoded.dtypes)
        print("Y shape:", Y.shape)
        print("X_encoded shape:", X_encoded.shape)
        print("Unique group labels:", X['group'].unique())

        # Fit the MANOVA model
        manova = MANOVA(endog=Y, exog=X_encoded)
        manova_results = manova.mv_test()

        # Save MANOVA results to a text file
        stat_output_path = os.path.join(output_dir, 'manova_results.txt')
        with open(stat_output_path, 'w') as f:
            f.write(str(manova_results))
        print(f"MANOVA results saved to: {stat_output_path}")

    except Exception as e:
        print(f"Error performing MANOVA: {e}")
        traceback.print_exc()

def log_problematic_contours(log_file_path, filename):
    """
    Logs the filenames of problematic contours to a text file.

    Parameters:
        log_file_path (str): Path to the log file.
        filename (str): Filename of the problematic contour.

    Returns:
        None
    """
    try:
        with open(log_file_path, 'a') as f:
            f.write(f"{filename}\n")
        print(f"Logged problematic contour: {filename}")
    except Exception as e:
        print(f"Error logging problematic contour {filename}: {e}")
        traceback.print_exc()

def main():
    args = parse_args()
    annotations_dict, images_info, categories = load_annotations(args.json, args.image_id, args.category_name)
    
    if not annotations_dict:
        if args.image_id:
            print(f"No annotations found for image ID {args.image_id}.")
        else:
            print("No annotations found in the JSON file.")
        return

    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize log file for problematic contours
    log_file_path = os.path.join(args.output_dir, 'problematic_contours.log')
    # Clear the log file if it exists
    open(log_file_path, 'w').close()
    print(f"Initialized log file at: {log_file_path}")

    # Load group labels mapping if provided
    group_labels_mapping = None
    if args.group_labels:
        group_labels_mapping = load_group_labels_mapping(args.group_labels)
    elif args.perform_manova:
        print("Error: Group labels mapping file is required to perform MANOVA.")
        sys.exit(1)

    # Prepare per-category data structures
    category_ordered_contours = {}
    category_filenames = {}
    category_original_images = {}
    category_masks = {}
    category_group_labels = {}

    for img_id, annotations in annotations_dict.items():
        image_info = images_info.get(img_id)
        if image_info is None:
            print(f"No image found with ID {img_id} in the JSON annotations.")
            continue

        # Get the image filename from the metadata
        image_filename = image_info.get('file_name', '')
        image_basename = os.path.basename(os.path.normpath(image_filename))  # Extract base filename without directories

        image_path = os.path.join(args.image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {image_path}")
            continue
        height, width = image.shape[:2]

        if not annotations:
            print(f"Image ID ('{image_filename}') {img_id}: No annotations found.")
            continue

        for ann_idx, ann in enumerate(annotations):
            category_name = categories.get(ann['category_id'], 'Unknown')
            cleaned_category_name = clean_filename(category_name)
            # Construct unique filename with category name
            unique_filename = f"{image_basename}_{cleaned_category_name}_ann{ann_idx+1}"

            mask = create_mask_from_annotation(ann, height, width)

            # Convert mask to binary
            binary_mask = (mask > 0).astype(np.uint8)

            # Extract the object's contour from the mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                print(f"Image ID {img_id}, Annotation {ann_idx+1}: No contours found.")
                continue

            object_contour = max(contours, key=cv2.contourArea)
            contour_points = object_contour.reshape(-1, 2)

            # Ensure Contour is Closed
            if not np.array_equal(contour_points[0], contour_points[-1]):
                contour_points = np.vstack([contour_points, contour_points[0]])

            # Remove duplicate points
            contour_points = remove_duplicate_points(contour_points)

            # Resample contour evenly
            resampled_contour = resample_contour_evenly(contour_points, args.num_landmarks)

            # No need to order since resampling keeps the points ordered
            ordered_contour = resampled_contour

            # Use the original contour to create and save masks
            binary_mask, foreground_mask = create_and_save_masks(
                contour_points,  # Use original contour points here
                image,
                unique_filename,
                args.output_dir,
                cleaned_category_name
            )

            # Determine group label
            if group_labels_mapping:
                # Use the mapping to get the group label
                image_basename_no_ext = os.path.splitext(image_basename)[0]
                group_label = group_labels_mapping.get(image_basename_no_ext)
                if group_label is None:
                    print(f"Warning: Group label for '{image_basename}' not found in mapping file.")
                    print(f"Available filenames in mapping (first 5): {list(group_labels_mapping.keys())[:5]} ...")
                    group_label = 'Unknown'
            else:
                group_label = None  # No group label

            # Initialize category data structures if not already present
            if category_name not in category_ordered_contours:
                category_ordered_contours[category_name] = []
                category_filenames[category_name] = []
                category_original_images[category_name] = []
                category_masks[category_name] = []
                category_group_labels[category_name] = []

            # Append data to category lists
            category_ordered_contours[category_name].append(ordered_contour)
            category_filenames[category_name].append(unique_filename)
            category_original_images[category_name].append(image)
            category_masks[category_name].append(binary_mask)
            category_group_labels[category_name].append(group_label)

    # Process each category separately
    for category_name in category_ordered_contours.keys():
        ordered_contours = category_ordered_contours[category_name]
        filenames = category_filenames[category_name]
        images = category_original_images[category_name]
        masks = category_masks[category_name]
        group_labels = category_group_labels[category_name]

        # Create output directory for the category
        category_output_dir = os.path.join(args.output_dir, clean_filename(category_name))
        os.makedirs(category_output_dir, exist_ok=True)

        if not ordered_contours:
            print(f"No valid contours found for category '{category_name}'. Skipping...")
            continue

        # === Step 1: Select the Reference Contour ===
        reference_contour = ordered_contours[0]
        print(f"Processing category '{category_name}' with {len(ordered_contours)} contours.")

        # === Step 2: Procrustes Alignment with Point Shifting ===
        # Perform Procrustes alignment to make size invariant and allow shifting of points along the contour
        procrustes_contours = procrustes_alignment(reference_contour, ordered_contours, method=args.alignment_method)

        # === Step 3: Visualize Ordered Contours Before Procrustes ===
        visualize_ordered_contours(ordered_contours, filenames, category_output_dir, category_name)

        # === Step 4: Save Individual CSV Files for Procrustes-Aligned Contours ===
        for contour, filename in zip(procrustes_contours, filenames):
            csv_filename = os.path.join(category_output_dir, f"{filename}_procrustes_aligned.csv")
            try:
                np.savetxt(csv_filename, contour, delimiter=",", fmt="%f")
                print(f"Procrustes-aligned contours saved to: {csv_filename}")
            except Exception as e:
                print(f"Error saving CSV for {filename}: {e}")
                traceback.print_exc()

        # === Step 5: Save Combined CSV for Procrustes-Aligned Contours ===
        combined_csv_path_procrustes = os.path.join(category_output_dir, f"{clean_filename(category_name)}_procrustes_aligned.csv")
        save_combined_csv(procrustes_contours, filenames, category_output_dir, combined_csv_path_procrustes)

        # === Step 6: Save Combined TPS for Procrustes-Aligned Contours ===
        combined_tps_path_procrustes = os.path.join(category_output_dir, f"{clean_filename(category_name)}_procrustes_aligned.tps")
        save_tps(reference_contour, procrustes_contours, filenames, category_output_dir, combined_tps_path_procrustes)

        # === Step 7: Perform PCA on the Procrustes-Aligned Contours ===
        n_samples = len(procrustes_contours)
        max_components = n_samples - 1 if n_samples > 1 else 1
        principal_components, pca_model, data_scaled = perform_pca(procrustes_contours, n_components=max_components)

        # === Step 8: Plot PCA with Thumbnails ===
        plot_pca_umap(principal_components[:, :2], images, masks, category_output_dir, category_name, method='PCA')

        # === Step 9: Perform UMAP and Plot ===
        umap_results, umap_model = perform_umap(data_scaled, n_components=2)
        plot_pca_umap(umap_results, images, masks, category_output_dir, category_name, method='UMAP')

        # === Step 10: Perform DBSCAN Clustering and Plot ===
        dbscan_labels = perform_dbscan(data_scaled, category_output_dir, category_name, principal_components[:, :2], images, masks)

        # === Step 11: Perform Hierarchical Clustering and Plot ===
        hierarchical_labels = perform_hierarchical_clustering(data_scaled, category_output_dir, category_name, principal_components[:, :2], images, masks)

        # === Step 12: Perform MANOVA on the PCA Scores if Requested ===
        if args.perform_manova and group_labels:
            perform_manova(principal_components, group_labels, category_output_dir)

        # === Step 13: Visualize All Procrustes-Aligned Contours Together ===
        visualize_all_procrustes_contours(procrustes_contours, reference_contour, args.num_landmarks, filenames, category_output_dir, category_name)

    print("Resampling, alignment, Procrustes analysis, PCA, UMAP, DBSCAN, hierarchical clustering, and saving completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
