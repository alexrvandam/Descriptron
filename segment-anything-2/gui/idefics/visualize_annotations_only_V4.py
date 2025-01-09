import sys
import argparse
import json
import cv2
import numpy as np
import os
import traceback
from pycocotools import mask as maskUtils

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Visualize instance segmentation annotations with contours, bounding boxes, and labels.')
    parser.add_argument('--json', required=True, help='Path to the COCO JSON file with predictions')
    parser.add_argument('--image_dir', required=True, help='Directory containing the original images')
    parser.add_argument('--output_dir', required=False, help='Directory to save annotated images', default='./annotated_outputs')
    parser.add_argument('--category_name', required=False, help='Name of the category to process (process all if not specified)', default=None)
    # Overlay options
    parser.add_argument('--overlay_bboxes', action='store_true', help='Overlay bounding boxes on images')
    parser.add_argument('--overlay_contours', action='store_true', help='Overlay contours on images')
    parser.add_argument('--overlay_labels', action='store_true', help='Overlay category labels on images')
    parser.add_argument('--color_labels', type=str, default='white', help='Color for labels (e.g., "red", "green", "blue", or "match" to match contour color)')
    parser.add_argument('--opacity', type=float, default=0.4, help='Opacity for overlays (0.0 transparent, 1.0 opaque)')
    return parser.parse_args()

def load_annotations(json_path, category_name=None):
    """
    Loads annotations from a COCO-formatted JSON file.

    Parameters:
        json_path (str): Path to the JSON file.
        category_name (str, optional): Specific category to filter annotations. Defaults to None.

    Returns:
        tuple: (annotations_per_image, images_info, categories)
            - annotations_per_image (dict): Mapping of image_id to its annotations.
            - images_info (dict): Mapping of image_id to image information.
            - categories (dict): Mapping of category_id to category_name.
    """
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

def generate_distinct_colors(n):
    """
    Generates 'n' distinct colors using the HLS color space.

    Parameters:
        n (int): Number of distinct colors to generate.

    Returns:
        list: List of BGR color tuples.
    """
    import colorsys
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for hue in hues:
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert RGB [0,1] to BGR [0,255]
        bgr = tuple(int(255 * c) for c in rgb[::-1])
        colors.append(bgr)
    return colors

def overlay_annotations(image, annotations, categories, contour_colors,
                       overlay_bboxes=False, overlay_contours=False, overlay_labels=False,
                       color_labels='white',
                       opacity=0.4):
    """
    Overlays bounding boxes, contours, and labels onto the image.

    Parameters:
        image (numpy.ndarray): Original image in BGR format.
        annotations (list of dict): List of annotations for the image.
        categories (dict): Mapping from category_id to category_name.
        contour_colors (list of tuples): List of BGR colors for contours.
        overlay_bboxes (bool): Whether to overlay bounding boxes.
        overlay_contours (bool): Whether to overlay contours.
        overlay_labels (bool): Whether to overlay category labels.
        color_labels (str): Color name for labels or 'match' to match contour color.
        opacity (float): Opacity for overlays.

    Returns:
        numpy.ndarray: Image with overlays.
    """
    # Create a copy of the image to draw overlays
    annotated_image = image.copy()

    # Define a color map for label colors
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'white': (255, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'black': (0, 0, 0),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128)
    }

    def get_color(color_str):
        """
        Converts a color name to a BGR tuple.

        Parameters:
            color_str (str): Name of the color.

        Returns:
            tuple: BGR color tuple.
        """
        if isinstance(color_str, str):
            return color_map.get(color_str.lower(), (255, 255, 255))  # Default to white
        elif isinstance(color_str, list) or isinstance(color_str, tuple):
            return tuple(color_str)
        else:
            return (255, 255, 255)  # Default to white

    # Determine if labels should match contour colors
    match_label_color = False
    if color_labels.lower() == 'match':
        match_label_color = True
    else:
        fixed_label_color = get_color(color_labels)

    # Iterate over each annotation with assigned color
    for idx, ann in enumerate(annotations):
        category_id = ann['category_id']
        category_name = categories.get(category_id, 'Unknown')

        contour_color = contour_colors[idx % len(contour_colors)]

        # **Overlay Bounding Boxes**
        if overlay_bboxes:
            bbox = ann.get('bbox', None)  # [x, y, width, height]
            if bbox:
                x, y, w, h = map(int, bbox)
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(annotated_image, top_left, bottom_right, contour_color, 2)

        # **Overlay Contours**
        if overlay_contours:
            segmentation = ann.get('segmentation', None)
            if segmentation:
                if isinstance(segmentation, list):
                    # Polygon format
                    for seg in segmentation:
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.polylines(annotated_image, [pts], isClosed=True, color=contour_color, thickness=2)
                else:
                    # RLE format
                    mask = maskUtils.decode(segmentation)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        cv2.polylines(annotated_image, [contour], isClosed=True, color=contour_color, thickness=2)

        # **Overlay Labels**
        if overlay_labels:
            # Position the label at the top-left corner of the bounding box
            bbox = ann.get('bbox', None)
            if bbox:
                x, y, w, h = map(int, bbox)
                label_position = (x, y - 10 if y - 10 > 10 else y + 10)
                if match_label_color:
                    label_color = contour_color
                else:
                    label_color = fixed_label_color
                cv2.putText(annotated_image, category_name, label_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, cv2.LINE_AA)

    # **Apply Transparency to Overlays**
    if overlay_bboxes or overlay_contours or overlay_labels:
        # Create an overlay image
        overlay = image.copy()
        annotated_overlay = annotated_image.copy()

        # Blend the original image and the annotated image
        cv2.addWeighted(annotated_overlay, opacity, overlay, 1 - opacity, 0, annotated_image)

    return annotated_image

def create_and_save_annotated_image(image, annotations, filename, output_dir, categories,
                                    overlay_bboxes=False, overlay_contours=False, overlay_labels=False,
                                    color_labels='white', opacity=0.4):
    """
    Creates and saves an annotated image with overlays.

    Parameters:
        image (numpy.ndarray): Original image.
        annotations (list of dict): Annotations for the image.
        filename (str): Base filename for saving.
        output_dir (str): Directory to save the annotated image.
        categories (dict): Mapping from category_id to category_name.
        overlay_bboxes (bool): Whether to overlay bounding boxes.
        overlay_contours (bool): Whether to overlay contours.
        overlay_labels (bool): Whether to overlay category labels.
        color_labels (str): Color for labels or 'match' to match contour color.
        opacity (float): Opacity for overlays.

    Returns:
        None
    """
    # Generate distinct colors for each annotation
    contour_colors = generate_distinct_colors(len(annotations))

    # Overlay annotations
    annotated_image = overlay_annotations(
        image,
        annotations,
        categories,
        contour_colors,
        overlay_bboxes=overlay_bboxes,
        overlay_contours=overlay_contours,
        overlay_labels=overlay_labels,
        color_labels=color_labels,
        opacity=opacity
    )

    # Define annotated image filename
    annotated_image_filename = f"annotated_{filename}.png"
    annotated_image_path = os.path.join(output_dir, annotated_image_filename)

    # Save annotated image
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"Saved annotated image to: {annotated_image_path}")

def main():
    """
    Main function to execute the visualization process.
    """
    args = parse_args()
    annotations_dict, images_info, categories = load_annotations(args.json, category_name=args.category_name)
    
    if not annotations_dict:
        print("No annotations found in the JSON file.")
        return

    # Prepare output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all annotations for visualization
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

        # Filter annotations by category if specified
        if args.category_name:
            category_ids = [cat_id for cat_id, name in categories.items() if name == args.category_name]
            filtered_annotations = [ann for ann in annotations if ann['category_id'] in category_ids]
        else:
            filtered_annotations = annotations

        if not filtered_annotations:
            print(f"No annotations for specified category in image {image_filename}.")
            continue

        # Save annotated image with overlays
        create_and_save_annotated_image(
            image,
            filtered_annotations,
            image_basename,
            args.output_dir,
            categories,
            overlay_bboxes=args.overlay_bboxes,
            overlay_contours=args.overlay_contours,
            overlay_labels=args.overlay_labels,
            color_labels=args.color_labels,
            opacity=args.opacity
        )

    print("Visualization of annotations completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
