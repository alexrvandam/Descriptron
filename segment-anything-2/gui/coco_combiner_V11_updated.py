import os
import sys
import json
import argparse
import logging
import glob
import cv2  # Use OpenCV for reading dimensions

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Argument parsing
    parser = argparse.ArgumentParser(description='Combine multiple COCO JSON files into one, ensuring width/height are included. Tries to append .jpg/.png/.tif if missing.')
    parser.add_argument('--input-jsons', nargs='+', required=True,
                        help='List of input COCO JSON files to combine. Can be space-separated or a CSV string.')
    parser.add_argument('--output-json', required=True, help='Output path for the combined COCO JSON file.')
    parser.add_argument('--skip-categories', nargs='*', default=[], help='List of category names to skip.')
    parser.add_argument('--images-folders', nargs='*', default=[],
                        help='One or more folders where input images are located. Used to look up file extensions if missing and for reading dimensions.')

    args = parser.parse_args()

    # Handle input JSON files (space-separated or CSV)
    input_json_files = []
    for item in args.input_jsons:
        input_json_files.extend(item.split(','))  # Split CSV strings
    input_json_files = [f.strip() for f in input_json_files if f.strip()]

    output_json_path = args.output_json
    skip_categories = args.skip_categories
    images_folders = args.images_folders

    # Check if the output filename has a '.json' extension
    if not output_json_path.lower().endswith('.json'):
        # Append '_combined.json' to the output filename (without extension)
        output_json_path = os.path.splitext(output_json_path)[0] + '_combined.json'
        logging.info(f"Output filename does not have a '.json' extension. Using '{output_json_path}' as the output file.")

    # Now proceed with combining the files
    combine_coco_json_files(input_json_files, output_json_path, skip_categories, images_folders)


def combine_coco_json_files(input_json_files, output_json_path, skip_categories=[], images_folders=[]):
    combined_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # IDs for new images and annotations
    new_image_id = 1
    new_annotation_id = 1

    # Mapping for categories: maps category name to new id
    category_name_to_new_id = {}
    new_category_id = 1  # Start category IDs from 1

    # Process each input file
    for json_file in input_json_files:
        if not os.path.exists(json_file):
            logging.error(f"File not found: {json_file}")
            continue

        logging.info(f"Processing file: {json_file}")
        with open(json_file, 'r') as file:
            coco_data = json.load(file)

        # Handle categories: map old category IDs to new ones
        old_category_id_to_new_id = {}
        for category in coco_data.get('categories', []):
            cat_name = category['name'].strip()
            if cat_name in skip_categories:
                logging.info(f"Skipping category '{cat_name}' as per skip list.")
                continue  # Skip this category

            if cat_name not in category_name_to_new_id:
                new_category = {
                    "id": new_category_id,
                    "name": cat_name,
                    "supercategory": category.get('supercategory', 'none')
                }
                # Explicitly copy keypoints and skeleton if present
                if 'keypoints' in category:
                    new_category['keypoints'] = category['keypoints']
                if 'skeleton' in category:
                    new_category['skeleton'] = category['skeleton']

                category_name_to_new_id[cat_name] = new_category_id
                combined_output['categories'].append(new_category)
                logging.info(f"Assigned new category ID {new_category_id} to category '{cat_name}'.")
                new_category_id += 1

            old_category_id_to_new_id[category['id']] = category_name_to_new_id[cat_name]

        # Local mapping for images for this file. Even if the same image ID appears in another file,
        # we always assign a new ID.
        old_image_id_to_new_id = {}
        if coco_data.get('images'):
            for image in coco_data.get('images', []):
                old_img_id = image['id']
                local_new_id = new_image_id
                new_image = image.copy()
                # Ensure file_name has an extension if possible
                if 'file_name' in new_image:
                    new_image['file_name'] = get_image_file_name(new_image['file_name'], images_folders)
                # Make sure we have width/height via OpenCV
                add_width_and_height_if_missing_cv2(new_image, images_folders)
                new_image['id'] = new_image_id
                combined_output['images'].append(new_image)
                new_image_id += 1

                old_image_id_to_new_id[old_img_id] = local_new_id

        # Handle annotations
        for annotation in coco_data.get('annotations', []):
            old_cat_id = annotation['category_id']
            new_cat_id = old_category_id_to_new_id.get(old_cat_id)
            if not new_cat_id:
                logging.info(f"Skipping annotation ID {annotation.get('id', 'unknown')} with category ID {old_cat_id} (category skipped).")
                continue  # Skip annotations with categories that were skipped

            old_img_id = annotation['image_id']
            # If the image was not already added, create a new image entry.
            if old_img_id not in old_image_id_to_new_id:
                local_new_id = new_image_id
                new_image_id += 1

                file_name = str(old_img_id)
                file_name = get_image_file_name(file_name, images_folders)

                new_image = {
                    "id": local_new_id,
                    "file_name": file_name
                }
                add_width_and_height_if_missing_cv2(new_image, images_folders)
                combined_output['images'].append(new_image)

                old_image_id_to_new_id[old_img_id] = local_new_id

            # Create the new annotation entry explicitly preserving keypoints
            new_annotation = annotation.copy()
            new_annotation['id'] = new_annotation_id
            new_annotation_id += 1

            # Update image_id and category_id to the new ones
            new_annotation['image_id'] = old_image_id_to_new_id[old_img_id]
            new_annotation['category_id'] = new_cat_id

            # Explicitly preserve keypoints fields if present
            if 'keypoints' in annotation:
                new_annotation['keypoints'] = annotation['keypoints']
            if 'num_keypoints' in annotation:
                new_annotation['num_keypoints'] = annotation['num_keypoints']

            combined_output['annotations'].append(new_annotation)

    # Debug prints
    logging.info(f"Total Combined Categories: {len(combined_output['categories'])}")
    logging.info(f"Total Combined Images: {len(combined_output['images'])}")
    logging.info(f"Total Combined Annotations: {len(combined_output['annotations'])}")

    # Save the combined output
    with open(output_json_path, 'w') as out_file:
        json.dump(combined_output, out_file, indent=4)
    logging.info(f"Combined COCO JSON saved to {output_json_path}")


def get_image_file_name(file_name, images_folders):
    """
    If file_name has an extension, return as-is.
    Otherwise:
      1) Try a wildcard search: file_name + '.*' in images_folders.
      2) If that fails, systematically try .jpg, .jpeg, .png, .tif, .tiff.
      3) Return the discovered file name or the input as a fallback.
    """
    base, ext = os.path.splitext(file_name)
    if ext:
        # Already has extension
        return file_name

    # 1) Try wildcard search: file_name + '.*'
    for folder in images_folders:
        pattern = os.path.join(folder, base + ".*")
        matches = glob.glob(pattern)
        if matches:
            found_file = os.path.basename(matches[0])
            logging.info(f"Found extension for '{file_name}' in folder '{folder}': using '{found_file}'.")
            return found_file
    # 2) If still not found, systematically try known extensions
    possible_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    for folder in images_folders:
        for e in possible_exts:
            candidate = os.path.join(folder, base + e)
            if os.path.exists(candidate):
                found_file = os.path.basename(candidate)
                logging.info(f"Found matching file for '{file_name}' in folder '{folder}': using '{found_file}'.")
                return found_file

    # Also try in current directory if images_folders fails
    for e in possible_exts:
        if os.path.exists(base + e):
            found_file = base + e
            logging.info(f"Found matching file in current dir for '{file_name}': using '{found_file}'.")
            return found_file

    # If we got here, no match
    logging.info(f"Could NOT find any file for '{file_name}' in images_folders or current dir.")
    return file_name


def add_width_and_height_if_missing_cv2(image_dict, images_folders):
    """
    Checks if 'width' or 'height' is missing from the image_dict.
    If so, tries reading with OpenCV (looping over folders + direct).
    Inserts the discovered w/h, logs success/failure.
    """
    if 'width' not in image_dict or 'height' not in image_dict:
        w, h = get_image_dimensions_cv2(image_dict['file_name'], images_folders)
        if w is not None and h is not None:
            image_dict['width'] = w
            image_dict['height'] = h
            logging.info(f"Width={w}, Height={h} for image '{image_dict['file_name']}' (ID {image_dict.get('id','?')})")
        else:
            logging.info(f"Could NOT determine width/height for image '{image_dict['file_name']}'")


def get_image_dimensions_cv2(file_name, images_folders):
    """
    Attempts to load the image using OpenCV from the provided folders or the current path,
    returning (width, height) if successful, else (None, None).
    """
    # (1) Check each images_folder
    for folder in images_folders:
        candidate = os.path.join(folder, file_name)
        if os.path.exists(candidate):
            img = cv2.imread(candidate)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    # (2) If not found in the folders, try directly
    if os.path.exists(file_name):
        img = cv2.imread(file_name)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return None, None

if __name__ == '__main__':
    main()
