import os
import json
import argparse
import copy
from PIL import Image
import logging
import sys

def load_json_file(json_file_path):
    """
    Loads a JSON file and returns the data as a dictionary.
    """
    with open(json_file_path, 'r') as f:
        try:
            data = json.load(f)
            logging.debug(f"Successfully loaded JSON file: {json_file_path}")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON file {json_file_path}: {e}")
            sys.exit(1)

def validate_json_structure(data):
    """
    Validates that the loaded JSON data is a dictionary.
    """
    if not isinstance(data, dict):
        logging.error(f"Loaded JSON data is not a dictionary. Actual type: {type(data)}")
        sys.exit(1)
    else:
        logging.debug("JSON data is a dictionary.")

def validate_coco_structure(data):
    """
    Validates the structure of a COCO JSON file.
    """
    # Check for required top-level keys
    required_keys = ['images', 'annotations', 'categories']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        logging.error(f"Missing required top-level keys: {missing_keys}")
        return False

    logging.debug("Top-level keys are present.")

    # Validate 'images' field
    if not isinstance(data['images'], list):
        logging.error(f"'images' field is not a list. Actual type: {type(data['images'])}")
        return False

    # Validate 'annotations' field
    if not isinstance(data['annotations'], list):
        logging.error(f"'annotations' field is not a list. Actual type: {type(data['annotations'])}")
        return False

    # Validate 'categories' field
    if not isinstance(data['categories'], list):
        logging.error(f"'categories' field is not a list. Actual type: {type(data['categories'])}")
        return False

    logging.debug("COCO JSON structure is valid.")
    return True

def correct_images_field(data):
    """
    Corrects the 'images' field if it contains file paths instead of dictionaries.
    Assigns integer IDs and collects image metadata.
    """
    images = data['images']
    corrected_images = []
    image_id_mapping = {}
    image_id_counter = 1

    for img_entry in images:
        if isinstance(img_entry, str):
            # img_entry is a file path
            file_name = os.path.basename(img_entry)
            full_image_path = img_entry

            # Check if the image file exists
            if not os.path.exists(full_image_path):
                logging.error(f"Image file does not exist: {full_image_path}")
                continue

            # Get image dimensions
            try:
                with Image.open(full_image_path) as img:
                    width, height = img.size
            except Exception as e:
                logging.error(f"Error opening image {full_image_path}: {e}")
                continue

            # Create image dictionary
            image_dict = {
                "id": image_id_counter,
                "file_name": file_name,
                "width": width,
                "height": height
            }
            image_id_mapping[img_entry] = image_id_counter
            image_id_mapping[file_name] = image_id_counter  # Map file_name as well
            corrected_images.append(image_dict)
            logging.debug(f"Added image: {image_dict}")
            image_id_counter += 1
        elif isinstance(img_entry, dict):
            # img_entry is already a dictionary, check for required keys
            required_img_keys = ['id', 'file_name', 'width', 'height']
            if not all(k in img_entry for k in required_img_keys):
                logging.error(f"Image entry missing required keys: {img_entry}")
                continue
            corrected_images.append(img_entry)
            image_id_mapping[img_entry['file_name']] = img_entry['id']
        else:
            logging.error(f"Invalid image entry type: {type(img_entry)}")
            continue

    data['images'] = corrected_images
    logging.info(f"Total images after correction: {len(corrected_images)}")
    return image_id_mapping

def correct_annotations_field(data, image_id_mapping):
    """
    Corrects the 'image_id' in annotations to be integer IDs.
    """
    annotations = data['annotations']
    corrected_annotations = []
    for ann in annotations:
        image_id = ann['image_id']
        if isinstance(image_id, str):
            # image_id is a file path or file name
            image_id_mapped = image_id_mapping.get(image_id)
            if image_id_mapped is None:
                # Try using basename
                image_id_mapped = image_id_mapping.get(os.path.basename(image_id))
            if image_id_mapped is None:
                logging.error(f"Cannot find image_id mapping for annotation: {ann}")
                continue
            ann['image_id'] = image_id_mapped
        elif isinstance(image_id, int):
            # image_id is already an integer
            pass
        else:
            logging.error(f"Invalid image_id type in annotation: {ann}")
            continue

        corrected_annotations.append(ann)

    data['annotations'] = corrected_annotations
    logging.info(f"Total annotations after correction: {len(corrected_annotations)}")

def clean_coco_json(original_json_path, cleaned_json_path, default_exclude_categories=[], user_exclude_categories=[]):
    """
    Cleans the COCO JSON by excluding specified categories and correcting fields.
    """
    # Load the original JSON
    data = load_json_file(original_json_path)
    validate_json_structure(data)

    # Correct the 'images' field
    image_id_mapping = correct_images_field(data)

    # Correct the 'annotations' field
    correct_annotations_field(data, image_id_mapping)

    # Combine default and user-specified categories to exclude
    categories_to_exclude = set(default_exclude_categories + user_exclude_categories)

    # Create a mapping from category names to IDs
    category_name_to_id = {cat['name']: cat['id'] for cat in data.get('categories', [])}

    # Determine which category IDs to exclude
    category_ids_to_exclude = []
    for cat_name in categories_to_exclude:
        if cat_name in category_name_to_id:
            category_ids_to_exclude.append(category_name_to_id[cat_name])
        else:
            logging.warning(f"Category '{cat_name}' not found in JSON categories.")

    # Filter out annotations with categories to exclude
    cleaned_annotations = [ann for ann in data.get('annotations', []) if ann.get('category_id') not in category_ids_to_exclude]

    # Find image IDs that are referenced by the remaining annotations
    image_ids_in_annotations = set(ann['image_id'] for ann in cleaned_annotations if 'image_id' in ann)

    # Filter images to include only those that are referenced by annotations
    cleaned_images = [img for img in data.get('images', []) if img.get('id') in image_ids_in_annotations]

    # Filter categories to exclude unwanted categories
    cleaned_categories = [cat for cat in data.get('categories', []) if cat.get('id') not in category_ids_to_exclude]

    # Update the data dictionary
    data['images'] = cleaned_images
    data['annotations'] = cleaned_annotations
    data['categories'] = cleaned_categories

    # Save the cleaned JSON
    with open(cleaned_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    logging.info(f"Cleaned JSON saved to {cleaned_json_path}")
    logging.info(f"Excluded Categories: {categories_to_exclude}")
    logging.info(f"Number of Images after Cleanup: {len(cleaned_images)}")
    logging.info(f"Number of Annotations after Cleanup: {len(cleaned_annotations)}")
    logging.info(f"Number of Categories after Cleanup: {len(cleaned_categories)}")

def validate_coco_json(data):
    """
    Validates the structure and content of the COCO JSON file.
    """
    # Validate basic structure
    if not validate_coco_structure(data):
        return False

    # Validate images
    for img in data['images']:
        required_img_keys = ['id', 'file_name', 'width', 'height']
        if not all(k in img for k in required_img_keys):
            logging.error(f"Image missing required keys: {img}")
            return False

    # Validate annotations
    for ann in data['annotations']:
        required_ann_keys = ['id', 'image_id', 'category_id', 'bbox', 'iscrowd']
        if not all(k in ann for k in required_ann_keys):
            logging.error(f"Annotation missing required keys: {ann}")
            return False

    # Validate categories
    for cat in data['categories']:
        required_cat_keys = ['id', 'name', 'supercategory']
        if not all(k in cat for k in required_cat_keys):
            logging.error(f"Category missing required keys: {cat}")
            return False

    # Cross-reference IDs
    image_ids = {img['id'] for img in data['images']}
    category_ids = {cat['id'] for cat in data['categories']}

    for ann in data['annotations']:
        if ann['image_id'] not in image_ids:
            logging.error(f"Annotation references missing image_id: {ann['image_id']}")
            return False
        if ann['category_id'] not in category_ids:
            logging.error(f"Annotation references missing category_id: {ann['category_id']}")
            return False

    logging.info("COCO JSON validation passed.")
    return True

def remove_duplicate_annotations(data):
    """
    Removes duplicate annotations based on category name and exact segmentation contour.
    """
    # Create a mapping from category_id to category_name
    category_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # Use a set to track unique annotations
    unique_annotations = []
    seen = set()

    for ann in data.get('annotations', []):
        category_name = category_id_to_name.get(ann['category_id'], None)
        if category_name is None:
            logging.warning(f"Category ID {ann['category_id']} not found in categories.")
            continue  # Skip if category_id not found

        # Get the segmentation data
        segmentation = ann.get('segmentation', [])

        # Convert segmentation to a hashable format
        if isinstance(segmentation, list):
            if all(isinstance(seg, list) for seg in segmentation):
                # Segmentation is a list of polygons
                flattened_segmentation = tuple(tuple(seg) for seg in segmentation)
            elif all(isinstance(seg, (int, float)) for seg in segmentation):
                # Segmentation is a flat list (e.g., RLE)
                flattened_segmentation = tuple(segmentation)
            else:
                logging.warning(f"Unsupported segmentation format in annotation ID {ann['id']}.")
                continue
        else:
            logging.warning(f"Unsupported segmentation type in annotation ID {ann['id']}.")
            continue

        # Create a unique key based on category name and segmentation
        key = (category_name, flattened_segmentation)

        if key not in seen:
            seen.add(key)
            unique_annotations.append(ann)
        else:
            logging.info(f"Duplicate annotation found and removed: Annotation ID {ann['id']}")

    # Replace annotations with the unique ones
    data['annotations'] = unique_annotations
    logging.info(f"Duplicate annotations removed. Total annotations now: {len(unique_annotations)}")

def save_json_files(data, output_dir, output_base_name):
    """
    Saves the data in both pretty-printed and compact JSON formats.
    Also saves annotations as JSONL files.
    """
    # Construct full output paths
    segmentation_path = os.path.join(output_dir, f"{output_base_name}_segmentation.json")
    segmentation_compact_path = os.path.join(output_dir, f"{output_base_name}_segmentation_compact.json")
    segmentation_jsonl_path = os.path.join(output_dir, f"{output_base_name}_segmentation.jsonl")

    keypoints_path = os.path.join(output_dir, f"{output_base_name}_keypoints.json")
    keypoints_compact_path = os.path.join(output_dir, f"{output_base_name}_keypoints_compact.json")
    keypoints_jsonl_path = os.path.join(output_dir, f"{output_base_name}_keypoints.jsonl")

    combined_path = os.path.join(output_dir, f"{output_base_name}_combined.json")
    combined_compact_path = os.path.join(output_dir, f"{output_base_name}_combined_compact.json")
    combined_jsonl_path = os.path.join(output_dir, f"{output_base_name}_combined.jsonl")

    # Save segmentation JSON (pretty-printed)
    with open(segmentation_path, 'w') as seg_file:
        json.dump(data['segmentation'], seg_file, indent=4)
    logging.info(f"Segmentation JSON saved to {segmentation_path}")

    # Save segmentation JSON (compact)
    with open(segmentation_compact_path, 'w') as seg_file_compact:
        json.dump(data['segmentation'], seg_file_compact, separators=(',', ':'))
    logging.info(f"Segmentation compact JSON saved to {segmentation_compact_path}")

    # Save segmentation annotations as JSONL
    with open(segmentation_jsonl_path, 'w') as seg_jsonl_file:
        for ann in data['segmentation']['annotations']:
            json_line = json.dumps(ann)
            seg_jsonl_file.write(json_line + '\n')
    logging.info(f"Segmentation JSONL saved to {segmentation_jsonl_path}")

    # Save keypoints JSON (pretty-printed)
    with open(keypoints_path, 'w') as kp_file:
        json.dump(data['keypoints'], kp_file, indent=4)
    logging.info(f"Keypoints JSON saved to {keypoints_path}")

    # Save keypoints JSON (compact)
    with open(keypoints_compact_path, 'w') as kp_file_compact:
        json.dump(data['keypoints'], kp_file_compact, separators=(',', ':'))
    logging.info(f"Keypoints compact JSON saved to {keypoints_compact_path}")

    # Save keypoints annotations as JSONL
    with open(keypoints_jsonl_path, 'w') as kp_jsonl_file:
        for ann in data['keypoints']['annotations']:
            json_line = json.dumps(ann)
            kp_jsonl_file.write(json_line + '\n')
    logging.info(f"Keypoints JSONL saved to {keypoints_jsonl_path}")

    # Save combined JSON (pretty-printed)
    with open(combined_path, 'w') as comb_file:
        json.dump(data['combined'], comb_file, indent=4)
    logging.info(f"Combined JSON saved to {combined_path}")

    # Save combined JSON (compact)
    with open(combined_compact_path, 'w') as comb_file_compact:
        json.dump(data['combined'], comb_file_compact, separators=(',', ':'))
    logging.info(f"Combined compact JSON saved to {combined_compact_path}")

    # Save combined annotations as JSONL
    with open(combined_jsonl_path, 'w') as comb_jsonl_file:
        for ann in data['combined']['annotations']:
            json_line = json.dumps(ann)
            comb_jsonl_file.write(json_line + '\n')
    logging.info(f"Combined JSONL saved to {combined_jsonl_path}")

    print("Conversion complete. Files saved as:")
    print(f"{segmentation_path} (segmentation pretty-printed JSON)")
    print(f"{segmentation_compact_path} (segmentation compact JSON)")
    print(f"{segmentation_jsonl_path} (segmentation annotations JSONL)")
    print(f"{keypoints_path} (keypoints pretty-printed JSON)")
    print(f"{keypoints_compact_path} (keypoints compact JSON)")
    print(f"{keypoints_jsonl_path} (keypoints annotations JSONL)")
    print(f"{combined_path} (combined pretty-printed JSON)")
    print(f"{combined_compact_path} (combined compact JSON)")
    print(f"{combined_jsonl_path} (combined annotations JSONL)")

def convert_coco_format(input_path, output_dir, output_base_name):
    """
    Converts the cleaned COCO JSON into segmentation and keypoints JSON files.
    """
    # Load cleaned JSON
    data = load_json_file(input_path)

    # Prepare base structures for segmentation and keypoints files
    segmentation_data = {
        "images": data['images'],
        "annotations": [],
        "categories": data['categories']
    }

    keypoints_data = {
        "images": data['images'],
        "annotations": [],
        "categories": data['categories']
    }

    # Process annotations
    for annotation in data.get('annotations', []):
        annotation_copy = copy.deepcopy(annotation)

        # Calculate the area if not present
        if 'area' not in annotation_copy:
            bbox = annotation_copy.get('bbox', [0, 0, 0, 0])
            annotation_copy['area'] = bbox[2] * bbox[3]
        # Set iscrowd to 0 if not present
        if 'iscrowd' not in annotation_copy:
            annotation_copy['iscrowd'] = 0

        # Separate annotations with keypoints and without
        if 'keypoints' in annotation_copy:
            keypoints_annotation = {
                "id": annotation_copy['id'],
                "image_id": annotation_copy['image_id'],
                "category_id": annotation_copy['category_id'],
                "keypoints": annotation_copy['keypoints'],
                "num_keypoints": annotation_copy.get("num_keypoints", 0),
                "iscrowd": annotation_copy['iscrowd'],
                "area": annotation_copy['area'],
                "bbox": annotation_copy['bbox']
            }
            keypoints_data['annotations'].append(keypoints_annotation)
            # Remove keypoints from the segmentation file
            annotation_copy.pop('keypoints', None)
            annotation_copy.pop('num_keypoints', None)

        # Add modified annotation to segmentation
        segmentation_data['annotations'].append(annotation_copy)

    # Merge the annotations from both segmentation and keypoints for combined output
    combined_data = {
        "images": segmentation_data['images'],
        "annotations": segmentation_data['annotations'] + keypoints_data['annotations'],
        "categories": segmentation_data['categories']
    }

    # Prepare data for saving
    output_data = {
        'segmentation': segmentation_data,
        'keypoints': keypoints_data,
        'combined': combined_data
    }

    # Save JSON files
    try:
        save_json_files(output_data, output_dir, output_base_name)
    except Exception as e:
        logging.error(f"Error saving JSON files: {e}")
        sys.exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='COCO JSON Converter with Cleanup and Deduplication')

    parser.add_argument('--input-json', required=True, help='Path to the original COCO JSON file.')
    parser.add_argument('--output-dir', required=True, help='Directory where the output files will be saved.')
    parser.add_argument('--exclude-categories', nargs='*', default=[], help='List of category names to exclude.')
    parser.add_argument('--output-base-name', default='converted_output', help='Base name for the output files.')

    args = parser.parse_args()

    input_path = args.input_json
    output_dir = args.output_dir
    user_exclude_categories = args.exclude_categories
    output_base_name = args.output_base_name

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to see all messages
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Define default categories to exclude
    default_exclude_categories = ["Trash", "Select Label", "Custom"]

    # Define path for the cleaned JSON
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    cleaned_json_filename = f"{base_name}_cleaned.json"
    cleaned_json_path = os.path.join(output_dir, cleaned_json_filename)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            sys.exit(1)

    # Perform the cleanup
    try:
        clean_coco_json(
            original_json_path=input_path,
            cleaned_json_path=cleaned_json_path,
            default_exclude_categories=default_exclude_categories,
            user_exclude_categories=user_exclude_categories
        )
    except Exception as e:
        logger.error(f"Error during COCO JSON cleanup: {e}")
        sys.exit(1)

    # Validate the cleaned JSON
    try:
        cleaned_data = load_json_file(cleaned_json_path)
        if not validate_coco_json(cleaned_data):
            logger.error("Cleaned COCO JSON failed validation.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during validation of cleaned JSON: {e}")
        sys.exit(1)

    # Remove duplicate annotations
    try:
        remove_duplicate_annotations(cleaned_data)
        # Save the deduplicated data back to the cleaned JSON path
        with open(cleaned_json_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)
        logging.info(f"Deduplicated JSON saved to {cleaned_json_path}")
    except Exception as e:
        logger.error(f"Error during removing duplicate annotations: {e}")
        sys.exit(1)

    # Proceed with the conversion using the cleaned and deduplicated JSON
    try:
        convert_coco_format(
            input_path=cleaned_json_path,
            output_dir=output_dir,
            output_base_name=output_base_name
        )
    except Exception as e:
        logger.error(f"Error during COCO conversion: {e}")
        sys.exit(1)

    logging.info("COCO JSON conversion completed successfully.")

if __name__ == '__main__':
    main()
