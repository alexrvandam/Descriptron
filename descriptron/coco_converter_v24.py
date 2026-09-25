#!/usr/bin/env python
import os
import json
import argparse
import copy
import logging
import sys
import cv2  # Using OpenCV for image dimensions

# -------------------------
# Helper functions
# -------------------------

def load_json_file(json_file_path):
    with open(json_file_path, 'r') as f:
        try:
            data = json.load(f)
            logging.debug(f"Successfully loaded JSON file: {json_file_path}")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON file {json_file_path}: {e}")
            sys.exit(1)

def validate_json_structure(data):
    if not isinstance(data, dict):
        logging.error(f"Loaded JSON data is not a dictionary. Actual type: {type(data)}")
        sys.exit(1)
    else:
        logging.debug("JSON data is a dictionary.")

def validate_coco_structure(data):
    required_keys = ['images', 'annotations', 'categories']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        logging.error(f"Missing required top-level keys: {missing_keys}")
        return False
    logging.debug("Top-level keys are present.")
    if not isinstance(data['images'], list):
        logging.error(f"'images' field is not a list. Actual type: {type(data['images'])}")
        return False
    if not isinstance(data['annotations'], list):
        logging.error(f"'annotations' field is not a list. Actual type: {type(data['annotations'])}")
        return False
    if not isinstance(data['categories'], list):
        logging.error(f"'categories' field is not a list. Actual type: {type(data['categories'])}")
        return False
    logging.debug("COCO JSON structure is valid.")
    return True

def get_image_dimensions_opencv(image_path):
    # Use OpenCV to load the image and return (width, height)
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"OpenCV failed to load image: {image_path}")
        return 0, 0
    # cv2.imread returns image shape as (height, width, channels)
    height, width = img.shape[:2]
    return width, height

def correct_images_field(data):
    images = data['images']
    corrected_images = []
    image_id_mapping = {}
    image_id_counter = 1
    for img_entry in images:
        if isinstance(img_entry, str):
            file_name = os.path.basename(img_entry)
            full_image_path = img_entry
            if not os.path.exists(full_image_path):
                logging.error(f"Image file does not exist: {full_image_path}")
                continue
            width, height = get_image_dimensions_opencv(full_image_path)
            logging.debug(f"Dimensions for image '{full_image_path}': width={width}, height={height}")
            image_dict = {
                "id": image_id_counter,
                "file_name": file_name,
                "width": width,
                "height": height,
                "area": width * height
            }
            image_id_mapping[file_name] = image_id_counter
            corrected_images.append(image_dict)
            image_id_counter += 1
        elif isinstance(img_entry, dict):
            required_img_keys = ['id', 'file_name', 'width', 'height']
            if not all(k in img_entry for k in required_img_keys):
                logging.error(f"Image entry missing required keys: {img_entry}")
                continue
            if "area" not in img_entry:
                img_entry["area"] = img_entry["width"] * img_entry["height"]
            corrected_images.append(img_entry)
            image_id_mapping[img_entry['file_name']] = img_entry['id']
        else:
            logging.error(f"Invalid image entry type: {type(img_entry)}")
            continue
    data['images'] = corrected_images
    logging.info(f"Total images after correction: {len(corrected_images)}")
    return image_id_mapping

def correct_annotations_field(data, image_id_mapping):
    annotations = data['annotations']
    corrected_annotations = []
    for ann in annotations:
        image_id = ann['image_id']
        if isinstance(image_id, str):
            image_id_mapped = image_id_mapping.get(image_id)
            if image_id_mapped is None:
                image_id_mapped = image_id_mapping.get(os.path.basename(image_id))
            if image_id_mapped is None:
                logging.error(f"Cannot find image_id mapping for annotation: {ann}")
                continue
            ann['image_id'] = image_id_mapped
        elif isinstance(image_id, int):
            pass
        else:
            logging.error(f"Invalid image_id type in annotation: {ann}")
            continue
        corrected_annotations.append(ann)
    data['annotations'] = corrected_annotations
    logging.info(f"Total annotations after correction: {len(corrected_annotations)}")

def clean_coco_json(original_json_path, cleaned_json_path, default_exclude_categories=[], user_exclude_categories=[], img_dir=None):
    data = load_json_file(original_json_path)
    validate_json_structure(data)
    
    # --- Modified image cleanup section starts here ---
    # Build list of category IDs to exclude.
    category_ids_to_exclude = []
    for cat in data.get('categories', []):
        if cat.get('name') in (default_exclude_categories + user_exclude_categories):
            category_ids_to_exclude.append(cat.get('id'))
    
    # Filter annotations by excluding those with certain category_ids.
    cleaned_annotations = [ann for ann in data.get('annotations', []) if ann.get('category_id') not in category_ids_to_exclude]
    data['annotations'] = cleaned_annotations
    
    # Get the set of image IDs referenced in the annotations.
    image_ids_in_annotations = set(ann['image_id'] for ann in cleaned_annotations if 'image_id' in ann)
    
    # Get the existing images list from the data.
    images = data.get('images', [])
    
    # Build a dictionary of existing images keyed by their id.
    images_dict = {}
    for img in images:
        img_id = img.get('id')
        if img_id is not None:
            images_dict[img_id] = img
    
    # For every image ID referenced in annotations that is not in images_dict,
    # create a minimal image record using OpenCV to get dimensions.
    for img_id in image_ids_in_annotations:
        if img_id not in images_dict:
            file_name = str(img_id)
            # Determine the full path using the provided img_dir if available.
            if not os.path.isabs(file_name):
                if img_dir:
                    full_path = os.path.join(img_dir, file_name)
                else:
                    full_path = os.path.join(os.getcwd(), file_name)
            else:
                full_path = file_name
            width, height = get_image_dimensions_opencv(full_path)
            logging.debug(f"Dimensions for missing image '{full_path}': width={width}, height={height}")
            images_dict[img_id] = {
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "area": width * height
            }
    
    # For each image record, if width, height, or area is missing or zero,
    # try to update using OpenCV and log the found dimensions.
    for img in images_dict.values():
        if not img.get("width") or not img.get("height"):
            file_path = img.get("file_name")
            if not os.path.isabs(file_path):
                if img_dir:
                    file_path = os.path.join(img_dir, file_path)
                else:
                    file_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(file_path):
                w, h = get_image_dimensions_opencv(file_path)
                logging.debug(f"Updating image '{file_path}': found width={w}, height={h}")
                img["width"] = w
                img["height"] = h
                img["area"] = w * h
            else:
                logging.warning(f"Image file {file_path} not found.")
        if "area" not in img or img["area"] == 0:
            img["area"] = img["width"] * img["height"]
            logging.debug(f"Set area for image '{img.get('file_name')}': {img['area']}")
    
    cleaned_images = list(images_dict.values())
    
    # Filter categories as before.
    cleaned_categories = [cat for cat in data.get('categories', []) if cat.get('id') not in category_ids_to_exclude]
    
    # Update the data with the new images, annotations, and categories lists.
    data['images'] = cleaned_images
    data['categories'] = cleaned_categories
    logging.info(f"Total images after cleanup: {len(cleaned_images)}")
    # --- Modified image cleanup section ends here ---
    
    with open(cleaned_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Cleaned JSON saved to {cleaned_json_path}")
    logging.info(f"Excluded Categories: {default_exclude_categories + user_exclude_categories}")
    logging.info(f"Number of Images after Cleanup: {len(cleaned_images)}")
    logging.info(f"Number of Annotations after Cleanup: {len(cleaned_annotations)}")
    logging.info(f"Number of Categories after Cleanup: {len(cleaned_categories)}")
    return cleaned_json_path

def remove_duplicate_annotations(data):
    category_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    unique_annotations = []
    seen = set()
    for ann in data.get('annotations', []):
        if "keypoints" in ann and ann["keypoints"] and sum(ann["keypoints"]) > 0:
            key = ("keypoints", ann["image_id"], tuple(ann["keypoints"]))
        else:
            seg = ann.get('segmentation', [])
            if isinstance(seg, list):
                if all(isinstance(s, list) for s in seg):
                    flat_seg = tuple(tuple(s) for s in seg)
                else:
                    flat_seg = tuple(seg)
                key = (category_id_to_name.get(ann['category_id']), ann["image_id"], flat_seg)
            else:
                key = (category_id_to_name.get(ann['category_id']), ann["image_id"])
        if key not in seen:
            seen.add(key)
            unique_annotations.append(ann)
        else:
            logging.info(f"Duplicate annotation removed: Annotation ID {ann.get('id')}")
    data['annotations'] = unique_annotations
    logging.info(f"After duplicate removal, total annotations: {len(unique_annotations)}")
    return data

def extract_keypoint_annotations(data):
    new_keypoint_annos = []
    remaining_annos = []
    existing_cat_ids = [cat['id'] for cat in data.get('categories', [])]
    new_kp_cat_id = max(existing_cat_ids) + 1 if existing_cat_ids else 1
    if not any(cat['name'].lower() == "keypoints" for cat in data.get("categories", [])):
        new_category = {"id": new_kp_cat_id, "name": "keypoints", "supercategory": "keypoints"}
        data["categories"].append(new_category)
        logging.info(f"Added new keypoints category: id {new_kp_cat_id}, name 'keypoints'")
    else:
        new_kp_cat_id = [cat['id'] for cat in data["categories"] if cat['name'].lower() == "keypoints"][0]
    existing_ann_ids = [ann['id'] for ann in data.get("annotations", [])]
    next_ann_id = max(existing_ann_ids) + 1 if existing_ann_ids else 1

    for ann in data.get("annotations", []):
        if "keypoints" in ann and ann["keypoints"] and sum(ann["keypoints"]) > 0:
            kp_ann = copy.deepcopy(ann)
            kp_ann["id"] = next_ann_id
            next_ann_id += 1
            kp_ann["category_id"] = new_kp_cat_id
            if (not kp_ann.get("bbox")) or (len(kp_ann.get("bbox", [])) == 0) or (sum(kp_ann.get("bbox", [])) == 0):
                kps = kp_ann["keypoints"]
                xs = kps[0::3]
                ys = kps[1::3]
                if xs and ys:
                    x_min = min(xs)
                    x_max = max(xs)
                    y_min = min(ys)
                    y_max = max(ys)
                    kp_ann["bbox"] = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    kp_ann["bbox"] = [0, 0, 0, 0]
            if kp_ann.get("area", 0) == 0:
                w = kp_ann["bbox"][2]
                h = kp_ann["bbox"][3]
                kp_ann["area"] = w * h
            new_keypoint_annos.append(kp_ann)
            ann["keypoints"] = []
            ann["num_keypoints"] = 0
        remaining_annos.append(ann)
    data["annotations"] = remaining_annos + new_keypoint_annos
    logging.info(f"After reassigning keypoint annotations, total annotations: {len(data['annotations'])}")
    return data

def remove_annotations_with_empty_bbox(data):
    valid_annotations = []
    removed_count = 0
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox", [])
        if not bbox or sum(bbox) == 0:
            removed_count += 1
            continue
        valid_annotations.append(ann)
    data["annotations"] = valid_annotations
    logging.info(f"After removing annotations with empty bbox, total annotations: {len(valid_annotations)}; removed {removed_count}")
    return data

def save_json_files(data, output_dir, output_base_name):
    segmentation_path = os.path.join(output_dir, f"{output_base_name}_segmentation.json")
    segmentation_compact_path = os.path.join(output_dir, f"{output_base_name}_segmentation_compact.json")
    segmentation_jsonl_path = os.path.join(output_dir, f"{output_base_name}_segmentation.jsonl")

    keypoints_path = os.path.join(output_dir, f"{output_base_name}_keypoints.json")
    keypoints_compact_path = os.path.join(output_dir, f"{output_base_name}_keypoints_compact.json")
    keypoints_jsonl_path = os.path.join(output_dir, f"{output_base_name}_keypoints.jsonl")

    combined_path = os.path.join(output_dir, f"{output_base_name}_combined.json")
    combined_compact_path = os.path.join(output_dir, f"{output_base_name}_combined_compact.json")
    combined_jsonl_path = os.path.join(output_dir, f"{output_base_name}_combined.jsonl")

    kp_cat_id = [cat['id'] for cat in data["categories"] if cat['name'].lower() == "keypoints"][0]
    seg_annotations = [ann for ann in data.get("annotations", []) if not (ann.get("keypoints") and sum(ann.get("keypoints", [])) > 0 and ann.get("category_id") == kp_cat_id)]
    segmentation_data = {"images": data.get("images"), "annotations": seg_annotations, "categories": data.get("categories")}
    with open(segmentation_path, 'w') as seg_file:
        json.dump(segmentation_data, seg_file, indent=4)
    logging.info(f"Segmentation (pretty) JSON saved to {segmentation_path}")

    with open(segmentation_compact_path, 'w') as seg_compact_file:
        json.dump(segmentation_data, seg_compact_file, separators=(',',':'))
    logging.info(f"Segmentation (compact) JSON saved to {segmentation_compact_path}")

    with open(segmentation_jsonl_path, 'w') as seg_jsonl_file:
        for ann in seg_annotations:
            seg_jsonl_file.write(json.dumps(ann) + "\n")
    logging.info(f"Segmentation JSONL saved to {segmentation_jsonl_path}")

    keypoints_annotations = [ann for ann in data.get("annotations", []) if ann.get("keypoints") and sum(ann.get("keypoints", [])) > 0 and ann.get("category_id") == kp_cat_id]
    keypoints_data = {"images": data.get("images"), "annotations": keypoints_annotations, "categories": data.get("categories")}
    with open(keypoints_path, 'w') as kp_file:
        json.dump(keypoints_data, kp_file, indent=4)
    logging.info(f"Keypoints (pretty) JSON saved to {keypoints_path}")

    with open(keypoints_compact_path, 'w') as kp_compact_file:
        json.dump(keypoints_data, kp_compact_file, separators=(',',':'))
    logging.info(f"Keypoints (compact) JSON saved to {keypoints_compact_path}")

    with open(keypoints_jsonl_path, 'w') as kp_jsonl_file:
        for ann in keypoints_annotations:
            kp_jsonl_file.write(json.dumps(ann) + "\n")
    logging.info(f"Keypoints JSONL saved to {keypoints_jsonl_path}")

    with open(combined_path, 'w') as comb_file:
        json.dump(data, comb_file, indent=4)
    logging.info(f"Combined (pretty) JSON saved to {combined_path}")

    with open(combined_compact_path, 'w') as comb_compact_file:
        json.dump(data, comb_compact_file, separators=(',',':'))
    logging.info(f"Combined (compact) JSON saved to {combined_compact_path}")

    with open(combined_jsonl_path, 'w') as comb_jsonl_file:
        for ann in data.get("annotations", []):
            comb_jsonl_file.write(json.dumps(ann) + "\n")
    logging.info(f"Combined JSONL saved to {combined_jsonl_path}")

def save_simple_compact(data, output_dir, output_base_name):
    simple_compact_path = os.path.join(output_dir, f"{output_base_name}_simple_compact.json")
    with open(simple_compact_path, 'w') as f:
        json.dump(data, f, separators=(',',':'))
    logging.info(f"Simple compact JSON saved to {simple_compact_path}")

def convert_coco_format(input_path, output_dir, output_base_name):
    data = load_json_file(input_path)
    data = extract_keypoint_annotations(data)
    data = remove_annotations_with_empty_bbox(data)
    save_json_files(data, output_dir, output_base_name)
    save_simple_compact(data, output_dir, output_base_name)

def main():
    parser = argparse.ArgumentParser(description='COCO JSON Converter with Cleanup, Deduplication, Keypoint Extraction, and Final Validation')
    parser.add_argument('--input-json', required=True, help='Path to the original COCO JSON file.')
    parser.add_argument('--output-dir', required=True, help='Directory where the output files will be saved.')
    parser.add_argument('--exclude-categories', nargs='*', default=[], help='List of category names to exclude.')
    parser.add_argument('--output-base-name', default='converted_output', help='Base name for the output files.')
    parser.add_argument('--reassign-image-ids', action='store_true',
                        help='Reassign integer image IDs so that annotations match them.')
    # NEW: Optional image directory for looking up image files.
    parser.add_argument('--img-dir', default=None, help='Optional directory containing images (used when image file paths are relative).')
    args = parser.parse_args()

    input_path = args.input_json
    output_dir = args.output_dir
    output_base_name = args.output_base_name
    img_dir = args.img_dir

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    default_exclude_categories = ["Trash", "Select Label", "Custom"]
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    cleaned_json_filename = f"{base_name}_cleaned.json"
    cleaned_json_path = os.path.join(output_dir, cleaned_json_filename)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            sys.exit(1)

    if args.reassign_image_ids:
        pass

    try:
        clean_coco_json(
            original_json_path=input_path,
            cleaned_json_path=cleaned_json_path,
            default_exclude_categories=default_exclude_categories,
            user_exclude_categories=args.exclude_categories,
            img_dir=img_dir
        )
    except Exception as e:
        logger.error(f"Error during COCO JSON cleanup: {e}")
        sys.exit(1)

    try:
        cleaned_data = load_json_file(cleaned_json_path)
        if not validate_coco_structure(cleaned_data):
            logger.error("Cleaned COCO JSON failed basic structure validation.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during validation of cleaned JSON: {e}")
        sys.exit(1)

    try:
        cleaned_data = remove_duplicate_annotations(cleaned_data)
        with open(cleaned_json_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)
        logging.info(f"Deduplicated JSON saved to {cleaned_json_path}")
    except Exception as e:
        logger.error(f"Error during removing duplicate annotations: {e}")
        sys.exit(1)

    try:
        convert_coco_format(cleaned_json_path, output_dir, output_base_name)
    except Exception as e:
        logger.error(f"Error during COCO conversion: {e}")
        sys.exit(1)

    logging.info("COCO JSON conversion completed successfully.")

if __name__ == '__main__':
    main()
