import os
import sys
import json
import argparse
import logging

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
    parser = argparse.ArgumentParser(description='Combine multiple COCO JSON files into one.')
    parser.add_argument('--input-jsons', nargs='+', required=True, help='List of input COCO JSON files to combine. Can be space-separated or a CSV string.')
    parser.add_argument('--output-json', required=True, help='Output path for the combined COCO JSON file.')
    parser.add_argument('--skip-categories', nargs='*', default=[], help='List of category names to skip.')
    
    args = parser.parse_args()
    
    # Handle input JSON files (space-separated or CSV)
    input_json_files = []
    for item in args.input_jsons:
        input_json_files.extend(item.split(','))  # Split CSV strings
    input_json_files = [f.strip() for f in input_json_files if f.strip()]  # Remove empty strings and strip whitespace

    output_json_path = args.output_json
    skip_categories = args.skip_categories

    # Check if the output filename has a '.json' extension
    if not output_json_path.lower().endswith('.json'):
        # Append '_combined.json' to the output filename (without extension)
        output_json_path = os.path.splitext(output_json_path)[0] + '_combined.json'
        logging.info(f"Output filename does not have a '.json' extension. Using '{output_json_path}' as the output file.")

    # Now proceed with combining the files
    combine_coco_json_files(input_json_files, output_json_path, skip_categories)

def combine_coco_json_files(input_json_files, output_json_path, skip_categories=[]):
    combined_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # IDs for new images and annotations
    new_image_id = 1
    new_annotation_id = 1
    category_name_to_new_id = {}
    new_category_id = 1  # Start category IDs from 1

    # Now process each input file
    for json_file in input_json_files:
        if not os.path.exists(json_file):
            logging.error(f"File not found: {json_file}")
            continue

        logging.info(f"Processing file: {json_file}")
        with open(json_file, 'r') as file:
            coco_data = json.load(file)
        
        # Handle categories
        # Map old category IDs to new ones
        old_category_id_to_new_id = {}
        for category in coco_data.get('categories', []):
            cat_name = category['name'].strip()
            if cat_name in skip_categories:
                logging.info(f"Skipping category '{cat_name}' as per skip list.")
                continue  # Skip this category
            if cat_name not in category_name_to_new_id:
                # Assign a new ID
                category_name_to_new_id[cat_name] = new_category_id
                new_category = {
                    "id": new_category_id,
                    "name": cat_name,
                    "supercategory": category.get('supercategory', 'none')
                }
                combined_output['categories'].append(new_category)
                logging.info(f"Assigned new category ID {new_category_id} to category '{cat_name}'.")
                new_category_id += 1
            old_category_id_to_new_id[category['id']] = category_name_to_new_id[cat_name]
        
        # Handle images
        old_image_id_to_new_id = {}
        for image in coco_data.get('images', []):
            new_image = image.copy()
            # Map old image ID to new image ID
            old_image_id = image['id']
            old_image_id_to_new_id[old_image_id] = new_image_id
            new_image['id'] = new_image_id
            combined_output['images'].append(new_image)
            new_image_id += 1
        
        # Handle annotations
        for annotation in coco_data.get('annotations', []):
            # Get the old category ID
            old_cat_id = annotation['category_id']
            # Map to new category ID
            new_cat_id = old_category_id_to_new_id.get(old_cat_id)
            if not new_cat_id:
                logging.info(f"Skipping annotation ID {annotation['id']} with category ID {old_cat_id} (category skipped).")
                continue  # Skip annotations with categories that were skipped
            # Assign new annotation ID
            new_annotation = annotation.copy()
            new_annotation['id'] = new_annotation_id
            new_annotation_id +=1
            # Update image_id
            old_image_id = annotation['image_id']
            new_image_id_mapped = old_image_id_to_new_id.get(old_image_id)
            if not new_image_id_mapped:
                logging.warning(f"Annotation ID {annotation['id']} references image ID {old_image_id} which was not found.")
                continue  # Skip annotations with images that were skipped
            new_annotation['image_id'] = new_image_id_mapped
            # Update category_id
            new_annotation['category_id'] = new_cat_id
            combined_output['annotations'].append(new_annotation)
    
    # Optional: Validate the combined JSON (implement validate_coco_json if needed)
    # if not validate_coco_json(combined_output):
    #     logging.error("Combined COCO JSON failed validation.")
    #     sys.exit(1)
    
    # Debug print to check the combined output
    logging.info(f"Total Combined Categories: {len(combined_output['categories'])}")
    logging.info(f"Total Combined Images: {len(combined_output['images'])}")
    logging.info(f"Total Combined Annotations: {len(combined_output['annotations'])}")

    # Save the combined output
    with open(output_json_path, 'w') as out_file:
        json.dump(combined_output, out_file, indent=4)
    logging.info(f"Combined COCO JSON saved to {output_json_path}")

if __name__ == '__main__':
    main()
