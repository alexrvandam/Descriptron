import argparse
import json
import os
import logging
from collections import defaultdict
import numpy as np
import re
import csv

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate species description from data.")
    parser.add_argument('--measurements', type=str, required=True, help='Path to measurements JSONL file.')
    parser.add_argument('--colors', type=str, required=True, help='Path to colors JSONL file.')
    parser.add_argument('--qa_folder', type=str, required=True, help='Path to folder containing QA JSON files.')
    parser.add_argument('--contours', type=str, required=True, help='Path to contours COCO JSON file.')
    parser.add_argument('--body_part', type=str, default='wing', help='Body part to describe (default: wing).')
    parser.add_argument('--species_list', type=str, help='Path to text file containing species names.')
    parser.add_argument('--category', type=str, default='all', help='Category to include (default: all).')
    parser.add_argument('--category_list', type=str, required=True, help='Path to text file containing category names.')
    parser.add_argument('--material_examined', type=str, required=True, help='Path to material examined CSV/TSV file.')
    parser.add_argument('--output', type=str, required=True, help='Path to output file to write the generated prompts (e.g., output.txt).')
    return parser.parse_args()

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON on line {line_number} in {file_path}: {e}")
        logging.debug(f"Loaded {len(data)} entries from {file_path}")
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}\n{e}")
    return data

def load_coco_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            logging.debug(f"Successfully loaded COCO JSON from {file_path}")
            return coco_data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding COCO JSON file: {file_path}\n{e}")
        return {}
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}\n{e}")
        return {}

def load_qa_folder(folder_path):
    qa_data = {}
    if not os.path.isdir(folder_path):
        logging.error(f"QA folder not found: {folder_path}")
        return qa_data

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    qa_entry = json.load(f)
                    species_name = extract_species_name_from_filename(filename)
                    logging.debug(f"Loaded QA data for species: {species_name} from file: {filename}")
                    qa_data[species_name] = qa_entry
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON file: {filepath}\n{e}")
            except Exception as e:
                logging.error(f"Error reading QA file: {filepath}\n{e}")
    return qa_data

def load_species_list(file_path):
    species_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                species_name = line.strip()
                if species_name:
                    species_list.append(species_name)
                    logging.debug(f"Loaded species: {species_name} from line {line_number}")
        logging.info(f"Total species loaded from species list: {len(species_list)}")
        return species_list
    except FileNotFoundError as e:
        logging.error(f"Species list file not found: {file_path}\n{e}")
        return []

def load_category_list(file_path):
    category_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                category_name = line.strip()
                if category_name:
                    category_list.append(category_name)
        logging.info(f"Loaded {len(category_list)} categories from {file_path}")
    except FileNotFoundError as e:
        logging.error(f"Category list file not found: {file_path}\n{e}")
    return category_list

def extract_species_name_from_filename(filename):
    """
    Extract the species name from the QA filename by removing the extension and any known suffixes.
    Assumes that the species name can include underscores and is followed by a known suffix like '_description'.
    Examples:
        - "globosa_male_description.json" -> "globosa_male"
        - "globosa_female_description.json" -> "globosa_female"
        - "marionae_description.json" -> "marionae"
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Define known suffixes that might follow the species name
    known_suffixes = ['description', 'qa', 'color', 'annotation', 'data']
    # Check for known suffixes
    for suffix in known_suffixes:
        if basename.endswith(f'_{suffix}'):
            species_name = basename.rsplit(f'_{suffix}', 1)[0]
            logging.debug(f"Extracted species name: {species_name} from filename: {filename}")
            return species_name
    # If no known suffix, assume the entire basename is the species name
    species_name = basename
    logging.debug(f"Extracted species name: {species_name} from filename: {filename}")
    return species_name

def extract_species_name_from_measurement_filename(filename):
    """
    Extract the species name from the measurement filename.
    Assumes that the species name is the part of the filename before the first '.jpg'.
    Example:
        - "globosa_male.jpg_entire_forewing_ann9_entire_forewing.png" -> "globosa_male"
    """
    basename = os.path.basename(filename)
    if '.jpg' in basename:
        species_part = basename.split('.jpg')[0]
    else:
        species_part = basename.split('.')[0]
    species_name = species_part
    logging.debug(f"Extracted species name: {species_name} from measurement filename: {filename}")
    return species_name

def extract_species_name_from_color_base_name(base_name):
    """
    Extract the species name from the color data base_name.
    Assumes that the species name is the part of the base_name before the first '.jpg'.
    Example:
        - "adelpha_female.jpg_entire_forewing_ann10_entire_forewing.png" -> "adelpha_female"
        - "nolanae_male.jpg_cell-r2_ann2_cell-r2.png" -> "nolanae_male"
    """
    basename = os.path.basename(base_name)
    if '.jpg' in basename:
        species_part = basename.split('.jpg')[0]
    else:
        species_part = basename.split('.')[0]
    species_name = species_part
    logging.debug(f"Extracted species name: {species_name} from color base_name: {base_name}")
    return species_name

def extract_category_name_from_base_name(base_name, category_list):
    """
    Extract the category name from the base_name in the color data by matching against known category names.
    """
    basename = os.path.basename(base_name)
    for category in category_list:
        # Handle categories with special characters or multiple parts
        pattern = f'_{re.escape(category)}_'
        if re.search(pattern, basename):
            logging.debug(f"Extracted category name: {category} from base_name: {base_name}")
            return category
        elif basename.endswith(f'_{category}.png'):
            logging.debug(f"Extracted category name: {category} from base_name: {base_name}")
            return category
    logging.warning(f"Could not extract category name from base_name: {base_name}")
    return 'Unknown'

def average_measurements(measurements_list):
    """
    Average the numeric measurement fields for a list of measurements.
    Non-numeric fields are taken from the first measurement.
    """
    numeric_fields = {}
    for measurement in measurements_list:
        for key, value in measurement.items():
            if isinstance(value, (int, float)) and key not in ['image_id', 'annotation_index', 'pixels_per_mm']:
                numeric_fields.setdefault(key, []).append(value)

    averaged_measurement = {}
    for key, values in numeric_fields.items():
        averaged_value = np.mean(values)
        averaged_measurement[key] = averaged_value
        logging.debug(f"Averaged {key}: {averaged_value} from values: {values}")

    # Include non-numeric fields from the first measurement
    for key in measurements_list[0]:
        if key not in averaged_measurement:
            averaged_measurement[key] = measurements_list[0][key]
            logging.debug(f"Including non-numeric field {key}: {measurements_list[0][key]}")

    return averaged_measurement

def load_material_examined(material_examined_file):
    """
    Load material examined data from a TSV file.
    The file should have two columns: species_name and label_jsonl_path.
    Example (TSV):
        species_name	label_jsonl_path
        globosa_female	/path/to/globosa_female_Q41.json
        adelpha_female	/path/to/adelpha_female_Q41.json
    """
    material_examined = defaultdict(list)
    try:
        with open(material_examined_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_number, row in enumerate(reader, start=2):  # Starting at 2 to account for header
                species_name = row.get('species_name', '').strip()
                label_path = row.get('label_jsonl_path', '').strip()
                if species_name and label_path:
                    material_examined[species_name].append(label_path)
                    logging.debug(f"Loaded material examined for species '{species_name}': {label_path}")
                else:
                    logging.warning(f"Skipping invalid row {row_number} in material examined file: {row}")
    except FileNotFoundError as e:
        logging.error(f"Material examined file not found: {material_examined_file}\n{e}")
    except Exception as e:
        logging.error(f"Error reading material examined file: {material_examined_file}\n{e}")
    return material_examined

def parse_label_jsonl(label_jsonl_path):
    """
    Integrate the provided JSON parsing script to extract 'gbifData'.
    
    Args:
        label_jsonl_path (str): Path to the label JSON file.
    
    Returns:
        dict: A dictionary containing the extracted 'material_examined' data.
    """
    extracted_info = {}
    try:
        # Path to your JSON file
        file_path = label_jsonl_path

        # Step 1: Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_json = file.read()

        # Step 2: Clean the JSON string
        clean_json_str = raw_json.replace('\\n', '\n').replace('\\"', '"')

        print(f"Debug: Cleaned JSON String for '{os.path.basename(file_path)}':")
        print(clean_json_str)
        print("\n" + "="*50 + "\n")

        # Step 3: Parse the JSON content
        try:
            data = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed for {file_path}: {e}")
            return extracted_info

        # Step 4: Extract 'gbifData'
        gbif_data = data.get('gbifData')

        if gbif_data:
            print(f"Debug: Extracted 'gbifData' for '{os.path.basename(file_path)}':")
            print(json.dumps(gbif_data, indent=4))
            print("\n" + "="*50 + "\n")
            extracted_info['material_examined'] = gbif_data
        else:
            logging.warning(f"'gbifData' key not found in the JSON file: {file_path}")

    except FileNotFoundError as e:
        logging.error(f"JSON file not found: {label_jsonl_path}\n{e}")
    except Exception as e:
        logging.error(f"Unexpected error while processing {label_jsonl_path}: {e}")
    return extracted_info

def combine_data(measurements, colors, qa_data, material_examined, species_list=None, category_filter='all', category_list=None):
    combined = []

    # Create mappings from species names to color data, grouped by category
    colors_by_species_and_category = defaultdict(lambda: defaultdict(dict))
    for color_entry in colors:
        base_name = color_entry.get('base_name', '')
        species_name = extract_species_name_from_color_base_name(base_name)
        category_name = extract_category_name_from_base_name(base_name, category_list)
        if category_name != 'Unknown':
            colors_by_species_and_category[species_name][category_name] = color_entry
            logging.debug(f"Mapped color data to species '{species_name}', category '{category_name}'")
        else:
            logging.warning(f"Could not extract category name from base_name: {base_name}")

    # QA data is already mapped by species_name in load_qa_folder
    qa_by_species = qa_data

    # Group measurements by species and category
    measurements_by_species_and_category = defaultdict(lambda: defaultdict(list))
    for measurement in measurements:
        image_filename = measurement.get('image_filename', '')
        species_name = extract_species_name_from_measurement_filename(image_filename)
        category_name = measurement.get('category_name', 'Unknown')
        # Normalize category names
        if category_list and category_name in category_list:
            normalized_category_name = category_name
        else:
            # Attempt to find a matching category name (case-insensitive)
            normalized_category_name = next((cat for cat in category_list if cat.lower() == category_name.lower()), 'Unknown')
        measurements_by_species_and_category[species_name][normalized_category_name].append(measurement)
        logging.debug(f"Grouped measurement for species '{species_name}', category '{normalized_category_name}'")

    # Process each species
    for species_name, categories in measurements_by_species_and_category.items():
        # If a species list is provided, skip species not in the list
        if species_list and species_name not in species_list:
            logging.warning(f"Species '{species_name}' not found in species list. Skipping.")
            continue

        logging.info(f"Processing species: {species_name}")

        # Get QA data for this species
        qa_info = qa_by_species.get(species_name, {})
        if not qa_info:
            logging.warning(f"No QA data found for species: {species_name}")

        # Get Material Examined data for this species
        label_paths = material_examined.get(species_name, [])
        material_examined_entries = []
        for label_path in label_paths:
            extracted = parse_label_jsonl(label_path)
            if extracted:
                material_examined_entries.append(extracted)
            else:
                logging.warning(f"No extracted material examined data for species '{species_name}' from file '{label_path}'")

        # Organize material examined entries by locality
        locality_dict = defaultdict(list)
        for entry in material_examined_entries:
            gbif_data = entry.get('material_examined', {})
            if gbif_data:
                locality = gbif_data.get('locality', 'Unknown Locality')
                locality_dict[locality].append(gbif_data)
                logging.debug(f"Added material examined data for locality '{locality}' in species '{species_name}'")
            else:
                logging.warning(f"No 'material_examined' data found in entry for species '{species_name}'")

        species_entry = {
            'species_name': species_name,
            'qa': qa_info,
            'categories': [],
            'material_examined': locality_dict  # Add material examined data
        }

        # Process each category for this species
        for category_name, measurements_list in categories.items():
            # If a category is specified and it's not 'all', skip other categories
            if category_filter != 'all' and category_name != category_filter:
                logging.debug(f"Skipping category '{category_name}' for species '{species_name}'")
                continue

            # Only process if category_name is in category_list
            if category_list and category_name not in category_list:
                logging.warning(f"Category '{category_name}' not in category list. Skipping.")
                continue

            # Average the measurements
            averaged_measurement = average_measurements(measurements_list)

            # Get color data for this species and category
            color_info = colors_by_species_and_category.get(species_name, {}).get(category_name, {})
            if not color_info:
                logging.warning(f"No color data found for species '{species_name}', category '{category_name}'")

            category_entry = {
                'category_name': category_name,
                'measurement': averaged_measurement,
                'color': color_info
            }
            species_entry['categories'].append(category_entry)
            logging.debug(f"Added category '{category_name}' for species '{species_name}'")

        combined.append(species_entry)
        logging.debug(f"Combined data for species: {species_name}")

    return combined

def generate_prompt(combined_data, body_part):
    prompts = []
    for entry in combined_data:
        species_name = entry['species_name']
        qa = entry['qa']
        material_examined = entry.get('material_examined', {})

        # Start the species section
        description = f"Species Description for {species_name}:\n"

        # Include QA data once per species
        description += "\nMorphological Features:\n"
        if qa:
            if isinstance(qa, dict):
                for question, answer in qa.items():
                    description += f"Q: {question}\nA: {answer}\n"
            elif isinstance(qa, list):
                for qa_pair in qa:
                    question = qa_pair.get('question', 'Unknown Question')
                    answer = qa_pair.get('answer', 'Unknown Answer')
                    description += f"Q: {question}\nA: {answer}\n"
            else:
                logging.warning(f"Unexpected QA data format for species: {species_name}")
                description += "- Morphological details are not available.\n"
        else:
            description += "- Morphological details are not available.\n"

        # Process each category
        for category_entry in entry['categories']:
            category_name = category_entry['category_name']
            measurement = category_entry['measurement']
            color = category_entry['color']

            # Add category subsection
            description += f"\nCategory: {category_name}\n"

            # Include measurements
            description += "\nMeasurements:\n"
            for key, value in measurement.items():
                if isinstance(value, (int, float)):
                    if 'pixels' not in key and 'mm' in key:
                        key_formatted = key.replace('_mm', '').replace('_', ' ').capitalize()
                        description += f"- {key_formatted}: {value:.4f} mm\n"
                    elif key == 'length_to_height_ratio':
                        description += f"- Length to height ratio: {value:.4f}\n"

            # Include color information
            description += "\nColoration:\n"
            if color:
                color_info = color.get('foreground_mask', {})
                avg_color_name = color_info.get('average_color_name', 'unknown')
                dominant_color_name = color_info.get('dominant_color_name', 'unknown')
                top_k_color_names = color_info.get('top_k_color_names', [])
                description += f"- The average color is {avg_color_name}.\n"
                description += f"- The dominant color is {dominant_color_name}.\n"
                if top_k_color_names:
                    top_colors = ', '.join(top_k_color_names)
                    description += f"- The top colors are: {top_colors}.\n"
            else:
                description += "- Color information is not available.\n"

        # Add Material Examined section
        description += "\nMaterial Examined:\n"
        if material_examined:
            for locality, specimens in material_examined.items():
                if isinstance(specimens, list):
                    if len(specimens) > 1:
                        description += f"- **Locality**: {locality}\n  - Number of specimens: {len(specimens)}\n"
                    elif len(specimens) == 1:
                        specimen = specimens[0]
                        description += f"- **Locality**: {locality}\n"
                        for field, value in specimen.items():
                            description += f"  - **{field.capitalize()}**: {value}\n"
                elif isinstance(specimens, dict):
                    description += f"- **Locality**: {locality}\n"
                    for field, value in specimens.items():
                        description += f"  - **{field.capitalize()}**: {value}\n"
                else:
                    logging.warning(f"Unexpected specimen format for locality '{locality}' in species '{species_name}'")
        else:
            description += "- No material examined data available.\n"

        # Append to prompts list
        prompts.append(description)
        logging.debug(f"Generated prompt for species: {species_name}")

    return prompts

def main():
    args = parse_args()

    logging.info("Loading measurements data...")
    measurements = load_jsonl(args.measurements)
    logging.info(f"Loaded {len(measurements)} measurement entries.")

    logging.info("Loading color data...")
    colors = load_jsonl(args.colors)
    logging.info(f"Loaded {len(colors)} color entries.")

    logging.info("Loading QA data from folder...")
    qa_data = load_qa_folder(args.qa_folder)
    logging.info(f"Loaded QA data for {len(qa_data)} species.")

    species_list = None
    if args.species_list:
        species_list = load_species_list(args.species_list)
        logging.info(f"Loaded species list with {len(species_list)} species.")

    category_list = None
    if args.category_list:
        category_list = load_category_list(args.category_list)
        if not category_list:
            logging.error("Category list is empty or could not be loaded. Exiting.")
            return
    else:
        logging.error("Category list file must be provided using --category_list")
        return

    logging.info("Loading material examined data...")
    material_examined = load_material_examined(args.material_examined)
    logging.info(f"Loaded material examined data for {len(material_examined)} species.")

    logging.info("Combining data...")
    combined_data = combine_data(measurements, colors, qa_data, material_examined, species_list, args.category, category_list)
    logging.info(f"Total species processed: {len(combined_data)}")

    logging.info("Generating prompts...")
    prompts = generate_prompt(combined_data, args.body_part)
    logging.info(f"Generated {len(prompts)} prompts.")

    # Write the prompts to the output file
    if args.output:
        output_path = args.output
        # Ensure the output file has a .txt extension
        if not output_path.lower().endswith('.txt'):
            output_path += '.txt'
            logging.debug(f"Added .txt extension to output file: {output_path}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    f.write(prompt)
                    f.write("\n" + "#" * 80 + "\n")
            logging.info(f"Prompts written to {output_path}")
        except Exception as e:
            logging.error(f"Error writing to output file: {output_path}\n{e}")
    else:
        # Output the prompts to stdout
        for prompt in prompts:
            print(prompt)
            print("\n" + "#" * 80 + "\n")

    logging.info("Processing completed.")

if __name__ == "__main__":
    main()
