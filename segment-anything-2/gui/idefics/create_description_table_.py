#!/usr/bin/env python3
"""
create_models_csv.py

This script generates a 'models.csv' file by consolidating species descriptions from multiple models
alongside their corresponding images based on the 'base_name'. It ensures that descriptions are correctly
aligned with species and handles potential discrepancies gracefully.

Usage:
    python create_models_csv.py \
        --image_dir /path/to/images \
        --base_names_file /path/to/base_names.txt \
        --descriptions_dir /path/to/descriptions \
        --output_csv /path/to/output/models.csv
"""

import pandas as pd
import os
import sys
import argparse
import logging
import glob
import re

def configure_logging():
    """
    Configures the logging settings.
    """
    logging.basicConfig(
        filename='create_models_csv.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def find_image_path(base_name, image_dir, allowed_extensions=['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
    """
    Finds the image file corresponding to the given base_name in the image_dir.

    Parameters:
    - base_name (str): The standardized species name.
    - image_dir (str): Path to the directory containing images.
    - allowed_extensions (list): List of allowed image file extensions.

    Returns:
    - str or None: Full path to the image file if found, else None.
    """
    pattern = os.path.join(image_dir, f"{base_name}*")
    matched_files = []
    for ext in allowed_extensions:
        matched_files.extend(glob.glob(f"{pattern}{ext}"))
    if len(matched_files) == 1:
        return matched_files[0]
    elif len(matched_files) > 1:
        logging.warning(f"Multiple images found for base_name '{base_name}'. Using the first match: '{matched_files[0]}'")
        return matched_files[0]
    else:
        logging.warning(f"No image found for base_name '{base_name}'.")
        return None

def load_base_names(base_names_file):
    """
    Loads base_names from a text file.

    Parameters:
    - base_names_file (str): Path to the base_names text file.

    Returns:
    - list: List of base_names.
    """
    if not os.path.exists(base_names_file):
        logging.error(f"Base names file '{base_names_file}' does not exist.")
        print(f"Error: Base names file '{base_names_file}' does not exist.")
        sys.exit(1)
    
    with open(base_names_file, 'r', encoding='utf-8') as file:
        base_names = [line.strip().lower() for line in file if line.strip()]
    
    # Check for duplicates
    duplicates = pd.Series(base_names).duplicated().sum()
    if duplicates > 0:
        logging.warning(f"Found {duplicates} duplicate base_names in '{base_names_file}'. Duplicates will be processed.")
        print(f"Warning: Found {duplicates} duplicate base_names in '{base_names_file}'. Duplicates will be processed.")
    
    logging.info(f"Loaded {len(base_names)} base_names from '{base_names_file}'.")
    print(f"Loaded {len(base_names)} base_names from '{base_names_file}'.")
    
    return base_names

def load_descriptions(descriptions_dir):
    """
    Loads descriptions from multiple model description files.

    Parameters:
    - descriptions_dir (str): Path to the directory containing description files.

    Returns:
    - dict: Dictionary with model_name as keys and list of descriptions as values.
    """
    if not os.path.isdir(descriptions_dir):
        logging.error(f"Descriptions directory '{descriptions_dir}' does not exist or is not a directory.")
        print(f"Error: Descriptions directory '{descriptions_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    description_files = glob.glob(os.path.join(descriptions_dir, "*.txt"))
    if not description_files:
        logging.error(f"No description files found in '{descriptions_dir}'. Ensure description files are in '.txt' format.")
        print(f"Error: No description files found in '{descriptions_dir}'. Ensure description files are in '.txt' format.")
        sys.exit(1)
    
    descriptions_dict = {}
    for desc_file in description_files:
        model_name = os.path.splitext(os.path.basename(desc_file))[0]
        with open(desc_file, 'r', encoding='utf-8') as file:
            descriptions = [line.strip() for line in file if line.strip()]
        
        descriptions_dict[model_name] = descriptions
        logging.info(f"Loaded {len(descriptions)} descriptions for model '{model_name}' from '{desc_file}'.")
        print(f"Loaded {len(descriptions)} descriptions for model '{model_name}' from '{desc_file}'.")
    
    return descriptions_dict

def create_models_csv(base_names, descriptions_dict, image_dir, output_csv, allowed_extensions=['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
    """
    Creates the models.csv file by consolidating base_names, model descriptions, and image paths.

    Parameters:
    - base_names (list): List of base_names.
    - descriptions_dict (dict): Dictionary with model_name as keys and list of descriptions as values.
    - image_dir (str): Path to the directory containing images.
    - output_csv (str): Path to save the generated CSV file.
    - allowed_extensions (list): List of allowed image file extensions.
    """
    records = []
    num_base_names = len(base_names)
    
    for model_name, descriptions in descriptions_dict.items():
        if len(descriptions) != num_base_names:
            logging.error(f"Number of descriptions ({len(descriptions)}) for model '{model_name}' does not match number of base_names ({num_base_names}). Skipping this model.")
            print(f"Error: Number of descriptions ({len(descriptions)}) for model '{model_name}' does not match number of base_names ({num_base_names}). Skipping this model.")
            continue
        
        for base_name, description in zip(base_names, descriptions):
            image_path = find_image_path(base_name, image_dir, allowed_extensions)
            if image_path:
                records.append({
                    'base_name': base_name,
                    'model_name': model_name,
                    'description': description,
                    'image_path': image_path
                })
            else:
                logging.warning(f"Image for base_name '{base_name}' not found. Entry for model '{model_name}' will have 'image_path' as None.")
                records.append({
                    'base_name': base_name,
                    'model_name': model_name,
                    'description': description,
                    'image_path': None
                })
    
    if not records:
        logging.error("No records to write to CSV. Exiting.")
        print("Error: No records to write to CSV. Ensure that descriptions and images are correctly provided.")
        sys.exit(1)
    
    models_df = pd.DataFrame(records)
    models_df.to_csv(output_csv, index=False)
    logging.info(f"Successfully created '{output_csv}' with {len(models_df)} records.")
    print(f"Successfully created '{output_csv}' with {len(models_df)} records.")

def main():
    """
    Main function to execute the script.
    """
    configure_logging()
    
    parser = argparse.ArgumentParser(description="Create 'models.csv' by consolidating species descriptions from multiple models and their corresponding images.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing image files.')
    parser.add_argument('--base_names_file', type=str, required=True, help='Path to the text file listing base_names (one per line).')
    parser.add_argument('--descriptions_dir', type=str, required=True, help='Path to the directory containing model description text files.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the generated models.csv file.')
    parser.add_argument('--allowed_extensions', type=str, nargs='*', default=['.png', '.jpg', '.jpeg', '.bmp', '.tiff'], help='List of allowed image file extensions. Default: .png, .jpg, .jpeg, .bmp, .tiff')
    
    args = parser.parse_args()
    
    # Validate image directory
    if not os.path.isdir(args.image_dir):
        logging.error(f"Image directory '{args.image_dir}' does not exist or is not a directory.")
        print(f"Error: Image directory '{args.image_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Load base_names
    base_names = load_base_names(args.base_names_file)
    
    # Load descriptions
    descriptions_dict = load_descriptions(args.descriptions_dir)
    
    # Create models.csv
    create_models_csv(
        base_names=base_names,
        descriptions_dict=descriptions_dict,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        allowed_extensions=args.allowed_extensions
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
