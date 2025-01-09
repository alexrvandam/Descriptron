import argparse
import csv
import os
import re
import json

def extract_q_key_lines(json_file_path, q_key):
    """
    Extracts the line containing the specified q_key and the line immediately following it from a JSON file.

    :param json_file_path: Path to the JSON file.
    :param q_key: The key to search for (e.g., 'Q41').
    :return: Tuple containing the q_key line and the following line. Returns (None, None) if not found.
    """
    q_key_line = None
    following_line = None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if f'"{q_key}' in line:
                    q_key_line = line.strip()
                    if i + 1 < len(lines):
                        following_line = lines[i + 1].strip()
                    break
    except Exception as e:
        print(f"Error reading file '{json_file_path}': {e}")
    return q_key_line, following_line

def ensure_complete_json(json_str):
    """
    Ensures that the extracted JSON string is complete by checking for balanced braces.
    If incomplete, appends the necessary closing braces.

    :param json_str: The JSON string to check.
    :return: A complete JSON string.
    """
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        # Calculate the number of missing closing braces
        missing_braces = open_braces - close_braces
        json_str += '}' * missing_braces
    return json_str

def sanitize_species_name(species_name):
    """
    Sanitizes the species name by replacing spaces with underscores and removing problematic characters.

    :param species_name: The original species name.
    :return: A sanitized species name suitable for filenames.
    """
    # Replace spaces with underscores
    sanitized = species_name.replace(' ', '_')
    # Remove any characters that are not alphanumeric or underscores
    sanitized = re.sub(r'[^A-Za-z0-9_]', '', sanitized)
    return sanitized

def process_tsv(species_list_path, output_json_dir, output_tsv_path, q_key):
    """
    Processes the TSV file to extract lines based on the specified q_key from each JSON file,
    ensures JSON completeness, writes separate JSON files for each unique JSON file, and creates a TSV mapping of species to JSON files.

    :param species_list_path: Path to the species_list TSV file.
    :param output_json_dir: Directory to save the individual JSON files.
    :param output_tsv_path: Path to the output TSV file mapping species to JSON files.
    :param q_key: The key to extract from the JSON files (e.g., 'Q41').
    """
    extracted_data = []
    processed_files = {}  # Maps input JSON file path to output JSON file path

    # Ensure the output directory exists
    os.makedirs(output_json_dir, exist_ok=True)

    with open(species_list_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                print(f"Skipping row {row_number}: less than 2 columns.")
                continue  # Skip rows that don't have at least two columns
            
            species, json_file_path = row[0].strip(), row[1].strip()
            
            if not species or not json_file_path:
                print(f"Skipping row {row_number}: empty species name or file path.")
                continue  # Skip rows with empty fields
            
            if not os.path.isfile(json_file_path):
                print(f"File not found: '{json_file_path}'. Skipping.")
                continue  # Skip if JSON file does not exist
            
            # Check if the JSON file has already been processed
            if json_file_path not in processed_files:
                q_key_line, following = extract_q_key_lines(json_file_path, q_key)
                if q_key_line and following:
                    # Combine q_key line and following line
                    combined_q_key = f"{q_key_line} {following}"
                    # Extract the JSON content using regex
                    # Assuming the JSON is within triple backticks and starts with ```json
                    match = re.search(r"```json\s*(\{.*\})\s*```", combined_q_key, re.DOTALL)
                    if match:
                        json_content = match.group(1).strip()
                    else:
                        # Attempt to extract JSON directly
                        # Find the first '{' and last '}' to extract JSON
                        json_start = combined_q_key.find('{')
                        json_end = combined_q_key.rfind('}')
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            json_content = combined_q_key[json_start:json_end+1].strip()
                        else:
                            print(f"No valid JSON found in {q_key} of file '{json_file_path}' for species '{species}'. Skipping.")
                            continue
                    
                    # Ensure the JSON is complete
                    json_content = ensure_complete_json(json_content)
                    
                    # Define the output JSON file path based on input JSON file name
                    input_basename = os.path.basename(json_file_path)
                    input_name, _ = os.path.splitext(input_basename)
                    output_json_path = os.path.join(output_json_dir, f"{input_name}_extracted_{q_key}.json")
                    
                    try:
                        with open(output_json_path, 'w', encoding='utf-8') as out_json:
                            out_json.write(json_content)
                        print(f"Saved complete {q_key} JSON to '{output_json_path}'.")
                        # Map the input JSON file to the output JSON file
                        processed_files[json_file_path] = output_json_path
                    except Exception as e:
                        print(f"Error writing JSON file '{output_json_path}': {e}")
                        continue
                else:
                    print(f"{q_key} not found or incomplete in file '{json_file_path}' for species '{species}'.")
                    continue

            # Retrieve the output JSON file path for this species
            output_json_path = processed_files.get(json_file_path)
            if output_json_path:
                extracted_data.append({'species_name': species, 'label_jsonl_path': output_json_path})
                print(f"Mapped species '{species}' to JSON file '{output_json_path}'.")
            else:
                print(f"Failed to map species '{species}' due to previous errors.")

    # Write the TSV mapping species to JSON files
    try:
        with open(output_tsv_path, 'w', encoding='utf-8', newline='') as out_tsv:
            writer = csv.DictWriter(out_tsv, fieldnames=['species_name', 'label_jsonl_path'], delimiter='\t')
            writer.writeheader()
            for entry in extracted_data:
                writer.writerow(entry)
        print(f"\nExtraction complete. Mapping saved to '{output_tsv_path}'.")
    except Exception as e:
        print(f"Error writing mapping TSV '{output_tsv_path}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract Q-key lines from JSON files and create mapping TSV.")
    parser.add_argument('--species_list', type=str, required=True, help='Path to the species_list TSV file.')
    parser.add_argument('--output_json_dir', type=str, default='extracted_q_key_jsons', help='Directory to save the extracted Q-key JSON files.')
    parser.add_argument('--output_tsv', type=str, default='material_examined.tsv', help='Path to the output TSV file mapping species to JSON files.')
    parser.add_argument('--q_key', type=str, default='Q41', help='Key to extract from JSON files (default: Q41).')
    return parser.parse_args()

if __name__ == "__main__":
    args = main()
    process_tsv(args.species_list, args.output_json_dir, args.output_tsv, args.q_key)
