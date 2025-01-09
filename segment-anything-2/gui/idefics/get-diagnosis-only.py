import argparse
import os
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract Description sections from species descriptions.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input species descriptions file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file for extracted descriptions.')
    return parser.parse_args()

def extract_descriptions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content into species entries using the separator '---' or similar patterns
    species_entries = re.split(r'\n-{3,}\n', content)

    descriptions = []

    for entry in species_entries:
        # Extract the species name
        species_match = re.search(r'Species Description for \*(.+?)\*:', entry)
        if species_match:
            species_name = species_match.group(1).strip()
        else:
            continue  # Skip if species name is not found

        # Extract the Description section
        description_match = re.search(r'Diagnosis:\n\n(.+?)(?:\n\n\w+:|\n\n-{3,}|\Z)', entry, re.DOTALL)
        if description_match:
            description_text = description_match.group(1).strip()
            # Append species name and description to the list
            descriptions.append(f"Species: {species_name}\nDiagnosis:\n{description_text}\n")
        else:
            print(f"Warning: Description section not found for species '{species_name}'.")

    # Write the extracted descriptions to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(descriptions))

    print(f"Extracted descriptions have been written to {output_file}")

def main():
    args = parse_arguments()
    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        exit(1)
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    extract_descriptions(args.input_file, args.output_file)

if __name__ == '__main__':
    main()
