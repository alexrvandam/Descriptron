import argparse
import os
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Update species descriptions by replacing old dichotomous keys with a new one.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input species descriptions file.')
    parser.add_argument('--key_file', type=str, required=True, help='Path to the new dichotomous key file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output updated species descriptions file.')
    return parser.parse_args()

def remove_old_keys(content):
    """
    Removes old 'Dichotomous Key' sections from the content.
    """
    # Regular expression to match 'Dichotomous Key:' sections
    pattern = r'Dichotomous Key:.*?(?=(Species Description for|\Z))'
    updated_content = re.sub(pattern, '', content, flags=re.DOTALL)
    return updated_content.strip()

def append_new_key(content, new_key):
    """
    Appends the new dichotomous key to the content.
    """
    return f"{content.strip()}\n\n{new_key.strip()}\n"

def main():
    args = parse_arguments()

    # Validate input files
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        exit(1)
    if not os.path.isfile(args.key_file):
        print(f"Error: Key file '{args.key_file}' does not exist.")
        exit(1)

    # Read the species descriptions file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove old dichotomous keys
    content_without_old_keys = remove_old_keys(content)

    # Read the new dichotomous key
    with open(args.key_file, 'r', encoding='utf-8') as f:
        new_key = f.read()

    # Append the new key
    updated_content = append_new_key(content_without_old_keys, new_key)

    # Write the updated content to the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print(f"Updated species descriptions have been written to {args.output_file}")

if __name__ == '__main__':
    main()
