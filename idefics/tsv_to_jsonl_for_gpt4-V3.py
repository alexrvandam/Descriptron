import argparse
import csv
import json
import base64
import os

def encode_image_base64(image_path):
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def convert_tsv_to_jsonl_with_base64(input_tsv, output_jsonl):
    """Converts TSV to JSONL with base64-encoded images."""
    with open(input_tsv, newline='', encoding='utf-8') as tsv_file, open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        for row in reader:
            image_path = row['image_path'].strip()  # Column name for the image path in TSV
            base64_image = encode_image_base64(image_path)
            
            if not base64_image:
                print(f"Skipping row with missing image: {image_path}")
                continue
            
            data = {
                "messages": [
                    {"role": "system", "content": "You are an assistant that provides detailed entomological descriptions of Psyllidae specimens."},
                    {"role": "user", "content": row['question']},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    },
                    {"role": "assistant", "content": row['answer']}
                ]
            }
            
            jsonl_file.write(json.dumps(data) + '\n')
        print(f"JSONL file with base64 images saved to: {output_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Convert TSV to JSONL with base64-encoded images for GPT-4 fine-tuning.")
    parser.add_argument('--input_tsv', type=str, required=True, help='Path to the input TSV file.')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to the output JSONL file.')
    args = parser.parse_args()

    convert_tsv_to_jsonl_with_base64(args.input_tsv, args.output_jsonl)

if __name__ == "__main__":
    main()
