# remove_images_from_coco.py
import argparse
import json

def load_file_list(file_list_path):
    with open(file_list_path, 'r') as f:
        # One file name per line; strip whitespace
        return {line.strip() for line in f if line.strip()}

def remove_images_from_coco(coco_json_path, file_names_to_remove, output_json_path):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    
    # Identify image IDs to remove based on file_name
    image_ids_to_remove = set()
    new_images = []
    for image in images:
        if image.get("file_name") in file_names_to_remove:
            image_ids_to_remove.add(image.get("id"))
        else:
            new_images.append(image)
    
    new_annotations = [ann for ann in annotations if ann.get("image_id") not in image_ids_to_remove]
    
    data["images"] = new_images
    data["annotations"] = new_annotations
    
    with open(output_json_path, 'w') as out_f:
        json.dump(data, out_f, indent=4)
    
    print(f"Removed {len(image_ids_to_remove)} images and associated annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove images from a COCO JSON file based on a list of file names."
    )
    parser.add_argument("--coco", required=True, help="Path to input COCO JSON file")
    parser.add_argument("--file_list", required=True, help="Text file with one image file name per line")
    parser.add_argument("--output", required=True, help="Output COCO JSON file path")
    args = parser.parse_args()
    
    file_names = load_file_list(args.file_list)
    remove_images_from_coco(args.coco, file_names, args.output)
