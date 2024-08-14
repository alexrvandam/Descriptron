import configparser
import json
import os

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths from the config file
train_data = config['Paths']['train_data']
val_data = config['Paths']['val_data']
test_data = config['Paths']['test_data']
model_output = config['Paths']['model_output']
plot_output = config['Paths']['plot_output']
json_input = config['Paths']['json_input']
json_output = config['Paths']['json_output']
image_directory = config['Paths']['image_directory']  # Directory containing the specific images to process

# Extract directory name for the output file name
directory_name = os.path.basename(image_directory)

# Get list of all image files in the directory
specific_images = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Load the predictions
with open(json_input, 'r') as f:
    predictions = json.load(f)

# List of all 63 sclerite classes
sclerite_classes = ["left_metafemur", "right_pedicel", "mesosternum", "mouth_parts", "right_scape", "left_scape", "left_mesotarsus", "right_mesofemur", "left_compound_eye", "right_procoxa", "right_metafemur", "left_mesotibia", "right_flagellum", "right_scrobe", "metepimeron", "left_club", "left_mesofemur", "head", "right_protarsus", "left_metacoxa", "right_elytra", "mesepimeron", "right_mesocoxa", "right_mesotibia", "ventrite_2", "pygidium", "left_metatrochanter", "left_protarsus", "right_protibia", "left_metatibia", "left_protrochanter", "mesepisternum", "ventrite_3", "ventrite_5", "ventrite_1", "metepisternum", "right_protrochanter", "left_protibia", "right_antenna", "left_mesocoxa", "right_metatrochanter", "scutellum", "left_profemur", "rostrum", "right_compound_eye", "right_mesotrochanter", "pronotum", "left_pedicel", "scrobe", "right_metatibia", "right_club", "right_profemur", "left_mesotrochanter", "left_elytra", "ventrite_4", "right_metatarsus", "right_mesotarsus", "left_procoxa", "right_metacoxa", "left_antenna", "metasterunum", "left_flagellum", "left_metatarsus"]

# Prepare VIA2 JSON structure
via2_data = {
    "_via_settings": {
        "project": {
            "name": "Sclerite Annotation",
            "id": "sclerite_project",
            "created": "Date and Time",
            "type": "image",
            "store": "localStorage"
        },
        "core": {
            "buffer_size": 18,
            "annotation_mode": "polygon"
        },
        "attribute": {
            "region": {
                "sclerite": {
                    "type": "dropdown",
                    "description": "",
                    "options": {sclerite: sclerite for sclerite in sclerite_classes},
                    "default_options": {}
                }
            },
            "file": {}
        }
    },
    "_via_img_metadata": {},
    "_via_attributes": {
        "region": {
            "sclerite": {
                "type": "dropdown",
                "description": "Sclerite type",
                "options": {sclerite: sclerite for sclerite in sclerite_classes},
                "default_options": {}
            }
        }
    }
}

image_id = 1

for pred in predictions:
    if pred["file_name"] in specific_images:
        file_data = {
            "filename": os.path.basename(pred["file_name"]),
            "size": os.path.getsize(pred["file_name"]),
            "regions": [],
            "file_attributes": {}
        }

        for region in pred["predictions"]:
            shape_attributes = {
                "name": "polygon",
                "all_points_x": [],
                "all_points_y": []
            }

            for contour in region["contours"]:
                x_points = [contour[i] for i in range(0, len(contour), 2)]
                y_points = [contour[i + 1] for i in range(0, len(contour), 2)]
                shape_attributes["all_points_x"].extend(x_points)
                shape_attributes["all_points_y"].extend(y_points)

            region_attributes = {
                "sclerite": region["category_name"]
            }

            region_data = {
                "shape_attributes": shape_attributes,
                "region_attributes": region_attributes
            }

            file_data["regions"].append(region_data)

        via2_data["_via_img_metadata"][str(image_id)] = file_data
        image_id += 1

# Save VIA2 JSON with directory name in the filename
via2_json_path = os.path.join(json_output, f'specific_{directory_name}_annotations.json')
with open(via2_json_path, 'w') as f:
    json.dump(via2_data, f)

print(f"Saved VIA2 JSON for specific images to {via2_json_path}")
