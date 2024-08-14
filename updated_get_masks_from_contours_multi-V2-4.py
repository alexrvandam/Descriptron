import configparser
import os
import json
import numpy as np
import cv2

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
image_directory = config['Paths']['image_directory']
train_data = config['Paths']['train_data']
val_data = config['Paths']['val_data']
test_data = config['Paths']['test_data']
model_output = config['Paths']['model_output']
plot_output = config['Paths']['plot_output']
json_input = config['Paths']['json_input']
json_output = config['Paths']['json_output']

def create_masks(image, contour):
    # Create a binary mask
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(binary_mask, [contour.astype(int)], -1, 255, thickness=cv2.FILLED)
    
    # Create an RGB foreground mask from the original image
    rgb_mask = np.zeros_like(image)
    rgb_mask[binary_mask == 255] = image[binary_mask == 255]
    
    return binary_mask, rgb_mask

def save_contour_as_text(contour, output_path):
    """ Save the contour as a simple XY text file. """
    np.savetxt(output_path, contour, delimiter=',', fmt='%.2f')
    print(f"Saved contour to {output_path}")

def save_contour_as_npy(contour, output_path):
    """ Save the contour as a .npy file. """
    np.save(output_path, contour)
    print(f"Saved contour to {output_path}")

def main(image_directory, json_path, output_dir):
    # Load JSON data
    with open(json_path, 'r') as f:
        contours_data = json.load(f)

    # Define the set of desired classes
    desired_classes = {'left_elytra', 'pronotum', 'head', 'rostrum'}

    # Filter contours data for specific classes and directory
    relevant_files = [os.path.basename(f) for f in os.listdir(image_directory) if f.startswith(('lateral', 'dorsal')) and f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Relevant files: {len(relevant_files)} found")

    filtered_contours = [
        entry for entry in contours_data
        if any(pred['category_name'] in desired_classes for pred in entry['predictions'])
        and os.path.basename(entry['file_name']) in relevant_files
    ]
    print(f"Filtered contours data: {len(filtered_contours)} entries found")

    for idx, entry in enumerate(filtered_contours):
        image_name = os.path.basename(entry['file_name'])
        image_path = os.path.join(image_directory, image_name)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        print(f"Processing image: {image_name}")

        # Load the image
        image = cv2.imread(image_path)

        # Filter predictions for the desired classes
        class_predictions = [pred for pred in entry['predictions'] if pred['category_name'] in desired_classes]
        
        if class_predictions:
            for prediction in class_predictions:
                try:
                    contours = prediction['contours']
                    if isinstance(contours, list) and all(isinstance(i, list) for i in contours) and all(len(i) % 2 == 0 for i in contours):
                        for contour in contours:
                            contour = np.array(contour, dtype=np.float32).reshape(-1, 2)  # Convert to Nx2 shape
                            
                            # Create binary and RGB masks
                            binary_mask, rgb_mask = create_masks(image, contour)

                            # Save binary and RGB masks in class-specific directories
                            class_output_dir = os.path.join(output_dir, prediction['category_name'])
                            os.makedirs(class_output_dir, exist_ok=True)

                            # Construct the output filenames with class name
                            base_filename = os.path.splitext(image_name)[0]
                            binary_mask_filename = f"binary_mask_{prediction['category_name']}_{base_filename}.png"
                            rgb_mask_filename = f"foreground_mask_{prediction['category_name']}_{base_filename}.png"
                            contour_text_filename = f"contour_{prediction['category_name']}_{base_filename}.txt"
                            contour_npy_filename = f"contour_{prediction['category_name']}_{base_filename}.npy"

                            binary_mask_path = os.path.join(class_output_dir, binary_mask_filename)
                            rgb_mask_path = os.path.join(class_output_dir, rgb_mask_filename)
                            contour_text_path = os.path.join(class_output_dir, contour_text_filename)
                            contour_npy_path = os.path.join(class_output_dir, contour_npy_filename)

                            # Save the masks and contours
                            cv2.imwrite(binary_mask_path, binary_mask)
                            cv2.imwrite(rgb_mask_path, rgb_mask)
                            save_contour_as_text(contour, contour_text_path)
                            save_contour_as_npy(contour, contour_npy_path)

                    else:
                        print(f"Invalid contour format in {image_name}, skipping this contour.")
                except ValueError as e:
                    print(f"Skipping {image_name} due to error: {e}")

if __name__ == "__main__":
    image_directory = config['Paths']['image_directory']
    json_path = config['Paths']['json_input']
    output_dir = config['Paths']['json_output']
    
    os.makedirs(output_dir, exist_ok=True)
    main(image_directory, json_path, output_dir)
