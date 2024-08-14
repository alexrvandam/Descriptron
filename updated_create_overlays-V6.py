import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
json_output = config['Paths']['json_output']
image_dir = config['Paths']['image_directory']
json_path = config['Paths']['json_input']



def create_masks(image, contour, color):
    # Overlay the contour with the specified color and transparency
    overlay = image.copy()
    cv2.drawContours(overlay, [contour.astype(int)], -1, color, thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)  # 0.4 opacity for overlay, 0.6 for original

def process_view(image_dir, json_path, output_dir, view_name):
    # Load JSON data
    with open(json_path, 'r') as f:
        contours_data = json.load(f)

    # Define the set of desired classes and corresponding colors
    desired_classes = {
        'left_elytra': (255, 0, 0),    # Red
        'pronotum': (0, 255, 0),       # Green
        'head': (0, 0, 255),           # Blue
        'rostrum': (255, 255, 0)       # Yellow
    }

    # Filter contours data for specific classes and directory
    relevant_files = [os.path.basename(f) for f in os.listdir(image_dir) if view_name in f and f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Relevant files: {len(relevant_files)} found")

    filtered_contours = [
        entry for entry in contours_data
        if any(pred['category_name'] in desired_classes for pred in entry['predictions'])
        and os.path.basename(entry['file_name']) in relevant_files
    ]
    print(f"Filtered contours data: {len(filtered_contours)} entries found")

    num_images = len(filtered_contours)
    cols = 10  # Number of columns
    rows = int(np.ceil(num_images / cols))

    # Create a figure with fixed grid size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for idx, entry in enumerate(filtered_contours):
        image_name = os.path.basename(entry['file_name'])
        image_path = os.path.join(image_dir, image_name)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Overlay the contours
        for prediction in entry['predictions']:
            if prediction['category_name'] in desired_classes:
                contours = prediction['contours']
                if isinstance(contours, list) and all(isinstance(i, list) for i in contours) and all(len(i) % 2 == 0 for i in contours):
                    for contour in contours:
                        contour = np.array(contour, dtype=np.float32).reshape(-1, 2)
                        image = create_masks(image, contour, desired_classes[prediction['category_name']])

        # Resize image to fit within the grid cell
        height, width, _ = image.shape
        max_dim = max(height, width)
        scale = 200 / max_dim  # Resize factor to make the largest dimension fit into 200 pixels
        resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Display image with contours in the grid cell
        axes[idx].imshow(resized_image)
        axes[idx].axis('off')
        axes[idx].set_title(image_name, fontsize=6)

    # Turn off unused axes
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    figure_path = os.path.join(output_dir, f'overlayed_images_{view_name}.png')
    plt.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Saved overlayed image for {view_name} view to {figure_path}")

def main(image_dir, json_path, output_dir):
    # Process both dorsal and lateral views
    process_view(image_dir, json_path, output_dir, 'dorsal')
    process_view(image_dir, json_path, output_dir, 'lateral')

if __name__ == "__main__":
    json_output = config['Paths']['json_output']
    image_dir = config['Paths']['image_directory']
    json_path = config['Paths']['json_input']
    #image_dir = "/vandam/insect_images/coleoptera/big_training/Curculionidae"
    #json_path = "/vandam/insect_images/coleoptera/pred_output/inference_results.json"
    #output_dir = "/vandam/insect_images/coleoptera/pred_output/masks_output" # or pred_output does not matter really
    
    os.makedirs(output_dir, exist_ok=True)
    main(image_dir, json_path, output_dir)

