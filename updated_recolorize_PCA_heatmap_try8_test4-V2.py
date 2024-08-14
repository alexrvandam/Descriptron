import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from math import ceil
from textwrap import wrap

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
json_output = config['Paths']['json_output']
color_output = config['Paths']['color_output']


def load_thumbnail(image_path, thumbnail_size=(50, 50)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    thumbnail = cv2.resize(img, thumbnail_size)
    return thumbnail

def load_thumbnail_key(image_path, thumbnail_size=(300, 300)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    thumbnail = cv2.resize(img, thumbnail_size)
    return thumbnail


def create_thumbnail_legend(ax, file_names, thumbnails):
    for i, (file, thumbnail) in enumerate(zip(file_names, thumbnails)):
        imagebox = OffsetImage(thumbnail, zoom=1)
        ab = AnnotationBbox(imagebox, (principal_components[i, 0], principal_components[i, 1]),
                            xycoords='data', boxcoords="offset points", pad=0.5)
        ax.add_artist(ab)
        #ax.annotate(file, (principal_components[i, 0], principal_components[i, 1]), 
                   # xytext=(10, 10), textcoords='offset points', ha='right', va='bottom')

def wrap_text(text, char_limit):
    words = text.split('_')
    wrapped_text = ''
    line = ''
    for word in words:
        if len(line + word) <= char_limit:
            line += word + ' '
        else:
            wrapped_text += line + '\n'
            line = word + ' '
    wrapped_text += line
    return wrapped_text.strip()


def normalize_color(image_path):
    # Read and normalize the image
    img = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_norm = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return img_norm

def generate_heatmap(reference_img, target_img):
    # Compute the absolute difference between images
    difference = cv2.absdiff(reference_img, target_img)
    gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    heatmap_img = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
    return heatmap_img

# Directory paths
input_directory = json_output
output_directory = color_output
# Load thumbnails
file_names = [f for f in sorted(os.listdir(input_directory)) if f.startswith('normalized_') and f.endswith('.jpg')]
thumbnails = [load_thumbnail(os.path.join(input_directory, f)) for f in file_names]

# Normalize colors of all images and rename them
normalized_images = []
file_names = []
for file in sorted(os.listdir(input_directory)):
    if file.startswith('segmented_roi_foreground_mask') and file.endswith('.jpg'):
        input_path = os.path.join(input_directory, file)
        img_norm = normalize_color(input_path)
        normalized_images.append(img_norm)

        # Rename and save normalized images
        new_name = f'normalized_{file}'
        output_path = os.path.join(output_directory, new_name)
        cv2.imwrite(output_path, img_norm)
        file_names.append(new_name)

# Use the first image as the reference for heatmaps
reference_img = normalized_images[0]

# Generate and save heatmaps
for i, img in enumerate(normalized_images[1:], start=1):
    heatmap = generate_heatmap(reference_img, img)
    heatmap_path = os.path.join(output_directory, f'heatmap_{i}.jpg')
    cv2.imwrite(heatmap_path, heatmap)

# Convert images to LAB, flatten, and average colors
image_vectors = []
for img in normalized_images:
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_flat = img_lab.reshape(-1, 3)
    image_vectors.append(img_flat.mean(axis=0))

# Perform PCA
data = np.array(image_vectors)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_normalized)

# Plot PCA results
plt.figure(figsize=(10, 8))
for i, pc in enumerate(principal_components):
    plt.scatter(pc[0], pc[1], label=file_names[i])

# Place legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Image Colors')
plt.tight_layout()
plt.show()

# Plot PCA results and create custom legend with thumbnails
fig, ax = plt.subplots(figsize=(10, 8))
for i, pc in enumerate(principal_components):
    ax.scatter(pc[0], pc[1])

create_thumbnail_legend(ax, file_names, thumbnails)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Image Colors')
plt.tight_layout()
plt.show()


def wrap_text(text, char_limit):
    """Wrap text based on a character limit."""
    return '\n'.join(wrap(text, char_limit))

# Load thumbnails - Assuming load_thumbnail_key is defined elsewhere
thumbnails = [load_thumbnail_key(os.path.join(input_directory, f)) for f in file_names]

# Determine grid size
n_cols = 4  # Number of columns for the grid
n_rows = ceil(len(thumbnails) / n_cols)

# Create a figure with subplots in a grid
fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))  # Adjusted figsize for larger thumbnails
fig.suptitle('Image Thumbnails', fontsize=16)

# Handle case where axs might not be 2-dimensional
if n_rows == 1 or n_cols == 1:
    axs = axs.reshape(n_rows, n_cols)

# Add thumbnails to the grid
char_limit = 20  # Character limit for each line in labels

for i in range(n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    ax = axs[row, col]  # Correctly reference ax from axs

    if i < len(thumbnails):
        ax.imshow(thumbnails[i], aspect='auto')  # Display image
        wrapped_label = wrap_text(file_names[i], char_limit)
        ax.set_title(wrapped_label, fontsize=10)
    else:
        ax.axis('off')  # Turn off axis for empty subplots

    ax.axis('off')  # Ensure no axis lines or labels show

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
plt.show()