import os
import numpy as np
from skimage import io, color
from skimage.segmentation import felzenszwalb
import cv2
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
json_output = config['Paths']['json_output']
color_output = config['Paths']['color_output']


# Function to perform color segmentation and hue normalization
def recolorize_like_segmentation(image_path, mask_path, save_path):
    image = io.imread(image_path)
    mask = io.imread(mask_path)
    
    masked_image = np.zeros_like(image)
    masked_image[mask == 255] = image[mask == 255]
    
    segments = felzenszwalb(masked_image, scale=100, sigma=0.5, min_size=50)
    segmented_img = color.label2rgb(segments, image, kind='avg', bg_label=0)
    
    io.imsave(save_path, segmented_img)

# Function to normalize the color of an image
def normalize_color(image_path, save_path):
    img = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_norm = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(save_path, img_norm)

# Function to compute a 3D color histogram in LAB space and normalize it
def compute_color_histogram(image, bins=(8, 8, 8)):
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([img_lab], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to visualize the 3D color histogram as a cube
def visualize_color_histogram(hist, bins=(8, 8, 8)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    bin_centers = [np.linspace(0, 256, bin) for bin in bins]
    bin_centers = np.meshgrid(*bin_centers)

    ax.scatter(bin_centers[0].ravel(), bin_centers[1].ravel(), bin_centers[2].ravel(), c=hist.ravel(), s=100*hist.ravel())
    ax.set_xlabel('L')
    ax.set_ylabel('A')
    ax.set_zlabel('B')
    plt.show()

# Function to analyze the color patterns by computing 3D histograms and performing PCA
def analyze_color_patterns(roi_files, roi_dir):
    histograms = []
    for roi_file in roi_files:
        roi_path = os.path.join(roi_dir, roi_file)
        img = cv2.imread(roi_path)
        hist = compute_color_histogram(img)
        histograms.append(hist)

        # Optionally visualize each histogram
        visualize_color_histogram(hist)

    histograms = np.array(histograms)
    
    scaler = StandardScaler()
    histograms_normalized = scaler.fit_transform(histograms)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(histograms_normalized)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(principal_components[:, 0], principal_components[:, 1])

    # Add thumbnails to the PCA plot
    thumbnails = [remove_black_background(load_thumbnail(os.path.join(roi_dir, f))) for f in roi_files]
    create_thumbnail_legend(ax, roi_files, thumbnails, principal_components)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Color Histograms')
    plt.tight_layout()
    plt.show()

def load_thumbnail(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def remove_black_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        result = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(largest_contour)
        result = result[y:y+h, x:x+w]
        return result
    return image

def create_thumbnail_legend(ax, file_names, thumbnails, principal_components):
    for i, (file, thumbnail) in enumerate(zip(file_names, thumbnails)):
        imagebox = OffsetImage(thumbnail, zoom=0.02)
        ab = AnnotationBbox(imagebox, (principal_components[i, 0], principal_components[i, 1]),
                            xycoords='data', frameon=False)
        ax.add_artist(ab)

# Main execution

# Directories

input_dir = json_output
output_dir = color_output
os.makedirs(output_dir, exist_ok=True)

# Get all files in the input directory
all_files = os.listdir(input_dir)

# Filter binary mask and foreground mask files
binary_mask_files = [f for f in all_files if f.startswith('binary_mask_')]
foreground_mask_files = [f for f in all_files if f.startswith('foreground_mask')]

# Create a dictionary to map base names to their binary mask and foreground mask files
file_pairs = {}
for bm_file in binary_mask_files:
    base_name = bm_file[len('binary_mask_'):-len('_test.jpg')]
    file_pairs[base_name] = {'binary_mask': bm_file}

for fm_file in foreground_mask_files:
    base_name = fm_file[len('foreground_mask'):-len('_test.jpg')]
    if base_name in file_pairs:
        file_pairs[base_name]['foreground_mask'] = fm_file

# Process each pair of image and mask
for base_name, files in file_pairs.items():
    if 'binary_mask' in files and 'foreground_mask' in files:
        image_path = os.path.join(input_dir, files['foreground_mask'])
        mask_path = os.path.join(input_dir, files['binary_mask'])
        recolorized_path = os.path.join(output_dir, f"recolorized_{files['foreground_mask']}")
        
        # Perform recolorization
        recolorize_like_segmentation(image_path, mask_path, recolorized_path)
        
        # Perform hue normalization
        normalized_path = os.path.join(output_dir, f"normalized_{files['foreground_mask']}")
        normalize_color(recolorized_path, normalized_path)

# Analyze color patterns after processing
roi_files = [f for f in os.listdir(output_dir) if f.startswith('normalized_') and f.endswith('.jpg')]
analyze_color_patterns(roi_files, output_dir)

