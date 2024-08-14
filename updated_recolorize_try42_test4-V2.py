import cv2
import numpy as np
import os
import fnmatch
from scipy import interpolate
from scipy.spatial import procrustes
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
json_output = config['Paths']['json_output']
color_output = config['Paths']['color_output']



def find_lowest_rightmost_point(contour):
    return contour[np.lexsort((-contour[:, 1], contour[:, 0]))][0]

def order_points_clockwise(contour):
    mean = np.mean(contour, axis=0)
    centered = contour - mean
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    sorted_contour = contour[np.argsort(angles)]
    start_idx = np.where(np.all(sorted_contour == find_lowest_rightmost_point(contour), axis=1))[0]
    ordered_contour = np.roll(sorted_contour, -start_idx[0], axis=0)
    return ordered_contour

def resample_contour_points(contour, num_points):
    cumulative_dist = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    cumulative_dist = np.insert(cumulative_dist, 0, 0)
    fx = interpolate.interp1d(cumulative_dist, contour[:, 0])
    fy = interpolate.interp1d(cumulative_dist, contour[:, 1])
    new_cumulative_dist = np.linspace(0, cumulative_dist[-1], num_points)
    new_x = fx(new_cumulative_dist)
    new_y = fy(new_cumulative_dist)
    return np.column_stack((new_x, new_y))

def normalize_contours(contours):
    normalized_contours = []
    for contour in contours:
        # Translate the contour to the origin
        mean_x = np.mean(contour[:, 0])
        mean_y = np.mean(contour[:, 1])
        contour -= [mean_x, mean_y]
        
        # Scale the contour to unit size
        scale = np.sqrt((contour ** 2).sum())
        contour /= scale
        
        normalized_contours.append(contour)
    
    return normalized_contours


def recolorize_like_segmentation(image_path, save_path):
    # Read the image file
    image = io.imread(image_path)
    # Convert the image to LAB color space
    #image_lab = color.rgb2lab(image)
    # Normalize the L channel to the range [0, 1]
    #image_lab[:, :, 0] /= 100
    # Perform Felzenszwalb segmentation
    segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
    # Recolorize the image based on segmentation
    segmented_img = color.label2rgb(segments, image, kind='avg', bg_label=0)
    # Convert back to RGB to save the image
    #segmented_img_rgb = (segmented_img * 255).astype('uint8')
    # Save the segmented image
    # Save the segmented image
    io.imsave(save_path, segmented_img)
    return segmented_img

# Normalize color
#img_norm = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#img_norm[:, :, 0] = clahe.apply(img_norm[:, :, 0])
#img_norm = cv2.cvtColor(img_norm, cv2.COLOR_LAB2BGR)

#def recolorize_like_segmentation(image):
#    segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
#    segmented_img = color.label2rgb(segments, image, kind='avg', bg_label=0)
#    return segmented_img



# Directory paths
contours_dir = json_output
original_image_dir = image_directory
resampled_dir = json_output
transformed_contours_dir = json_output



# Process each contour and image pair
# Initialize lists to hold contours
resampled_contours = []
normalized_contours = []
transformed_contours = []

# Read file names
contour_files = sorted([f for f in os.listdir(contours_dir) if fnmatch.fnmatch(f, 'binary_mask_*_contours.txt')])
original_image_files = sorted([f for f in os.listdir(original_image_dir) if fnmatch.fnmatch(f, 'foreground_mask*.jpg')])




# Ensure the file lists are matched correctly
assert len(contour_files) == len(original_image_files), "The number of contour files and image files should be the same"




# Process each contour and image pair
for i, filename in enumerate(contour_files):
    contour_path = os.path.join(contours_dir, filename)
    image_name = original_image_files[i]
    image_path = os.path.join(original_image_dir, image_name)

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or path is incorrect for {image_name}")

    # Load and process the contour
    with open(contour_path, 'r') as file:
        contour = np.array([[float(x) for x in line.strip().split(',')] for line in file if line.strip()])
    ordered_contour = order_points_clockwise(contour)
    resampled_contour = resample_contour_points(ordered_contour, 800)

    # Draw the resampled contour on the image
    cv2.drawContours(img, [resampled_contour.astype(np.int32)], -1, (0, 255, 0), 2)

    # Save the resampled contour coordinates to a file
    resampled_contour_path = os.path.join(resampled_dir, f'resampled_{filename}')
    np.savetxt(resampled_contour_path, resampled_contour, fmt='%f,%f')

    # Show the image with the contour
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(resampled_contour[:, 0], resampled_contour[:, 1], 'r-') # use 'ro' for dots
    plt.title('Resampled Contour on Image')
    plt.show()

    # Optionally save the image with the contour drawn on it
    image_with_contour_path = os.path.join(resampled_dir, f'image_with_resampled_contour_{image_name}')
    cv2.imwrite(image_with_contour_path, img)
    # Normalize contour
    normalized_contour = normalize_contours([resampled_contour])[0]
    
    # Perform Procrustes analysis using the first contour as a reference
    if i == 0:
        reference_contour = normalized_contour
    _, transformed_contour, _ = procrustes(reference_contour, normalized_contour)
    
    # Adjust transformed contour to fit the image dimensions
    transformed_contour -= np.min(transformed_contour, axis=0)
    scales = np.array(img.shape[1::-1]) / np.ptp(transformed_contour, axis=0)
    scale = min(scales)
    transformed_contour *= scale
    transformed_contour += (np.array(img.shape[1::-1]) - np.max(transformed_contour, axis=0)) / 2

    # Create and save the ROI based on the transformed contour
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(transformed_contour)], 255)
    roi = cv2.bitwise_and(img, img, mask=mask)
    roi_path = os.path.join(resampled_dir, f'roi_{image_name}')
    cv2.imwrite(roi_path, roi)

    # Save the transformed contour
    transformed_contour_path = os.path.join(transformed_contours_dir, f'transformed_{filename}')
    np.savetxt(transformed_contour_path, transformed_contour, fmt='%f,%f')

    # Optionally, visualize the original and transformed contours on the image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(resampled_contour[:, 0], resampled_contour[:, 1], 'r-', linewidth=2)
    plt.title('Original Contour')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(transformed_contour[:, 0], transformed_contour[:, 1], 'g-', linewidth=2)
    plt.title('Transformed Contour')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Assuming the resampled and transformed contour files are saved in the 'resampled_dir' and 'transformed_contours_dir' directories respectively.
resampled_contour_filenames = sorted(fnmatch.filter(os.listdir(resampled_dir), 'resampled_binary_mask_*.txt'))
transformed_contour_filenames = sorted(fnmatch.filter(os.listdir(transformed_contours_dir), 'transformed_binary_mask_*.txt'))

resampled_contours = []
transformed_contours = []

# Check if the number of files match
assert len(resampled_contour_filenames) == len(transformed_contour_filenames), "Number of resampled and transformed contour files do not match."

# Load the contour data from the files
for resampled_filename, transformed_filename in zip(resampled_contour_filenames, transformed_contour_filenames):
    resampled_path = os.path.join(resampled_dir, resampled_filename)
    transformed_path = os.path.join(transformed_contours_dir, transformed_filename)
    
    # Load the resampled contour
    resampled_contour = np.loadtxt(resampled_path, delimiter=',')
    resampled_contours.append(resampled_contour)

    # Load the transformed contour
    transformed_contour = np.loadtxt(transformed_path, delimiter=',')
    transformed_contours.append(transformed_contour)

print(f"Entering the loop with {len(resampled_contours)} resampled contours and {len(transformed_contours)} transformed contours.")
if len(resampled_contours) != len(transformed_contours):
    print("The number of resampled and transformed contours does not match.")
    # Handle the error appropriately


for i, (original_contour, transformed_contour) in enumerate(zip(resampled_contours, transformed_contours)):
    print(f"Processing {i+1}/{len(resampled_contours)}: {original_image_files[i]}")  # This should print

    # Ensure the image path is valid
    original_image_path = os.path.join(original_image_dir, original_image_files[i])
    if not os.path.exists(original_image_path):
        print(f"Image path does not exist: {original_image_path}")
        continue
       
    # Read the original image
    original_image_path = os.path.join(original_image_dir, original_image_files[i])
    img = cv2.imread(original_image_path)
    
    # Check if the image is loaded
    if img is None:
        print(f"Image not found or path is incorrect for {original_image_files[i]}")
        continue
    
    # Select three points from the contours for the affine transformation
    pts1_indices = [200, 400, 600]  # Update this line if it's outside the loop in your original code
    if len(original_contour) < max(pts1_indices) + 1 or len(transformed_contour) < max(pts1_indices) + 1:
        print(f"Not enough points in contour to perform affine transformation for {original_image_files[i]}")
        continue
    
    pts1 = np.float32([original_contour[idx] for idx in pts1_indices])
    pts2 = np.float32([transformed_contour[idx] for idx in pts1_indices])

    # Compute the affine matrix using the selected points
    matrix = cv2.getAffineTransform(pts1, pts2)

    # Apply affine transformation to the entire image
    transformed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

    print("Affine transformation applied.")

    # Extract ROI using the transformed contour
    mask = np.zeros(transformed_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(transformed_contour)], 255)
    roi = cv2.bitwise_and(transformed_img, transformed_img, mask=mask)

    # Save or display the ROI
    roi_path = os.path.join(resampled_dir, f'roi_{original_image_files[i]}')
    cv2.imwrite(roi_path, roi)
    print(f"ROI saved to {roi_path}")


# Assuming 'roi_dir' is the directory containing the ROI images
roi_dir = json_output
save_dir = json_output


# Get all the ROI files
roi_files = [f for f in os.listdir(roi_dir) if f.startswith('roi_') and f.endswith('.jpg')]

# Loop through the ROI files and perform the recolorize like segmentation
for roi_file in roi_files:
    roi_path = os.path.join(roi_dir, roi_file)
    save_path = os.path.join(save_dir, f'segmented_{roi_file}')
    recolorize_like_segmentation(roi_path, save_path)
    print(f'Segmented image saved to: {save_path}')

print('All images processed.')



    # Show the transformed image with marked points using OpenCV
    #for point in pts1:
    #    cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)  # Draw original points in red on the original image
    #for point in pts2:
    #    cv2.circle(transformed_img, tuple(point), 5, (0, 255, 255), -1)  # Draw transformed points in yellow on the transformed image

    #cv2.imshow('Original Image with Points', img)
    #cv2.imshow('Transformed Image with Points', transformed_img)
    #cv2.waitKey(0)  # Wait for a key press to close the windows
    #cv2.destroyAllWindows()

    #print("Images displayed.")