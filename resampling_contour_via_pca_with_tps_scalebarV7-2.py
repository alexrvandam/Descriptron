import os
import fnmatch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.preprocessing import MinMaxScaler
import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths from config.ini
image_directory = config['Paths']['image_directory']
contours_input_folder = config['Paths']['json_output']  # Where contour .txt files are stored
output_folder = config['Paths']['model_output']  # Where results will be stored

# Define the required resampling functions

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

def resample_contour_points_fixed_first(contour, num_points):
    fixed_point = contour[0]
    remaining_points = contour[1:]
    cumulative_dist = np.cumsum(np.sqrt(np.sum(np.diff(remaining_points, axis=0)**2, axis=1)))
    cumulative_dist = np.insert(cumulative_dist, 0, 0)
    fx = interpolate.interp1d(cumulative_dist, remaining_points[:, 0], kind='linear', fill_value="extrapolate")
    fy = interpolate.interp1d(cumulative_dist, remaining_points[:, 1], kind='linear', fill_value="extrapolate")
    new_cumulative_dist = np.linspace(0, cumulative_dist[-1], num_points - 1)
    new_x = fx(new_cumulative_dist)
    new_y = fy(new_cumulative_dist)
    resampled_points = np.column_stack((new_x, new_y))
    return np.vstack((fixed_point, resampled_points))

def align_contours_to_reference(reference_contour, contours):
    pca = PCA(n_components=2)
    ref_orientation = pca.fit(reference_contour).components_[0]
    
    aligned_contours = []
    for contour in contours:
        orientation = pca.fit(contour).components_[0]
        angle = np.arccos(np.dot(ref_orientation, orientation))
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        rotated_contour = np.dot(contour - np.mean(contour, axis=0), rotation_matrix) + np.mean(reference_contour, axis=0)
        aligned_contours.append(rotated_contour)
    return aligned_contours

def procrustes_analysis(reference_contour, contours):
    transformed_contours = []
    for contour in contours:
        _, mtx2, _ = procrustes(reference_contour, contour)
        transformed_contours.append(mtx2)
    return transformed_contours

def calculate_length_height(contour):
    # Ensure contour is a 2D array of shape (N, 2)
    if contour.ndim != 2 or contour.shape[1] != 2:
        contour = contour.reshape(-1, 2)
    
    # Perform PCA on the contour points
    pca = PCA(n_components=2)
    pca.fit(contour)
    
    # Get the principal components
    pc1, pc2 = pca.components_
    
    # Project the contour points onto the principal components
    projections_pc1 = np.dot(contour, pc1)
    projections_pc2 = np.dot(contour, pc2)
    
    # Calculate the length and height based on the projections
    length = projections_pc1.max() - projections_pc1.min()
    height = projections_pc2.max() - projections_pc2.min()
    
    return length, height

def visualize_length_height(image, contour, length, height):
    # Ensure contour is a 2D array of shape (N, 2)
    if contour.ndim != 2 or contour.shape[1] != 2:
        contour = contour.reshape(-1, 2)
    
    # Calculate bounding box
    min_x, min_y = contour.min(axis=0)
    max_x, max_y = contour.max(axis=0)
    
    # Draw the original image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw the contour
    plt.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2)
    
    # Draw the max height and length
    plt.plot([min_x, max_x], [min_y, min_y], 'r-', linewidth=2)  # Length
    plt.plot([min_x, min_x], [min_y, max_y], 'g-', linewidth=2)  # Height
    
    plt.title(f"Length: {length:.2f} px, Height: {height:.2f} px")
    plt.show()

def process_image(image_path, contour_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Load the contour from the .txt file
    contour = np.loadtxt(contour_path, delimiter=',')
    
    # Calculate length and height
    length, height = calculate_length_height(contour)
    
    # Visualize length and height on the contour
    visualize_length_height(image, contour, length, height)
    
    return contour, length, height

def process_contours(contours_input_dir, output_dir):
    contour_files = [f for f in os.listdir(contours_input_dir) if fnmatch.fnmatch(f, 'contour_*_*.txt')]
    contours = []
    for filename in contour_files:
        base_name = filename.replace('contour_', '').replace('.txt', '')
        image_path = os.path.join(image_directory, f"{base_name}.jpg")
        contour_path = os.path.join(contours_input_dir, filename)
        
        if os.path.exists(image_path):
            contour, length, height = process_image(image_path, contour_path)
            contours.append(contour)
            
            # Load scale bar data if available
            image_result_file = os.path.join(output_dir, base_name + '_scalebar.txt')
            if os.path.exists(image_result_file):
                with open(image_result_file, 'r') as f:
                    lines = f.readlines()
                    pixels_per_mm = float(lines[3].strip().split(': ')[1])
                    metrics = calculate_metrics(contour, pixels_per_mm)
                    save_contour_metrics(base_name, metrics, output_dir)
                    
                    # Save the resampled contours for further analysis
                    resample_and_save_contours(image_directory, contours_input_dir, output_dir)
            else:
                print(f"Scale bar file not found for {base_name}")
        else:
            print(f"Image file not found for contour: {filename}")
    
    return contours

# Function to calculate ROI metrics
def calculate_metrics(contour, pixels_per_mm):
    if contour.shape[0] < 3:
        print("Contour has less than 3 points, skipping metrics calculation.")
        return {}

    length, height = calculate_length_height(contour)
    length_to_height_ratio = length / height if height != 0 else None
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    aspect_ratio = float(length) / height if height != 0 else None
    extent = area / (length * height) if length * height != 0 else None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area != 0 else None
    equivalent_diameter = np.sqrt(4 * area / np.pi)
    
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour) if len(contour) >= 5 else ((None, None), (None, None), None)
    
    metrics = {
        "length_pixels": length,
        "height_pixels": height,
        "length_to_height_ratio": length_to_height_ratio,
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "equivalent_diameter_pixels": equivalent_diameter,
        "major_axis_length_pixels": MA,
        "minor_axis_length_pixels": ma,
        "orientation_degrees": angle,
        "length_um": length / pixels_per_mm * 1000 if pixels_per_mm else None,
        "height_um": height / pixels_per_mm * 1000 if pixels_per_mm else None,
        "area_um2": area / (pixels_per_mm ** 2) * 1e6 if pixels_per_mm else None,
        "perimeter_um": perimeter / pixels_per_mm * 1000 if pixels_per_mm else None,
        "equivalent_diameter_um": equivalent_diameter / pixels_per_mm * 1000 if pixels_per_mm else None,
        "major_axis_length_um": MA / pixels_per_mm * 1000 if MA and pixels_per_mm else None,
        "minor_axis_length_um": ma / pixels_per_mm * 1000 if ma and pixels_per_mm else None
    }
    return metrics

# Function to save contour metrics to a file
def save_contour_metrics(base_name, metrics, output_dir):
    if not metrics:
        print(f"No metrics calculated for {base_name}, skipping saving.")
        return

    contour_metrics_file = os.path.join(output_dir, base_name + '_contour_metrics.txt')
    with open(contour_metrics_file, 'w') as f:
        f.write(f"Length in pixels: {metrics['length_pixels']}\n")
        f.write(f"Height in pixels: {metrics['height_pixels']}\n")
        f.write(f"Length to height ratio: {metrics['length_to_height_ratio']}\n")
        f.write(f"Area in pixels: {metrics['area_pixels']}\n")
        f.write(f"Perimeter in pixels: {metrics['perimeter_pixels']}\n")
        f.write(f"Aspect ratio: {metrics['aspect_ratio']}\n")
        f.write(f"Extent: {metrics['extent']}\n")
        f.write(f"Solidity: {metrics['solidity']}\n")
        f.write(f"Equivalent diameter in pixels: {metrics['equivalent_diameter_pixels']}\n")
        f.write(f"Orientation in degrees: {metrics['orientation_degrees']}\n")
        f.write(f"Major axis length in pixels: {metrics['major_axis_length_pixels']}\n")
        f.write(f"Minor axis length in pixels: {metrics['minor_axis_length_pixels']}\n")
        f.write(f"Length in um: {metrics['length_um']}\n")
        f.write(f"Height in um: {metrics['height_um']}\n")
        f.write(f"Area in um^2: {metrics['area_um2']}\n")
        f.write(f"Perimeter in um: {metrics['perimeter_um']}\n")
        f.write(f"Equivalent diameter in um: {metrics['equivalent_diameter_um']}\n")
        f.write(f"Major axis length in um: {metrics['major_axis_length_um']}\n")
        f.write(f"Minor axis length in um: {metrics['minor_axis_length_um']}\n")
    print(f"Contour metrics saved to: {contour_metrics_file}")

# Function to resample and save contours
def resample_and_save_contours(input_folder, contours_input_folder, output_folder):
    contour_files = [f for f in os.listdir(contours_input_folder) if fnmatch.fnmatch(f, 'contour_*_*.txt')]
    if not contour_files:
        print("No contour files found.")
        return

    contours = []
    for filename in contour_files:
        with open(os.path.join(contours_input_folder, filename), 'r') as file:
            contour_data = []
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        x, y = map(float, stripped_line.split(','))
                        contour_data.append([x, y])
                    except ValueError:
                        print(f"Skipping line in file {filename} due to formatting error: {line}")
                        continue
            if contour_data:
                contours.append(np.array(contour_data))
            else:
                print(f"No valid data in contour file: {filename}")

    if not contours:
        print("No valid contours found after processing files.")
        return

    # Normalize and resample all contours to 800 points
    normalized_contours = [resample_contour_points_fixed_first(contour, 800) for contour in contours]

    # Check if there are any normalized contours
    if not normalized_contours:
        print("No normalized contours available.")
        return

    # Align contours based on the first normalized and reordered contour
    reference_contour = order_points_clockwise(normalized_contours[0])
    aligned_contours = align_contours_to_reference(reference_contour, normalized_contours)

    # Reorder all aligned contours
    ordered_contours = [order_points_clockwise(contour) for contour in aligned_contours]

    # Resample all ordered contours to distribute points evenly along the contour
    resampled_ordered_contours = [resample_contour_points_fixed_first(contour, 800) for contour in ordered_contours]

    # Perform Procrustes analysis on the resampled ordered contours
    procrustes_contours = procrustes_analysis(resampled_ordered_contours[0], resampled_ordered_contours)

    # Save the resampled ordered contours with a new suffix
    for contour, filename in zip(resampled_ordered_contours, contour_files):
        new_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_pca_oriented_contours.csv")
        np.savetxt(new_filename, contour, delimiter=",")

    # Save the contours as a TPS file
    tps_output_file = os.path.join(output_folder, "pca_oriented_contours.tps")
    save_contours_as_tps(resampled_ordered_contours, contour_files, tps_output_file)

    # Save the contours as CSV files
    save_contours_as_csv(resampled_ordered_contours, contour_files, output_folder)

    # Save the procrustes contours as CSV files
    save_procrustes_contours_as_csv(procrustes_contours, contour_files, output_folder)

    # Save the contours as npy files
    save_contours_as_npy(resampled_ordered_contours, contour_files, output_folder, 'pca_oriented_contours')
    save_contours_as_npy(procrustes_contours, contour_files, output_folder, 'procrustes_contours')

    # Save vectorized PCA-oriented contours
    save_vectorized_contours_as_npy(resampled_ordered_contours, output_folder, 'vectorized_pca_oriented_contours.npy')

    # Save vectorized Procrustes-aligned contours
    save_vectorized_contours_as_npy(procrustes_contours, output_folder, 'vectorized_procrustes_contours.npy')

    # Visualization check
    visualize_contours(resampled_ordered_contours, "Resampled Ordered Contours")
    visualize_contours(procrustes_contours, "Procrustes Aligned Contours")

def visualize_contours(contours, title):
    plt.figure(figsize=(10, 10))
    for contour in contours:
        plt.plot(contour[:, 0], contour[:, 1], '-o')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example usage

# Process contours and save metrics
process_contours(contours_input_folder, output_folder)
