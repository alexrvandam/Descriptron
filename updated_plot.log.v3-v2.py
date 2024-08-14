import configparser
import re
import matplotlib.pyplot as plt

# Load the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access the paths
train_data = config['Paths']['train_data']
val_data = config['Paths']['val_data']
test_data = config['Paths']['test_data']
model_output = config['Paths']['model_output']
plot_output = config['Paths']['plot_output']
json_input = config['Paths']['json_input']
json_output = config['Paths']['json_output']
log_file_path = config['Paths']['log_file_path']  # Load the log file path from config.ini

# Regular expressions to match the required metrics
loss_pattern = re.compile(r"total_loss: ([0-9.]+)")
ap_pattern = re.compile(r"copypaste: ([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+),([0-9.]+)")
iteration_pattern = re.compile(r"iter: (\d+)")

# Lists to store the extracted metrics
iterations = []
losses = []
ap50_scores = []
ap_scores = []

# Flags to control parsing
collect_ap = False

# Parse the log file
with open(log_file_path, "r") as file:
    for line in file:
        # Extract iteration
        iter_match = iteration_pattern.search(line)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            if collect_ap:
                iterations.append(current_iteration)
                collect_ap = False

        # Extract loss
        loss_match = loss_pattern.search(line)
        if loss_match:
            losses.append(float(loss_match.group(1)))

        # Check for AP lines
        if "copypaste: Task: bbox" in line:
            collect_ap = True
            continue

        # Extract AP and AP50 scores
        if collect_ap:
            ap_match = ap_pattern.search(line)
            if ap_match:
                ap_scores.append(float(ap_match.group(1)))
                ap50_scores.append(float(ap_match.group(2)))

# Ensure that the lists are of equal length
min_length = min(len(iterations), len(losses), len(ap50_scores), len(ap_scores))
iterations = iterations[:min_length]
losses = losses[:min_length]
ap50_scores = ap50_scores[:min_length]
ap_scores = ap_scores[:min_length]

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(iterations, losses, label='Total Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss over Iterations')
plt.legend()
plt.grid()
plt.savefig('training_loss_curve_v3.png')
plt.show()

# Plot Validation AP50 Scores
plt.figure(figsize=(10, 5))
plt.plot(iterations, ap50_scores, label='Validation AP50 (bbox)', color='orange')
plt.xlabel('Iteration')
plt.ylabel('AP50')
plt.title('Validation AP50 (bbox) over Iterations')
plt.legend()
plt.grid()
plt.savefig('validation_ap50_bbox_curve.png')
plt.show()

# Plot Validation AP Scores
plt.figure(figsize=(10, 5))
plt.plot(iterations, ap_scores, label='Validation AP (bbox)', color='green')
plt.xlabel('Iteration')
plt.ylabel('AP')
plt.title('Validation AP (bbox) over Iterations')
plt.legend()
plt.grid()
plt.savefig('validation_ap_bbox_curve.png')
plt.show()
