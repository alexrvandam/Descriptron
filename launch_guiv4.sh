#!/bin/bash

# Define paths
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_DIR="$PROJECT_DIR/conda"
GUI_SCRIPT="$PROJECT_DIR/segment-anything-2/gui/descriptron-label-206.py"

# Function to log messages
log() {
    echo -e "\e[34m[INFO]\e[0m $1"
}

# Check if the Conda installation exists
if [ -d "$CONDA_DIR" ]; then
    # Source Conda
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo -e "\e[31m[ERROR]\e[0m Conda not found at '$CONDA_DIR'."
    exit 1
fi

# Set your OpenAI API linux and mac
export OPENAI_API_KEY="your_api_key_here"

# Set your OpenAI API Windows
# setx OPENAI_API_KEY "your_api_key_here"

# Activate the 'samm' environment
conda activate samm

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo -e "\e[31m[ERROR]\e[0m Failed to activate 'samm' environment."
    exit 1
fi

# Debugging: Print Conda info
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Conda executable: $(which conda)"
echo "Python executable: $(which python)"

# Run the GUI script using Python
cd "$PROJECT_DIR/segment-anything-2/gui/"
python "$GUI_SCRIPT"

# Deactivate Conda environment after GUI is closed
conda deactivate
