# Descriptron
A pipeline and model zoo for dark taxa mophology using CNNs/ViTs and classic computer vision

Descriptron Version
v0.1.0

# Introduction
Descriptron v0.1.0 is intended to automate geometric morphometrics of organismal morphological features. It combines the state-of-the-art CNN based instance segmentation of Detectron2 (https://github.com/facebookresearch/detectron2) implemented in PyTorch with classic computer vision implemented in OpenCV. The instance segmentation is via a CNN because that is what CNN's are best at and the precise pixel level data extraction is done using OpenCV as that is what classic computer vision is best at. I aim to use the correct tool for the job at hand. It is all written in python so it is more stable and all the programs play well together.

# Installation
The initial model is available through GoogleDrive found here:

You will need conda or miniconda3 (https://docs.anaconda.com/miniconda/)

You need to make sure that your version of pytorch (https://pytorch.org/get-started/previous-versions/) and cuda (https://developer.nvidia.com/cuda-toolkit-archive) compatible with the detectron2 version you want to install if you don't want to use the requirements.txt or the .yaml provide.


# Dependencies
Once you have conda/miniconda and cuda installed you then need to install detectron2 and pytorch to match.  The version of cuda that works for your system, you will need to find out but it was built and tested on cuda 11.7.1 which is what I recommend
it may also work on cuda 10.2.89
it was run on a single Teslav100-32 GPU so a single GPU but with fairly beefy memory (32GB) it may use less mem but I have not tested it on smaller systems.

# Quick install via .yaml, installs everything accept for conda/miniconda and cuda/cudatoolkit

```shell 
conda env create -f descriptron-train-predict.yaml
conda activate descriptron-tp
```
If you want to configure to your own environment without making a new one (not recommended) then the minimum packages some of which are
some of these are default in a conda environment with python but I list them all 

detectron2
os
sys
json
cuda
numpy
opencv
random
shapely
warnings
pandas
torch
matplotlib

you can look in the .yaml file to see the versions I used and if there is anything compatable on your system, so probbly the easiest way to do this is just via the .yaml file provided to do the training and prediction.

# Image Data Annotation
to run the training first annotate some polygons with VIA2 found here:(https://www.robots.ox.ac.uk/~vgg/software/via/)
For the custom extraction of contours and subsequent semi-landmark collection. 

Use the polygon tool to make your annotations and the dropdown menu to make your classes, you can follow the bare_base.json file as a template.

Export as a plain .json file of polygon contours.

The .json file is how you define your classes and what you want to train.

# Configuration file setup
Then configure the configuration file with the file paths where your input and output data is going to be.

```shell
[Paths]
train_data = /vandam/insect_images/coleoptera/coleoptera_train  # Image directory for training
val_data = /vandam/insect_images/coleoptera/coleoptera_val  # Image directory for validation
test_data = /vandam/insect_images/coleoptera/big_training # main root dir of test data big_training/view/family/image.jpg
model_output = /vandam/insect_images/coleoptera/output  # Output directory for model, config, etc.
plot_output = /vandam/insect_images/coleoptera/plots  # Directory for saving plots
train_json_file_path = /vandam/insect_images/coleoptera/via_project_13Jun2024_16h38m_curculionidae_json.json  # Handcrafted polygons JSON
json_input = /vandam/insect_images/coleoptera/pred_output/inference_results.json  # Inference results JSON (output)
json_output = /vandam/insect_images/coleoptera/pred_output  # Parent folder for inference results
coco_json_path = /vandam/insect_images/coleoptera/coco_format_rn50_5kstp.json  # COCO format JSON path
log_file_path = ./detectron2_rn50_trainstp.18.6.24.out  # Log file from training here it is coded as your current working directory
metadata_json_path = /vandam/insect_images/coleoptera/output/metadata_big35kv4_rn50.json  # Metadata JSON from training
image_directory = /vandam/insect_images/coleoptera/big_training/lateral/Curculionidae  # Use the same path as test_data but a specific subdirectory where the images are housed
color_output = /vandam/insect_images/coleoptera/pred_output/color_output # output from color analyses
```

You will need to pay attention to this file carefully if the scripts do not work for you the most likely source of the erro ris not having the file system set up correctly in the config file for the input and output files you need to be read correctly by the scripts, you will need to modify the config and possibly the config related section of the scripts to work on your file system and how your data is structured if you encounter significant number of errors. I am working on setting this up for future versions in a more dynamic way so the user can specify the input and output at each step via command options but for now a config file will have to do so I can make these scripts available to the community.  

Next simply run the trainer.

```shell
python res50_train_stop5kv7.py
```
you can examine the log file if you and then plot the AP scores to examine for overfitting

```shell
python plot.log.v3
```

Then if the training looks good simply run the predictor on your test data set.

```shell
python contour_pred_rn50_v9_part2.py
```




