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


# install dependencies
Once you have conda/miniconda and cuda installed you then need to install detectron2 and pytorch to match.  The version of cuda that works for your system, you will need to find out but it was built and tested on cuda 11.7.1 which is what I recommend
it may also work on cuda 10.2.89
it was run on a single Teslav100-32 GPU so a single GPU but with fairly beefy memory (32GB) it may use less mem but I have not tested it on smaller systems.

# Quick install via .yaml, installs everything accept for conda/miniconda and cuda/cudatoolkit

```shell 
conda env create -f detectron2-env.yaml
conda activate detectron2-env
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

to run the training first annotate some polygons with VIA2 found here:(https://www.robots.ox.ac.uk/~vgg/software/via/)
For the custom extraction of contours and subsequent semi-landmark collection
It can be installed via: 

