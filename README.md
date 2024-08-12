# Descriptron
A pipeline and model zoo for dart taxa mophology using CNNs/ViTs and classic computer vision

Descriptron Version
v0.1.0

# Introduction
Descriptron v0.1.0 is intended to automate geometric morphometrics of organismal morphological features. It combines the state-of-the-art CNN based instance segmentation of Detectron2 implemented in PyTorch with classic computer vision implemented in OpenCV. The instance segmentation is via a CNN because that is what CNN's are best at and the precise pixel level data extraction is done using OpenCV as that is what classic computer vision is best at. I aim to use the correct tool for the job at hand. It is all written in python so it is more stable and all the programs play well together.

# Installation
The initial model is available through GoogleDrive found here:

You will need to make sure that your version of pytorch and cuda are compatible with the detectron2 version you want to install if you don't want to use the requirements.txt or the .yaml provide.

you will need conda or miniconda3 

# install dependencies

# install via .yaml

```shell 
conda env create -f detectron2-env.yaml
conda activate detectron2-env
```
# as an alternative install with requirements.txt 
you can see if it will work in your environment if you have pip or install pip just skip the conda create statment

```shell
cd descriptron
conda env create -n detectron2-env python=3.9.15
conda activate detectron2-env
pip install -r requirements.txt
```


The python environment can be found here:

It can be installed via: 

