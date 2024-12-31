# Descriptron
A pipeline and code hub for dark taxa mophology using CNNs/ViTs and classic computer vision

Descriptron Version
v0.1.0

![lateral_6_prediction](https://github.com/user-attachments/assets/d14a5582-30d9-467e-acd8-e1f11db5edfd)
example of unfiltered output

# Introduction
Descriptron v0.1.0 is intended to automate geometric morphometrics of organismal morphological features. It combines the state-of-the-art CNN based instance segmentation of Detectron2 (https://github.com/facebookresearch/detectron2) implemented in PyTorch with classic computer vision implemented in OpenCV. The instance segmentation is via a CNN because that is what CNN's are best at and the precise pixel level data extraction is done using OpenCV as that is what classic computer vision is best at. I aim to use the correct tool for the job at hand. It is all written in python so it is more stable and all the programs play well together.

This work here helps move towards solving automated semi-landmarking of 2D data using Coleoptera sclerites as an example. This should be helpful to people interested in ground-truthing measurements for species delimitation using morphological shape and size as data in that delimitation process. This is a first version future versions will continue to expand the capabilities of computer vision and deep-learning for making geometrica and standard morphometric analyses more available and in a stable format for the community.

The long term goal of this project is to combine LLM ViTs with instance segmentation and image depth data to produce fine grained descriptions and concomitant semi-supervised and unsupervised clustering for putative novel species binning and description.

# Installation
The initial model is available through GoogleDrive found here:

https://drive.google.com/drive/folders/1MqcgrG3fYxsUIJ9ROnFx8jhLNSkp9f4D


# Citation for now a Bibtex of this page
@software{Van_Dam_Descriptron_2024,
  author = {Van Dam, Alex R.},
  title = {Descriptron},
  version = {main},
  url = {https://github.com/alexrvandam/Descriptron},
  date = {2024-12-31},
  note = {Accessed: 2024-12-31}
}

