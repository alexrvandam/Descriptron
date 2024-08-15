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
The data structure is like so:
```shell
/your_folder/insect_images/coleoptera/big_training/Family/view_imageID_.jpg
/your_folder/insect_images/coleoptera/big_training/Curculionidae/dorsal_1234ID_.jpg
/your_folder/insect_images/coleoptera/big_training/Curculionidae/lateral_1234ID_.jpg
/your_folder/insect_images/coleoptera/big_training/Curculionidae/lateral_9375ID_.jpg
.......
```
Please sort your images in advance by view hopefully you have some notes on their orientation, at the moment the two views the model was trained on was dorsal and later, but you can add as many views as you want and fine-tune a new model and post to the git community here or on hugging face page that would be wonderful. The ideas is that as a community we slowly build up more and more morphological features for fine grained instance segmentation for ground-truthed measurements of shape, size, etc.

If you do not want to pre-sort view I have an experimental script that might work for soting automatically by cardinal views, dorsal, lateral, ventral, frontal, posterior for Coleoptera only it is in the View folder. It is not thoroughly tested but it may work for you if you use it please cite this page. A more genral arthropoda view sorting script as part of Descriptron will be produced in future versions.

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

Careful attention is needed at this step, if the scripts do not work for you the most likely source of the erro ris not having the file system set up correctly in the config file for the input and output files you need to be read correctly by the scripts, you will need to modify the config and possibly the config related section of the scripts to work on your file system and how your data is structured if you encounter significant number of errors. I am working on setting this up for future versions in a more dynamic way so the user can specify the input and output at each step via command options but for now a config file will have to do so I can make these scripts available to the community. If you are really struggling to get these to work please write to me I can try to help solve the issue but please try first.

# Model Training

Next simply run the trainer. Remove the last "&" if you do not want it to run in the background.

```shell
python res50_train_stop5kv7.py > res50_train_output.log 2> error.re50.train.log &
```
Examine the log file if you and then plot the AP scores to examine for overfitting

# Plot the loss and AP scores and inspect for over-fitting and model performance
```shell
python plot.log.v3
```
![training_loss_curve_v3](https://github.com/user-attachments/assets/d7fe61f0-9029-42ef-9507-7bfc1e2beabe)
![validation_ap50_bbox_curve](https://github.com/user-attachments/assets/8ccdd14e-792e-47a1-9259-987aad833bdf)
![validation_ap_bbox_curve](https://github.com/user-attachments/assets/101643c1-918a-48e9-8d06-e311e5fb3209)

Then if the training looks good simply run the predictor on your test data set.

```shell
python contour_pred_rn50_v9_part2.py > prediction_output.log 2> prediction_error_.log &
```

It is optional but you can plot the overall percent success and possibly overprediction of the bbox detections and numbers as well as generate some summary statistics via another plotting script, as well as to visualize for a very crude quantitative assessment. It unfortunatley requires some hard-coded file paths at the momenent and you need to change directory into where your output .json predictions are to run it. To do this change line 77 from

```shell
image_counts = count_images_by_family_and_view('/vandam/insect_images/coleoptera/big_training/', ["dorsal", "lateral", "frontal"])
```
to your own file path of the parent directory for your images that you use for prediction where the file below big training contains family names eg. "Lucanidae" and then inside of that folder the files have the view followed by the imageID eg. "dorsal_987987ID_.jpg" this was just an example but it is basically going to look for the files to be two levels down.

```shell
image_counts = count_images_by_family_and_view('/your/parent/directory/coleoptera/big_training/', ["dorsal", "lateral", "frontal"])
```
once you have that accomplished then run the script in the folder where your predicted sclerites or predictions are eg. in the same folder as your inference_results.json. It should give you results similar to those below but hopefully with better scores.

```shell
cd /your/predictions/pre_output/
python updated_extract_json_stats8.py
```

![heatmap_percentages_lateral](https://github.com/user-attachments/assets/f87655fb-bc14-488b-aa55-7fa4acae9876)

It might also be a good idea to get a qualitative assessment for how good or not the predictions are this can be done by another graphing script that shows thumbnails of the images in a grid layout. 

```shell
python updated_create_overlays-V6.py
```
![overlayed_images_lateral_reduced](https://github.com/user-attachments/assets/10b7c9e3-cc07-4db2-a365-ffb00fba82a4)

# Automated extraction of semi-landmark data from 2D images

Ok now that you have a .json file of predicted contours you can now extract the contours as you like and or also extract binary and foreground masks of the image that 'mask' or cover over all the other parts of the image you do not want. If the config file was setup correctly then by running the 

To get specific sclerites from a single folder and return the binary and foreground mask as well as simple xy coordinates and a .npy file of the contours run simply add or remove the classes you want on line 48 desired_classes from file "updated_get_masks_from_contours_multi-V2-4.py", it should return the images and files in subdirectories by sclerite you need to decide on the view in the script in this case the two views returned are lateral and dorsal and on the specified folder via the config, below is the line that specifies the view if you have more or less change this line

```python
    relevant_files = [os.path.basename(f) for f in os.listdir(image_dir) if f.startswith(('lateral', 'dorsal')) and f.endswith(('.jpg', '.jpeg', '.png'))]To return all the sclerites
````
and finally run the script

```shell
python updated_get_masks_from_contours_multi-V2-4.py
```
The previous output should have generated some simple xy coordinates this is the input for the next script, the following script turns the contours into semi-landmakrs. It will in this order, align them via principal component PC, redistribute the points evenly along the contour given a number of the users choosing 800 in this case, and then re-order the numbering based on a clockwise coordinate rule based on the first file with the others following a similar numbering scheme to preserve homology. It will export the new semi-landmark contours in a variety of formats some might be useful for your project more than others you decide. If you want fewer or more points change the 800 on the following lines to whatever you want. You will need to adjust the graphing number of points as well, but many contours have more than 800 points some have fewer it seemed like a comfortable middle ground but you should choose what ever works best for your downstream analyses.

```python
    # Normalize and resample all contours to 800 points
    normalized_contours = [resample_contour_points_fixed_first(contour, 800) for contour in contours]
```

```python
    # Resample all ordered contours to distribute points evenly along the contour
    resampled_ordered_contours = [resample_contour_points_fixed_first(contour, 800) for contour in ordered_contours]
```
Then just run the script

```shell
python updated_modified_resampling_contour_via_pca_with_tps_scalebarV6_align_resample_reorder-3.py
```
Before with PC alingment 
![aligned_contours](https://github.com/user-attachments/assets/17d841d8-50e9-45f0-86c7-fbe47ff747d3)
After semi-landmark conversion
![Resampled_contours_figure](https://github.com/user-attachments/assets/f2698066-3de3-405a-9511-f52c5bf5e27b)
![Procrustes_aligned_contours_figure](https://github.com/user-attachments/assets/e05984c0-e71e-4ec6-bcd5-126f9cbe85c5)
![later-toy-height-length](https://github.com/user-attachments/assets/3a61d20e-0bcf-43e8-9bee-3bea5f550c7d)
and with output similar to this for morphometrics of specific sclerites
```shell
Length in pixels: 1655.2730712890625
Height in pixels: 449.5775146484375
Length to height ratio: 3.6818413734436035
Area in pixels: 598013.5
Perimeter in pixels: 3902.452004432678
Aspect ratio: 3.681841322921721
Extent: 0.8035941212302876
Solidity: 0.9638111585215986
Equivalent diameter in pixels: 872.5906465723095
Orientation in degrees: 95.48037719726562
Major axis length in pixels: 464.1995849609375
Minor axis length in pixels: 1774.2918701171875
Length in um: 3492.137281200554
Height in um: 948.475769300501
Area in um^2: 2661670.5834179
Perimeter in um: 8233.021106398055
Equivalent diameter in um: 1840.908537072383
Major axis length in um: 979.3240189049314
Minor axis length in um: 3743.231793496176
```

# Foreground Mask Color Analyses and Extraction in python
The provided color analyses is invariant to the ordering of the points in the contours, and the color cube analyses is invariant to the shape and ordering of the color as both are crude, a more sophosticated color analyses of color pattern and average color is coming in future versions, but for now you can extract RGB values and basic color statistics, hue normalize, FHS color segment the interior regions of specific contours derived from the predicted sclerites and perform a PCA on the color cube histograms. As the starting material you need the binary mask and the foreground mask produced from the updated_get_masks_from_contours_multi-V2-4.py ran previously. If you are not familiar the binary mask is black background white is the region of interest or RoI.the foreground mask is the same shape but with RGB colors from the original image.The script will CLAHE normalize the hue and also perform the Felzenzwalb segmentationyou can mix and match what you want performed simply comment out the parts you don't want the FHS analyses takes a lot of memory and time to run. 

<img width="288" alt="Screen Shot 2024-08-15 at 5 30 36 AM" src="https://github.com/user-attachments/assets/8f5ec748-f728-45e1-93d5-e00e4c15b3ff">

In this example it is CLAHE hue normalization only found towards the bottom of the script 
```python
# Process each pair of image and mask
for base_name, files in file_pairs.items():
    if 'binary_mask' in files and 'foreground_mask' in files:
        image_path = os.path.join(input_dir, files['foreground_mask'])
        mask_path = os.path.join(input_dir, files['binary_mask'])
        #recolorized_path = os.path.join(output_dir, f"recolorized_{files['foreground_mask']}")
        
        # Perform recolorization
        #recolorize_like_segmentation(image_path, mask_path, recolorized_path)
        
        # Perform hue normalization
        normalized_path = os.path.join(output_dir, f"normalized_{files['foreground_mask']}")
        normalize_color(recolorized_path, normalized_path)
```
then simply run
```shell
python updated_FHS_and_normalize.py
```
to get color values and human readable output run
```shell
python color_analyses_out_to_human_readable_LAB.py
```
with results for an individual sclerite similar to this that can be used as quantitative color data for evolutionary biology studies
```shell
Average color of ASUCOB0015133_habitus_dorsal_1615428878_lg foreground_mask: [46.48795589,21.08326686,12.18220018] -> black
Average color of ASUCOB0015133_habitus_dorsal_1615428878_lg recolorized: [45.79914431,20.52126361,11.83857177] -> black
Average color of ASUCOB0015133_habitus_dorsal_1615428878_lg normalized: [70.77440976,45.71462225,37.67963767] -> coffee brown
Average color of ASUCOB0015133_habitus_dorsal_1615428878_lg normalized_only: [74.09385714,48.49679117,39.94669202] -> coffee brown
Dominant color of ASUCOB0015133_habitus_dorsal_1615428878_lg foreground_mask: [1,1,3] -> black
Dominant color of ASUCOB0015133_habitus_dorsal_1615428878_lg recolorized: [18,2,2] -> black
Dominant color of ASUCOB0015133_habitus_dorsal_1615428878_lg normalized: [25,11,10] -> black
Dominant color of ASUCOB0015133_habitus_dorsal_1615428878_lg normalized_only: [9,10,12] -> black
Top 10 colors of ASUCOB0015133_habitus_dorsal_1615428878_lg foreground_mask: [25.91003838,4.82128161,3.0798225] -> black
[147.48581158,103.03450624,67.0282255] -> sienna
[72.19642359,36.78169786,18.79985873] -> coffee brown
[232.27738337,202.97008114,148.57302231] -> burly wood
[55.41154443,22.23220952,10.29126414] -> coffee brown
[9.22327199,1.48090335,2.02440026] -> black
[113.43202623,76.40000852,48.27887768] -> deep brown
[40.18283057,11.95099754,5.75521212] -> black
[90.28516928,55.7377211,32.42806072] -> coffee brown
[190.50227964,144.64620061,93.33054711] -> peru
Top 10 colors of ASUCOB0015133_habitus_dorsal_1615428878_lg recolorized: [27.43188712,6.46149292,3.53816531] -> black
[91.35335104,56.72663411,34.11859276] -> coffee brown
[185.62928958,139.39780253,91.50692354] -> peru
[40.3451744,13.09716831,6.00971895] -> black
[143.85589137,100.71408235,66.75667981] -> sienna
[13.78980982,2.45581299,2.23362326] -> black
[232.92755682,202.51136364,149.4765625] -> burly wood
[55.29070091,24.14743574,11.72277128] -> coffee brown
[116.89117092,73.36614731,48.84855996] -> deep brown
[74.02878927,39.66555209,22.51782105] -> coffee brown
Top 10 colors of ASUCOB0015133_habitus_dorsal_1615428878_lg normalized: [20.32547889,11.4713023,10.80859859] -> black
[151.20188869,111.50175329,84.55344929] -> gray / grey
[69.39466122,40.10302962,35.54353538] -> coffee brown
[50.20186618,28.99579513,26.91027808] -> coffee brown
[218.27702617,173.96940195,124.71387027] -> tan
[131.59950135,91.95844615,69.71775556] -> deep brown
[174.4154708,134.72507553,101.85548842] -> rosy brown
[110.42066428,71.91500218,55.46182591] -> chestnut brown
[34.79129036,21.4945286,19.93186633] -> black
[88.68779349,53.66320921,44.52150434] -> coffee brown
Top 10 colors of ASUCOB0015133_habitus_dorsal_1615428878_lg normalized_only: [17.30226725,10.81619125,10.79391683] -> black
[113.75907963,73.78527072,58.76725686] -> deep brown
[158.68930934,119.29709906,89.57009849] -> gray / grey
[71.43125296,41.05031519,36.50784457] -> coffee brown
[136.01659321,94.7005509,72.04795064] -> deep brown
[187.04458816,147.1529223,108.34061072] -> rosy brown
[51.93298127,29.41846141,27.41216775] -> coffee brown
[92.34595141,56.20035592,47.31100789] -> taupe brown
[229.05110962,190.2896772,137.1565232] -> burly wood
[34.08382534,19.90815538,19.00964704] -> black
```

Color Cube Histogram PCA via
```shell
python updated_color_color_cube-V2.py
```
![color_cube4](https://github.com/user-attachments/assets/cfcbdeee-0f85-4b3f-a5f0-2a4985027752)

![PCA_color_histograms](https://github.com/user-attachments/assets/87e432ec-db6e-413d-b267-5f7ea9502f58)

This is an ongoing project and improvements will be made. Probably the best application is to work on a genus or tribe with a somewhat similar appearence. Probably about 50-100 annotations will get you very accurate predictions depending on how large/small the feature or sclerite is and how variable the appearence of your taxonomic group of interest. For example much of the training data was Entaminae weevils and the test data that had this subfamily and had high resolution in focus images tended to do well. I'm happy you are interested in this projct, please let me know if you have comments or questions and please cite this page and or the pre-print if you use the scripts for your own work.

