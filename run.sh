#!/bin/bash

#
# Please ensure that you have prerequisites installed. You need to have conda installed and availbale on your path
# This script is not robust. Watch out for failures at each step.

# Create the environment
conda env create -f environment.yml

# Activate the environment
source activate semantic-segmentation-lab

# Download VGG, Augment VGG, and Train using Kitti Road Dataset
python ./TrainAugmentedVGG.py

# Run Semantic Segmentation on Images
python ./SemanticImageSegmentation.py

# Optionally, Run Semantic Segmentation on Video
python ./SemanticVideoSegmentation.py
