## Overview

This repository contains the code for segmenting vehicles and persons on street images using YOLO-NAS for object detection and Meta SAM for image segmentation.

## Key Features

Combines the strengths of YOLO-NAS (fast object detection) and Meta SAM (prompt-based segmentation) for accurate segmentation.
Targets vehicles and persons specifically, relevant for various applications.
Demonstrates effective integration of two state-of-the-art models for combined tasks.
## Dependencies

cuda-toolkit (11.0 or newest, tested with cuda 12.2 )
Anaconda
Python 3.9
PyTorch 2.1.0
torchvision
OpenCV

## Installation
Install cuda toolkit and cudnn (cuda-toolkit website)[https://developer.nvidia.com/cuda-toolkit]
install anaconda (anaconda website for download)[https://www.anaconda.com/download]
install pytorch 2.12 using (pytorch official website)[https://pytorch.org/]
Clone this repository:
Bash
git clone https://github.com/SuchiraLaknath/Vehicle_Segmentation_with_YoloNas_and_MetaSam.git
Install required libraries:
Bash
pip install -r requirements.txt

## Usage

### Download pre-trained YOLO-NAS and Meta SAM models.
Download Vit_b model of meta sam (ViT-B SAM model.)[https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth]
Copy and paste the downloaded "sam" ".pth" file into chekpoints directory.
For Inference Yolo_weights need not to be download, they'll be downloaded automatically when code is runninng for first time

set the inference image path that you need to segment in "config_inference.py" as "image_path".
Run the segmentation script:
Bash
python app.py



## Authors

Suchira Laknath Wanasinghe (LinkedIn)[https://www.linkedin.com/in/suchira-wanasinghe-3734711b6/] , (github)[https://github.com/SuchiraLaknath]
## Acknowledgments

Deci AI (for YOLO-NAS)[https://deci.ai/blog/yolo-nas-foundation-model-object-detection/] 
Meta AI (for Meta SAM)[https://segment-anything.com/]

## To Do
Video inference,
Quantze the models,
Develop the custom training code
develop a program to convert bbox datasets to segmantation mask dataset
deploying the model
