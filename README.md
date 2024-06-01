This repository contains the code and data used for my thesis. 

It includes python code for preprocessing the image annotations, code to inference images based on the models, and code to generate plots that were used in the report. 

It also includes config files, which stem from the tool MMDetection. 
MMDetection is an open source object detection toolbox based on PyTorch.
MMDetection uses a modular design, all modules with different functions can be configured through the config. 
Thus, config files can be created to customize the functions from MMDetection.
The config files included in this repository were used to train, validate and test the models.
They were also used for the experiments conducted in the research. Such as resizing the images, altering the pre-training datasets, and hyperparameter testing.

Two of the three open datasets used for pre-training are included in this repository. The ground-level dataset from November and the ground-level dataset from May.
The grass weed dataset was too large to upload to GitHub.
The dataset containing the aerial images of the Japanese Knotweed is not included, due to an NDA disclosed with Datacation.
