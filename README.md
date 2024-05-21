# DGDATA: Deep Generative Domain Adaptation with Temporal Relation Attention Mechanism for Cross-User Activity Recognition

## Authors
- Xiaozhou Ye, Email: xye685@aucklanduni.ac.nz
- Kevin Wang, Email: kevin.wang@aucklanduni.ac.nz

## Affiliation
Department of Electrical, Computer, and Software Engineering, The University of Auckland

## Overview

This repository contains the code and data for our paper: "Deep Generative Domain Adaptation with Temporal Relation Attention Mechanism for Cross-User Activity Recognition". The goal of this project is to enhance cross-user Human Activity Recognition (HAR) by integrating temporal dependency relations during domain adaptation.

## Setup Environment
Python 3.10.11

pip install -r requirements.txt

asttokens==2.4.1
colorama==0.4.6
comm==0.2.2
contourpy==1.2.0
cycler==0.12.1
decorator==5.1.1
exceptiongroup==1.2.1
executing==2.0.1
filelock==3.14.0
fonttools==4.50.0
fsspec==2024.5.0
giotto-ph==0.2.2
giotto-tda==0.6.0
igraph==0.11.5
intel-openmp==2021.4.0
ipython==8.24.0
ipywidgets==8.1.2
jedi==0.19.1
Jinja2==3.1.4
joblib==1.4.2
jupyterlab_widgets==3.0.10
kiwisolver==1.4.5
MarkupSafe==2.1.5
matplotlib==3.8.3
matplotlib-inline==0.1.7
mkl==2021.4.0
mpmath==1.3.0
networkx==3.3
numpy==1.26.4
packaging==24.0
pandas==2.2.1
parso==0.8.4
patsy==0.5.6
pillow==10.2.0
plotly==5.22.0
POT==0.9.3
prompt-toolkit==3.0.43
pure-eval==0.2.2
pyflagser==0.4.5
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
scikit-learn==1.4.2
scipy==1.13.0
six==1.16.0
stack-data==0.6.3
statsmodels==0.14.2
sympy==1.12
tbb==2021.12.0
tenacity==8.3.0
texttable==1.7.0
threadpoolctl==3.5.0
torch==2.3.0
traitlets==5.14.3
typing_extensions==4.11.0
tzdata==2024.1
wcwidth==0.2.13
widgetsnbextension==4.0.10

## Data Preparation
The datasets used in this project are OPPT, PAMAP2, and DSADS. Download the datasets from their respective sources.

http://www.opportunity-project.eu/node/56.html

https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

# File Overview

This repository contains various scripts and modules essential for training and running a deep generative temporal relation model with autoregression across different datasets (DSADS, OPPT, and PAMAP2).

## Main Training Scripts

- `GPU_DSADS_deepGenTRNet_main_with_autoregression.py`: Main script for training the model using the DSADS dataset.
- `GPU_OPPT_deepGenTRNet_main_with_autoregression.py`: Main script for training the model using the OPPT dataset.
- `GPU_PAMAP2_deepGenTRNet_main_with_autoregression.py`: Main script for training the model using the PAMAP2 dataset.

## Utility Modules

- `utils.py`: Contains helper functions for data processing and visualization.

## Model Training and Definitions

- `gen_model/train.py`: Script for the training phase of deep learning models.

### Algorithms

- `gen_model/alg/DeepGenTempRelaNet.py`: Defines the deep generative temporal relation model class.
- `gen_model/alg/linear_regression.py`: Contains the definition and training function for a linear regression model.
- `gen_model/alg/modelopera.py`: Utility module for various model operations.
- `gen_model/alg/opt.py`: Includes functions for obtaining and setting the optimizer.

### Loss Functions

- `gen_model/loss/common_loss.py`: Implements various loss functions used within the models.

### Network Models

- `gen_model/network/Adver_network.py`: Defines a neural network model including a backpropagation function and a discriminator model.
- `gen_model/network/common_network.py`: Contains definitions for common neural network models.
- `gen_model/network/feature_extraction_network.py`: Definitions for feature extraction models in neural networks.

### Utilities

- `gen_model/utils/util.py`: Provides helper functions for logging, formatting, and setting random seeds.

### Data process

- `OPPT_get_features_samples.py`: reads the OPPT dataset, selects specific activities and sensor channels, and extracts data for a source user and a target user

- `PAMAP2_get_features_samples.py`: reads the PAMAP2 dataset, selects specific activities and sensor channels, and extracts data for a source user and a target user

- `DSADS_get_features_samples.py`: reads the DSADS dataset, selects specific activities and sensor channels, and extracts data for a source user and a target user

# Hyperparameters

The hyperparameters used in the model can significantly impact performance. Here are the key hyperparameters and their default values:

| Hyperparameter                          | Default Value |
|-----------------------------------------|---------------|
| Training Epochs                         | 100           |
| Adam Optimizer Weight Decay             | 0.0005        |
| Adam Optimizer Beta                     | 0.2           |
| Reconstruction Loss Coefficient (α)     | 1.0           |
| Mean-Variance Loss Coefficient (ζ)      | 10.0          |
| Class Constraint Loss Coefficient (γ)   | 30.0          |
| Domain Constraint Loss Coefficient (δ)  | 1.0           |
| Temporal State Constraint Loss Coefficient (η) | 10.0   |


# Basic Example
Here is a basic example to get you started with training the model on the OPPT dataset:

python GPU_OPPT_deepGenTRNet_main_with_autoregression.py

This script will load the OPPT dataset, initialize the model, and start training.
