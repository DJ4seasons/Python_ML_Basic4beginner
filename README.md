# Python_ML_Basic4beginner
#### Python code examples for beginning Machine Learning(ML)
#### Assume having basic knowledge of Numpy and Matplotlib

1. Multi-Layer Perceptron(MLP; Feed-Forward Neural Network) with Scikit-Learn (MLPClassifier)

1.1. Testing scaler and label encoding

1.2. Testing hyperparameter tuning with gridsearchCV

1.3. An idea of ensemble method

1.4. Saving and Loading model settings

2. MLP with Tensorflow+Keras
3. Convolution Neural Network(CNN) with Tensorflow+Keras

### Data 
1. NOAA climate data record(CDR) Outgoing Longwave Radiation(OLR)

Monthly 1979-2019, 2.5deg X 2.5deg, DOI: 10.7289/V5W37TKD, from https://www.ncdc.noaa.gov/cdr/atmospheric/outgoing-longwave-radiation-monthly


2. Nino3.4 index 
Monthly, from https://psl.noaa.gov/data/correlation/nina34.data


### Problem
Forecast [El Nino / Neutral / La Nina] (based on Nino3.4 index) with OLR data by 3-month

### Modules/Packages needed (Check_python_module_py3.py)

import sys

import os.path

import numpy

import netCDF4

import math

import datetime

import matplotlib.pyplot

import scipy

import sklearn

import pickle

import joblib

import tensorflow.keras

