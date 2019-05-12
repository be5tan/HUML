
# We adapt the tutorial from 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# to the IRIS dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy as sp
from IPython import embed
from os import path
import sys
import time

# To write down results from classification.
from sklearn.metrics import classification_report, confusion_matrix  

# To split the data in train and test.
from sklearn.model_selection import train_test_split 

# To assign numerical values to the labels.
from sklearn import preprocessing

# SVM com Sklearn, for comparison.
from sklearn.svm import SVC

# We want to use PyTorch
import torch


# We devine a convolutional neural network class
# In this setting we need a 1D convolutional net

class my_cnn(torch.nn.Module):

    def __init__(self):

        # This is a magical line, but without the program does not work.
        # It initializes all the classes you refer to (i.e. the torch Module class in this case)
        super().__init__()

        # We define the first convolutional layer:
        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size =3, stride = 1, padding = 1)

    def forward(self, x):

        # We compute the forward process of the net:
        x = self.conv1(x)

        return(x)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

# We take out the labels:
data_points  = dataset.iloc[:, :-1].values
data_label   = dataset.iloc[:, 4].values

le = preprocessing.LabelEncoder()
le.fit(data_label)
numerical_label = le.transform(data_label)

# We split the data. 
data_test, data_train, label_test, label_train = train_test_split(data_points, numerical_label, test_size=0.30)
label_train = np.matrix(label_train, dtype = np.double).T

N = len(data_train)
D_in = 4
H = 100
D_out = 1

dtype = torch.double
#device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.from_numpy(data_train).double()
y = torch.from_numpy(label_train).double()

# We have to put them into the shape (dataset_size, 1, data_point_size)
# where the 1 represents the number of input channels
shape_tens = x.shape
x = x.reshape([shape_tens[0], 1, shape_tens[1]])
x = x.double()

# We define the network:
net = my_cnn()
net = net.double()

embed()