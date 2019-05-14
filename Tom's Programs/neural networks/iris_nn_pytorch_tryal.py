
# We adapt the tutorial from 
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# to the IRIS dataset

# We also follow
#https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/

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
# And also from one-hot encoding.
from sklearn import preprocessing

# SVM com Sklearn, for comparison.
from sklearn.svm import SVC

# We want to use PyTorch
import torch

# For simplicity we write
import torch.nn.functional as torchf


# We devine a convolutional neural network class
# In this setting we need a 1D convolutional net

class my_cnn(torch.nn.Module):

    def __init__(self):

        # This is a magical line, but without the program does not work.
        # It initializes all the classes you refer to (i.e. the torch Module class in this case)
        super().__init__()

        # We define the first convolutional layer:
        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size =3, stride = 1, padding = 1)
        # Then the pooling:
        self.pool1 = torch.nn.MaxPool1d(kernel_size = 2, stride = 2,  padding =0)
        # Then the first linear layer:
        self.lin1  = torch.nn.Linear(2, 10)

        #Then the second linear layer:
        self.lin2  = torch.nn.Linear(10, 3)

    def forward(self, x):

        # We compute the forward process of the net
        # Starting with the convolution part:
        x = self.pool1(self.conv1(x))
        x = torchf.relu(x)

        # And then the linear part.
        # First we need to put it into the right shape:
        # -1 tells him to figure out the correct dimension by himself.
        x = x.reshape(-1,2)
        x = self.lin1(x)
        x = torchf.relu(x)
        x = self.lin2(x)

        return(x)

# Loss and optimizer for our cnn:
def createLossAndOptimizer(net, learning_rate=0.002):
    
    #Loss function: It's the classical entropy:
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer (stochastic gradient descent):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

# We take out the labels:
data_points  = dataset.iloc[:, :-1].values
data_label   = dataset.iloc[:, 4].values

# We have to do one-hot encoding of the labels.
oh = preprocessing.LabelBinarizer()
oh.fit(data_label)
oh_label = oh.transform(data_label)

# We split the data. 
data_test, data_train, label_test, label_train = train_test_split(data_points, oh_label, test_size=0.30)
label_train = np.matrix(label_train, dtype = np.double).T

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

# We define a simple training algorithm.
# First we define the loss and optimisation algorithm via the
# definition at the beginning of the program
(loss_fn, optim_fn) = createLossAndOptimizer(net)

embed()

for t in range(10):

    # Forward pass:
    y_pred = net(x)

    # We compute the loss function:
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # We zero the gradient of the optimizer
    # (this passage just has to happen)
    optim_fn.zero_grad()

    # We go backward in the loss:
    loss.backward()

    # and based on the gradient and via the optimizer we
    # adjourn the parameters:
    optimizer.step()


embed()