
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

# To solve quadratic programming problems.
from cvxopt import matrix, solvers

# To write down results from classification.
from sklearn.metrics import classification_report, confusion_matrix  

# To split the data in train and test.
from sklearn.model_selection import train_test_split 

# To assign numerical values to the labels.
from sklearn import preprocessing

# SVM com Sklearn, for comparison.
from sklearn.svm import SVC

# Import hand-made SVM program.
#from huml import svm
#from huml import kernel_eval
#from huml import norm_eval

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

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-5
for t in range(700):
    # Forward pass: compute predicted y
    h = data_train.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - label_train).sum()
    # print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - label_train)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = data_train.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print(loss)

# Now we do the same with PyTorch
import torch

dtype = torch.double
device = torch.device("cpu")
#device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.from_numpy(data_train)
y = torch.from_numpy(label_train)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-5
for t in range(700):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print(loss)