import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy as sp
from IPython import embed
from os import path
import sys
import time

# To solve quadratic programming problems
from cvxopt import matrix, solvers

# To write down results from classification
from sklearn.metrics import classification_report, confusion_matrix  

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

# We change the value of the labels to numbers
dataset = dataset.replace({'Class': {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}})

# We take out the labels:
data_points  = dataset.iloc[:, :-1].values
data_label   = dataset.iloc[:, 4].values
signed_label = np.array([1.0 if x == 1 else -1.0 for x in data_label])

# We split the data.
from sklearn.model_selection import train_test_split  
data_test, data_train, label_test, label_train = train_test_split(data_points, signed_label, test_size=0.30)

data_size = data_train.shape[0]

# Then we get the matrix for which we want to optimize
# (see the optimisation problem in Wikipedia):
vect     = np.multiply(data_train.T, label_train).T
Q_matrix = matrix(np.dot(vect, vect.T))

# We use te following lambda parameter:
lambda_par = 1

# As well as the other parameters:
P_vector = matrix(-np.ones(shape = data_size))
Id       = np.eye(data_size)
G_matrix = matrix(np.block([[-Id], [Id]]))
H_vector = matrix(np.append(np.zeros(shape = data_size), np.ones(shape = data_size)/(data_size*lambda_par*2)) )
A_matrix = matrix(np.asmatrix(label_train))
B_vector = matrix([0.0])

# We solve the quadratic programming problem.
sol=solvers.qp(Q_matrix, P_vector, G_matrix, H_vector, A_matrix, B_vector)

# We save the optimal direction.
dual_direction= np.asarray(list(sol['x']))

# Then we pass from the dual to the primal problem
svm_direction = np.dot(dual_direction.T, vect)

# We still have to compute the offset (the affine translation b).
# This is not completely trivial. We have to use the complementary
# slackness condition from the Karush-Kuhn-Tucker conditions (see
# the associated Wikipedia page).

# First we take the data points which lie on the boundary:
active_points = np.array([1.0 if np.abs(x)>1e-8 else 0.0 for x in dual_direction])

# Then for stability we take the average of the of the results
# in these directions (suggested by a MSE answer).
svm_offset = np.dot(active_points, np.dot(data_train, svm_direction) - label_train)/np.sum(active_points)

# We can do predictions
predict_test  = np.sign(np.dot(data_test, svm_direction.T)- svm_offset)

# We see how good we perform.
print(confusion_matrix(label_test, predict_test))  
print(classification_report(label_test, predict_test))

# We can also plot the results:
color_dict = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}

my_cmap = [color_dict[data_label[_]] for _ in range(len(data_label))]

# To plot it we have to projsect on a two-dimensional space
# The first coordinate will be the projection along the svm_direction
# centered around the offset.\
first_direction    = np.zeros(shape = 4)
first_direction[0] = 1.0
plt.scatter(np.dot(data_points, svm_direction.T)- svm_offset, np.dot(data_points, first_direction.T), c = my_cmap)
plt.show()
