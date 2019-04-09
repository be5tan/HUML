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
from huml import svm
from huml import kernel_eval
from huml import norm_eval

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

# We take out the labels:
data_points  = dataset.iloc[:, :-1].values
data_label   = dataset.iloc[:, 4].values

import huml
pca = huml.PCA(data_points)
pca.decompose()
# We indicate the dimension to which
# we want to reduce the problem
data_red_dim = pca.project(2)

# We split the data. 
data_test, data_train, label_test, label_train = train_test_split(data_red_dim, data_label, test_size=0.30)

# We creat the SVM class and fit it to the data
exmpl = svm(data_train, label_train, lambda_par = 0.2, kernel = 'linear')
exmpl.fit()

# Now we can do predictions
predict_test = exmpl.predict(data_test)

# We see how good we perform.
print(confusion_matrix(label_test, predict_test))  
print(classification_report(label_test, predict_test))

embed()

N   = 50
M   = np.zeros(shape = (N,N))
vect = np.zeros(shape = (1,2))
x   = (np.arange(N) -N/2)/(N/5)
y   = (np.arange(N) -N/2)/(N/2)

for i in range(N):
	for j in range(N):
		vect[0,0] = x[i]/N
		vect[0,1] = y[i]/N
		M[i, j] = exmpl.predict_numerical(vect)

embed()

plt.imshow(M)
plt.show()