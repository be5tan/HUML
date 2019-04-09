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
from huml import PCA

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

# We take out the labels:
data_points  = dataset.iloc[:, :-1].values
data_label   = dataset.iloc[:, 4].values

# We split the data. 
data_test, data_train, label_test, label_train = train_test_split(data_points, data_label, test_size=0.30)

# We creat the SVM class and fit it to the data
exmpl = svm(data_train, label_train, lambda_par = 0.2, kernel = 'exponential')
exmpl.fit()

# Now we can do predictions
predict_test = exmpl.predict(data_test)

# We see how good we perform.
print(confusion_matrix(label_test, predict_test))  
print(classification_report(label_test, predict_test))

# Now we pass to PCA.
pca = PCA(data_points)

# We run PCA
pca.decompose()

# And we project the data on the second component
pca.project(2)

# We make numerical labels
le = preprocessing.LabelEncoder()
le.fit(data_label)
numerical_label = le.transform(data_label)

embed()

# We create a color dictionary for the plot
color_dict = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red',\
4: 'tab:purple', 5: 'tab:brown', 6: 'tab:pink', 7: 'tab:gray', 8: 'tab:olive', 9: 'tab:cyan'}
my_cmap = [color_dict[numerical_label[_]] for _ in range(len(numerical_label))]

# We plot the projected data
plt.scatter(pca.projected[:,0], pca.projected[:,1], c = my_cmap)
plt.show()