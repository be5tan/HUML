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
from sklearn import svm

# Import hand-made SVM program.
from my_svm import my_svm

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
svm_example = my_svm(data_train, label_train)
svm_example.fit()

# Now we can do predictions
predict_test = svm_example.predict(data_test)

# We see how good we perform.
print(confusion_matrix(label_test, predict_test))  
print(classification_report(label_test, predict_test))

# We can also plot the results:
color_dict = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}

le = preprocessing.LabelEncoder()
le.fit(data_label)

my_cmap = [color_dict[le.transform(data_label)[_]] for _ in range(len(data_label))]

# To plot it we have to projsect on a two-dimensional space
# The first coordinate will be the projection along the svm_direction
# centered around the offset.\
first_direction    = np.zeros(shape = 4)
first_direction[0] = 1.0
plt.scatter(np.dot(data_points, svm_example.svm_ova[0,:].T)- svm_example.svm_ova_of[0,:],\
 np.dot(data_points, first_direction.T), c = my_cmap)
plt.show()
plt.scatter(np.dot(data_points, svm_example.svm_ova[1,:].T)- svm_example.svm_ova_of[1,:],\
 np.dot(data_points, first_direction.T), c = my_cmap)
plt.show()
plt.scatter(np.dot(data_points, svm_example.svm_ova[2,:].T)- svm_example.svm_ova_of[2,:],\
 np.dot(data_points, first_direction.T), c = my_cmap)
plt.show()

embed()

# We can also confron this with the sklearn SVM:
clf = svm.LinearSVC()
clf.fit(data_train, label_train)

print(confusion_matrix(label_test, clf.predict(data_test)))  
print(classification_report(label_test, clf.predict(data_test)))