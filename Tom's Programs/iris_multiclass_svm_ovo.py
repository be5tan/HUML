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

# We split the data. 
data_test, data_train, label_test, label_train = train_test_split(data_points, data_label, test_size=0.30)

# We creat the SVM class and fit it to the data
exmpl = svm(data_train, label_train, lambda_par = 0.2, kernel = 'exponential')
embed()
exmpl.fit()

# Now we can do predictions
predict_test = exmpl.predict(data_test)

# We see how good we perform.
print(confusion_matrix(label_test, predict_test))  
print(classification_report(label_test, predict_test))



# The part below works only for linear SVM. Need to change it to make it work
# for the new class.

# # We can also plot the results:
# color_dict = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}

# le = preprocessing.LabelEncoder()
# le.fit(data_label)

# numerical_label = le.transform(data_label)
# my_cmap = [color_dict[numerical_label[_]] for _ in range(len(data_label))]

# # To plot it we have to projsect on a two-dimensional space
# # The first coordinate will be the projection along the svm_direction
# # centered around the offset.\
# first_direction    = np.zeros(shape = 4)
# first_direction[0] = 1.0
# plt.scatter(np.dot(data_points, exmpl.svm_ovo[0][1].T)- exmpl.svm_ovo_of[0,1,:],\
#  np.dot(data_points, first_direction.T), c = my_cmap)
# plt.show()
# plt.scatter(np.dot(data_points, exmpl.svm_ovo[1][2].T)- exmpl.svm_ovo_of[1,2, :],\
#  np.dot(data_points, first_direction.T), c = my_cmap)
# plt.show()
# plt.scatter(np.dot(data_points, exmpl.svm_ovo[0][2].T)- exmpl.svm_ovo_of[0,2,:],\
#  np.dot(data_points, first_direction.T), c = my_cmap)
# plt.show()

# # We can also confron this with the sklearn SVM:
# clf = SVC( kernel = 'linear')
# clf.fit(data_train, label_train)

# print(confusion_matrix(label_test, clf.predict(data_test)))  
# print(classification_report(label_test, clf.predict(data_test)))

# # Let us test how we perform on one the last two flowers.

# from huml import processing
# from sklearn.svm import LinearSVC

# embed()

# pr = processing()
# dataset_v = dataset.values
# data_sliced = pr.slice(dataset_v, ['Iris-versicolor', 'Iris-virginica'])

# # We take out the labels:
# data_points  = data_sliced[:, :-1]
# data_label   = data_sliced[:, 4]

# # We split the data. 
# data_test, data_train, label_test, label_train = train_test_split(data_points, data_label, test_size=0.30)

# for _ in range(1, 21):

# 	# We creat the SVM class and fit it to the data
# 	exmpl = svm(data_train, label_train, _/(25.0))
# 	exmpl.fit()

# 	# Now we can do predictions
# 	predict_test = exmpl.predict(data_test)

# 	print('lambda = {}'.format(_/(25.0)))
# 	# We see how good we perform.
# 	print(confusion_matrix(label_test, predict_test))  
# 	print(classification_report(label_test, predict_test))

# 	# We confront with Sklearn
# 	clf = LinearSVC()
# 	clf.fit(data_train, label_train)

# 	print(confusion_matrix(label_test, clf.predict(data_test)))  
# 	print(classification_report(label_test, clf.predict(data_test)))