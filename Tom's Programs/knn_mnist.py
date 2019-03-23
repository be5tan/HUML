# We implement a function that takes an elemnet of mnist
# and plots a picture.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy as sp
from IPython import embed
from os import path
import sys
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix  

# First, we import the data:

train_data = np.loadtxt("mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("mnist_test.csv", 
                       delimiter=",")

# We rescale and we take out labels and images:
fac = 255  *0.99 + 0.01
train_imgs   = train_data[:, 1:] / fac
test_imgs    = test_data[:, 1:] / fac
train_labels = train_data[:, :1]
test_labels  = test_data[:, :1]

# We produce a picture
# for i in range(5):
#    img = train_imgs[i].reshape((28,28))
#    plt.imshow(img, cmap="Greys")
#    plt.show()

embed()

# Now we want ot use KNN for classification
neigh = KNeighborsClassifier(n_neighbors=3)
train_results = np.ravel(train_data[:, :1])
neigh.fit(train_data[:, 1:], train_results)
# We can do predictions:
y_pred = neigh.predict(test_data[:,1:])

print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))
