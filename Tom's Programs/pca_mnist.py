# We perform PCA on the MNIST dataset

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
from scipy.sparse import issparse
from numpy.linalg import eig

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

avrg = np.mean(train_imgs, axis = 0)
train_centered = train_imgs - avrg
Cov = np.dot(train_centered.T, train_centered)

# We can check whether the matrix is sparse
# issparse(Cov)
# Its not sparse!

# We diagonalize the matrix
[eig_val, eig_vec] = eig(Cov)
PCA_1 = eig_vec[:,0]
PCA_2 = eig_vec[:,1]

# We can also plot the first eigenvector as an image
PCA_img_1 = PCA_1.reshape((28,28))
plt.imshow(PCA_img_1, cmap="Greys")
plt.show()

# And the second
PCA_img_2 = PCA_2.reshape((28,28))
plt.imshow(PCA_img_2, cmap="Greys")
plt.show()

# We plot the projection of the data along these two
# eigenspaces. For a better image we take only part
# of the data.

train_imgs_reduced   = train_data[:500, 1:] / fac
train_labels_reduced = train_data[:500, :1]
labels_reduced_str   = (train_labels_reduced.astype(int)).astype(str)

proj_data_1 = np.dot(train_imgs_reduced, PCA_1)
proj_data_2 = np.dot(train_imgs_reduced, PCA_2)

color_dict = {'0': 'tab:blue', '1': 'tab:orange', '2': 'tab:green', '3': 'tab:red','4': 'tab:purple', '5': 'tab:brown', '6': 'tab:pink', '7': 'tab:gray', '8': 'tab:olive', '9': 'tab:cyan'}

my_cmap = [color_dict[labels_reduced_str[_, 0]] for _ in range(len(labels_reduced_str))]

plt.scatter(proj_data_1, proj_data_2, c = my_cmap)
plt.show()

# We predict with PCA up to the first eig_num eigenvalues
eig_num = 50

# We set up KNN
neigh = KNeighborsClassifier(n_neighbors=3)
train_results = np.ravel(train_data[:, :1])

# We define KNN sets
train_imgs_PCA = []
test_data_PCA = []

outfile = open("Results_PCA.txt", 'w')

for _ in range(eig_num):

	# We compute the projection

	train_imgs_PCA = np.dot(train_imgs, eig_vec[:, :(_+1)])
	test_data_PCA = np.dot(test_imgs, eig_vec[:, :(_+1)])

	# We fit the KNN
	neigh.fit(train_imgs_PCA, train_results)

	# We predict
	y_pred = neigh.predict(test_data_PCA)

	# We print the results
	outfile.write( classification_report(test_labels, y_pred))
	outfile.write("\n \n Step number is {} \n \n".format(_))

outfile.close()

embed()

# We classify via KNN in 1 PCA

# We open a file on which we write the results
outfile = open("Results_PCA.txt", 'w')

train_imgs_PCA_1 = np.dot(train_imgs, eig_vec[:, :1])

neigh = KNeighborsClassifier(n_neighbors=3)
train_results = np.ravel(train_data[:, :1])
neigh.fit(train_imgs_PCA_1, train_results)

test_data_PCA_1 = np.dot(test_imgs, eig_vec[:, :1])
y_pred = neigh.predict(test_data_PCA_1)

print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))

outfile.write( classification_report(test_labels, y_pred) )
# We classify via KNN in 2 PCA
train_imgs_PCA_2 = np.dot(train_imgs, eig_vec[:,:2])

neigh = KNeighborsClassifier(n_neighbors=3)
train_results = np.ravel(train_data[:, :1])
neigh.fit(train_imgs_PCA_2, train_results)

y_pred = neigh.predict(test_data[:,1:])

print(confusion_matrix(test_labels, y_pred))
print(classification_report(test_labels, y_pred))

# We classify via KNN in 10 PCA

outfile.close()
