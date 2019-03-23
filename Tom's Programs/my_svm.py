import numpy as np
import scipy as sp
from IPython import embed
from os import path
import sys
import time

# To solve quadratic programming problems.
from cvxopt import matrix, solvers

# To assign numerical values to the labels.
from sklearn import preprocessing

class my_svm:
	def __init__(self, input_dataset, input_labels):
		
		# Input dataset must be of the form:
		# Rows    = datapoints
		# Columns = features
		self.shape      = input_dataset.shape
		self.data_size  = input_dataset.shape[0]
		self.feat_size  = input_dataset.shape[1]
		self.data       = input_dataset

		# We give numerical values to the labels
		le = preprocessing.LabelEncoder()
		le.fit(input_labels)
		self.dict       = dict(zip(le.classes_, le.transform(le.classes_)))
		self.dict_inv   = dict(zip(le.transform(le.classes_), le.classes_))
		self.num_labels = len(le.classes_)
		self.cur_label  = 0
		self.labels     = le.transform(input_labels)

		# Here we will later just write one-vs-all
		# labels.
		self.one_vs_all = np.array([1.0 if x == self.cur_label else -1.0 for x in self.labels])

		# Now we introduce all the output variables.
		# The vector of direction (depending on who we are confronting)
		self.svm_ova    = np.zeros(shape = (self.num_labels, self.feat_size))
		# The offset coefficient
		self.svm_ova_of = np.zeros(shape = (self.num_labels, 1))

	def fit_ova(self):
		# We follow the same definitions as in the Wikipedia page.

		# We use te following lambda parameter:
		lambda_par = 1.0

		# Then we get the matrix for which we want to optimize
		vect     = np.multiply(self.data.T, self.one_vs_all).T
		Q_matrix = matrix(np.dot(vect, vect.T))

		
		# As well as the other parameters:
		P_vector = matrix(-np.ones(shape = self.data_size))
		Id       = np.eye(self.data_size)
		G_matrix = matrix(np.block([[-Id], [Id]]))
		H_vector = matrix(np.append(np.zeros(shape = self.data_size), \
			np.ones(shape = self.data_size)/(self.data_size*lambda_par*2)) )
		A_matrix = matrix(np.asmatrix(self.one_vs_all, dtype = np.float64))
		B_vector = matrix([0.0])

		# We solve the quadratic programming problem.
		sol=solvers.qp(Q_matrix, P_vector, G_matrix, H_vector, A_matrix, B_vector)

		# We save the optimal direction.
		dual_direction  = np.asarray(list(sol['x']))

		# Then we pass from the dual to the primal problem:
		self.svm_ova[self.cur_label, :] = np.dot(dual_direction.T, vect)

		# We still have to compute the offset (the affine translation b).
		# This is not completely trivial. We have to use the complementary
		# slackness condition from the Karush-Kuhn-Tucker conditions (see
		# the associated Wikipedia page).

		# First we take the data points which lie on the boundary:
		active_points   = np.array([1.0 if np.abs(x)>1e-8 else 0.0 for x in dual_direction])

		# Then for stability we take the average of the of the results
		# in these directions (suggested by a MSE answer).
		self.svm_ova_of[self.cur_label, :] = np.dot(active_points, np.dot(self.data, self.svm_ova[self.cur_label, :])\
		 - self.one_vs_all)/np.sum(active_points)

	def fit(self):
		#Here we fit the svm to the data

		for label_index in range(self.num_labels):

			self.cur_label  = label_index

			# First we build the correct one-vs-all labels
			self.one_vs_all = np.array([1.0 if x == label_index else -1.0 for x in self.labels])

			# And we can find the svm on this label
			self.fit_ova()

	def predict(self, input_dataset_test):

		# We say that a point is of class I, if the SVM for I gives
		# a larger result than all the others.
		predictions_numerical = np.argmax( np.dot(input_dataset_test, self.svm_ova.T) - self.svm_ova_of.T , axis = 1)

		# The prediction is a number. We transform it into a label
		# via the preprocessing procedure.
		predictions = [self.dict_inv[_] for _ in predictions_numerical]

		return predictions
