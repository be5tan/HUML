
import numpy as np
import scipy as sp
from IPython import embed
from os import path
import sys
import time

# To solve quadratic programming problems.
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# To compute gaussian kernel
from scipy.spatial.distance import pdist, squareform

# To assign numerical values to the labels.
from sklearn import preprocessing

class processing:
	def __init__(self):
		# Nothin yet
		a = []

	def slice( self, input_data, required_labels ):

		# We take the indeces which satisfy the required condition
		tf_ind  = [ (_ in required_labels) for _ in input_data[:,-1]]
		indeces = tuple(np.ix_(tf_ind)[0])

		return input_data[indeces, :]

def norm_eval(data_1, data_2):

	size_train = data_1.shape[0]
	size_test  = data_2.shape[0]
	temp = np.zeros(shape = (size_train, size_test))

	for i in range(size_train):
		for j in range(size_test):
			temp[i,j] = np.square(np.linalg.norm(data_1[i,:] - data_2[j,:]))


	# We compute the norm on every row.
	return temp

def kernel_eval( data_1, data_2, **kwargs ):

	if 'kernel' in kwargs:
		kernel = kwargs['kernel']
	else:
		kernel = 'linear'

	if kernel == 'linear':
		return np.dot(data_1, data_2.T)

	if kernel == 'exponential':
		gamma_par = 1

		return sp.exp(-norm_eval(data_1, data_2)/gamma_par**2)
	
	if kernel == 'polynomial':
		power     = 2
		return np.dot(data_1, data_2.T)

class svm:
	def __init__( self, input_data, input_labels, **kwargs ):

		# Possible **kwargs:
		# 1) lambda_par
		# 2) kernel
		
		# Input dataset must be of the form:
		# Rows    = datapoints
		# Columns = features
		self.feat_size  = input_data.shape[1]
		self.data       = np.c_[input_data, input_labels]
		self.labels     = input_labels

		# We give numerical values to the labels
		le = preprocessing.LabelEncoder()
		le.fit(self.labels)
		self.dict       = dict(zip(le.classes_, le.transform(le.classes_)))
		self.dict_inv   = dict(zip(le.transform(le.classes_), le.classes_))
		self.num_labels = len(le.classes_)
		self.cur_label  = 0
		self.label_vs   = 1
		self.data[:,-1] = le.transform(input_labels)

		# Here we will later just write one-vs-one
		# data and labels.
		self.data_ovo   = np.zeros(shape = 1)
		self.labels_ovo = np.zeros(shape = 1)

		# And the data without labels, with the associated size
		self.data_red   = np.zeros(shape = 1)
		self.data_size  = np.zeros(shape = 1)

		# Now we introduce all the output variables.
		# The vector of direction (depending on who we are confronting)
		self.svm_ovo    = [ [ [] for i in range(self.num_labels) ] for j in range(self.num_labels) ]
		# The offset coefficient
		self.svm_ovo_of = np.zeros(shape = (self.num_labels, self.num_labels, 1))

		# We define the regularisation parameter:
		if 'lambda_par' in kwargs:
			self.lambda_par = kwargs['lambda_par']
		else:
			self.lambda_par = 0.3

		# We define the kernel we want to use
		if 'kernel' in kwargs:
			self.kernel = kwargs['kernel']
		else:
			self.kernel = 'linear'

	def prediction_function(self, to_evaluate):

		# This evaulates sum_i y_i alpha_i K(x_i, x) for an imput x
		linear_direction = self.svm_ovo[self.cur_label][self.label_vs]
		linear_direction = np.multiply(linear_direction, self.labels_ovo)

		return np.dot(linear_direction, kernel_eval(self.data_red, to_evaluate, kernel = self.kernel))

	def prepare_data_ovo(self):

		# We extract the data and labels which have to do with the 
		# labels "cur_label" and "label_vs"
		pr = processing()
		self.data_ovo   = pr.slice(self.data, [self.cur_label, self.label_vs])

		# We pick the last row of the reduced dataset
		# And we reduce to two labels (+/-1).
		self.labels_ovo = np.array([1.0 if x == self.cur_label else -1.0 for x in self.data_ovo[:,-1]])

		# Finally we save the data without the labels, and it's dimension
		self.data_red   = self.data_ovo[:,:-1]
		self.data_size  = self.data_red.shape[0]

	def fit_ovo(self):

		# We follow the same definitions as in the Wikipedia page.
		# We get the matrix for which we want to optimize
		Q = np.multiply(kernel_eval(self.data_red, self.data_red, kernel = self.kernel), np.tensordot(self.labels_ovo, \
			self.labels_ovo, axes= 0)).astype(np.double)
		#vect     = (np.multiply(self.data_red.T, self.labels_ovo).T).astype(np.double)
		Q_matrix = matrix(Q)

		
		# As well as the other parameters:
		P_vector = matrix(-np.ones(shape = self.data_size))
		Id       = np.eye(self.data_size)
		G_matrix = matrix(np.block([[-Id], [Id]]))
		H_vector = matrix(np.append(np.zeros(shape = self.data_size), \
			np.ones(shape = self.data_size)/(self.data_size*self.lambda_par*2)) )
		A_matrix = matrix(np.asmatrix(self.labels_ovo, dtype = np.float64))
		B_vector = matrix([0.0])

		# We solve the quadratic programming problem.
		sol=solvers.qp(Q_matrix, P_vector, G_matrix, H_vector, A_matrix, B_vector)

		# We save the optimal direction.
		dual_direction  = np.asarray(list(sol['x']))

		# Then we store the dual:
		self.svm_ovo[self.cur_label][self.label_vs] = dual_direction

		# We still have to compute the offset (the affine translation b).
		# This is not completely trivial. We have to use the complementary
		# slackness condition from the Karush-Kuhn-Tucker conditions (see
		# the associated Wikipedia page).

		# First we take the data points which lie on the boundary:
		active_points   = np.array([1.0 if np.abs(x)>1e-8 else 0.0 for x in dual_direction])

		# Then for stability we take the average of the of the results
		# in these directions (suggested by a MSE answer).
		self.svm_ovo_of[self.cur_label, self.label_vs, :] = np.dot(active_points, \
			self.prediction_function(self.data_red) - self.labels_ovo)/np.sum(active_points)

	def fit(self):
		#Here we fit the svm to the data

		for label_index in range(self.num_labels):
			for label_vs_index in range(label_index + 1, self.num_labels):

				self.cur_label  = label_index
				self.label_vs   = label_vs_index

				# First we build the correct one-vs-all data
				self.prepare_data_ovo()

				# And we can find the svm on this label
				self.fit_ovo()

	def predict(self, input_dataset_test):

		# We predict the one versus one classification first.
		prediction_vector = np.zeros(shape = (input_dataset_test.shape[0], self.num_labels, self.num_labels))

		for label_index in range(self.num_labels):
			for label_vs_index in range(label_index+1, self.num_labels):
				
				# We always need the sliced data!
				self.cur_label = label_index
				self.label_vs  = label_vs_index
				self.prepare_data_ovo()

				# In this way we can compute the prediction function
				prediction_vector[:, label_index, label_vs_index] = self.prediction_function(input_dataset_test)\
				 - self.svm_ovo_of[label_index, label_vs_index, :].T

				#We make the matrix antisymmetric.
				prediction_vector[:, label_vs_index, label_index] = - prediction_vector[:, label_index, label_vs_index]

		# We pass from numerical to +/-1 for voting:
		prediction_vector= 0.5*(np.sign(prediction_vector) +1)
		# Then sum all the results, to get the votes.
		prediction_votes = np.zeros(shape = (input_dataset_test.shape[0], self.num_labels))
		prediction_votes = np.sum(prediction_vector, axis = 2)

		# And we choose the class that gets the most votes.
		predictions_numerical = np.argmax(prediction_votes, axis =1)

		# The prediction is a number. We transform it into a label
		# via the preprocessing procedure.
		predictions = [self.dict_inv[_] for _ in predictions_numerical]

		return predictions