import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split  

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)  
print(dataset.head())

# Preprocess and assign labels
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)  

print(X_train.shape)
print(Y_train.shape)

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors = 5)
# classifier.fit(X, Y)
