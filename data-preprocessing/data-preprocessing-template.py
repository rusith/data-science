# Data Preprocessing
# Some libraries we would need
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
# importing the data set
dataset = pd.read_csv("Data.csv")

#Splitting the data set in to a matrix and a vector . matrix contains the 
#independant variables while vector contains the dependannt variable

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking care of the missing data

imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

#Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# splitting the set in to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler = StandardScaler()
Y_train = scaler.fit_transform(y_train)