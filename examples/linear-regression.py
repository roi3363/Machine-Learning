#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data = pd.read_csv(os.getcwd() + '/data/countries.csv')

# append a ones column to the front of the data set


# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.array(X.values)
y = np.array(y.values)
theta = np.matrix(np.array([0,0]))

plt.scatter(X[:,1], y)
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()
results = compute_cost(X, y, theta)
plt.scatter(data['population'], data['profit'])
plt.show()
