import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import datasets

iris_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris = pd.read_csv(os.getcwd() + '/data/Iris.csv')
X = iris.drop('Id', axis=1)
y = X.pop('Species').map(iris_map)
# iris.drop('Species', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
model = LinearRegression()
f = model.fit(X_train, y_train)
h = model.predict(X_test)
print(r2_score(y_test, h))
h.sort()
plt.plot(h)
plt.show()
