import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(tol=0.1)

mnist = fetch_openml('mnist_784')
# print(mnist)
x = mnist['data']
y = mnist['target']
# print(x, y)
# print(x.shape)
digit = x[10000]
digit_image = digit.reshape(28,28)
plt.imshow(digit_image)
#plt.imshow(digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# print(y[10000])

# In MNIST Dataset we already have the data set splitted as 60000 (for training) and remaining 10000 (for testing)
x_train = x[:60000]
x_test = x[60000:]
y_train = y[:60000]
y_test = y[60000:]

# Shuffling the dataset to get more accuracy and precision from training set
shuffle_index = np.random.permutation(60000)
x_train = x_train[shuffle_index]
y_train = y_train[shuffle_index]

# Creating a logistic regression classifier
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
clf.fit(x_train,y_train)
y_predict = clf.predict([digit])
# print(y_predict)

a = cross_val_score(clf,x_train,y_train_2,cv=3,scoring = "accuracy")
# print(a.mean())