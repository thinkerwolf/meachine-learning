# -*- coding: utf-8 -*-
'''
Created on 2019年6月4日

@author: wukai
'''
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from calcu import *

print('Loading and Visualizing Data ...')

os.chdir(os.path.split(os.path.abspath(__file__))[0])
data = sio.loadmat('ex3data1.mat')
X = data['X']  # 5000个样本  400个特性
y = data['y']
m, n = X.shape
ran = np.random.randint(0, m)
print("Display One data ")
displayOneData(X[ran, :])
print("The data should be: ", y[ran][0])
print("Display multi data ")
displayMultiData(np.random.permutation(X), 100)


print("\nTest lrCostFunction() with regulation")
theta_t = np.array([-2, -1, 1, 2])
X_t = np.column_stack((np.ones(5), np.reshape(np.linspace(1, 15, 15), (5, 3)) / 10 ))
y_t = np.array([[1], [0], [1], [0], [1]])
lam_t = 3.0
J = lrCostFunction(theta_t, X_t, y_t, lam_t)
print('Cost: ', J)
print('Expected cost: 2.590000694417512');


print('\nTraining One-vs-All Logistic Regression...')
lam = 0.1
all_theta = oneVsAll(X, y, 10, lam)

y_pr = predictOneVsAll(all_theta, X)
py = (y.flatten() == y_pr)
acc = len(py[(py == True)]) * 100.0 / len(y)
print('Accuracy = {}'.format(acc))

plt.show()




