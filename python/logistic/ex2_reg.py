# -*- coding: utf-8 -*-
'''
Created on 2019年6月1日

@author: wukai
'''
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import loadtxt
import os
from calcu import * 

print("Plotting data....")
# os.chdir(os.path.realpath('.'))
os.chdir(os.path.split(os.path.abspath(__file__))[0])
data = loadtxt("ex2data2.txt", delimiter=",")
X = data[:, 0 : 2]
y = data[:, 2 : 3]

plotData(X, y)

X = mapFeature(X[:, 0 : 1], X[:, 1 : 2], needNdArrary = True)
# X = np.hstack(( np.ones((m, 1)), X))
(m, n) = X.shape

initial_theta = np.zeros(n)
lam = 1
cost = costFunctionReg(initial_theta, X, y, lam)
grad = gradientReg(initial_theta, X, y, lam)
print('\nCost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta: ')
print(' ', grad[0 : 5])
print('Expected gradients (approx) - first five value:\n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

test_theta = np.linspace(1.0, 1.0, n)
cost = costFunctionReg(test_theta, X, y, 10.0)
print(' ', test_theta[0])
grad = gradientReg(test_theta, X, y, 10.0)
print(' ', test_theta[0])
print('\nCost at test theta(with lambda = 10): ', cost)
print('Expected cost (approx): 3.16')
print('Gradient at test theta: ')
print(' ', grad[0 : 5])

print('Expected gradients (approx) - first five value:\n 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')


print('\nUse builtin function to find optimal patameter theta')
initial_theta = np.zeros(n)
lam = 1.0;
result = opt.minimize(fun = costFunctionReg, x0 = initial_theta, args = (X, y, lam),  method='Newton-CG', jac = gradientReg)
print(result)

print('Plot the decision boundary')
plotDecisionBoundary(result.x, X, y)

p = predict(result.x, X)
py = (p == y)
acc = len(py[(py == True)]) * 100.0 / len(y)
print('Train Accuracy: ', acc)
print('Expected accuracy (approx): 83.1')



plt.show()







