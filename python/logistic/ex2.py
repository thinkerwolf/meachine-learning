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
data = loadtxt("ex2data1.txt", delimiter=",")
X = data[:, 0 : 2]
y = data[:, 2 : 3]
plotData(X, y)
(m, n) = X.shape
X = np.hstack(( np.ones((m, 1)), X))

initial_theta = np.zeros(n + 1)
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('\nCost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta: ')
print(' ', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

test_theta = np.array([-24., 0.2, 0.2])
cost = costFunction(test_theta, X, y)
grad = gradient(test_theta, X, y)
print('\nCost at test theta: ', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta: ')
print(' ', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')


print('\nUse builtin function to find optimal patameter theta')
ini_theta = np.zeros(n + 1)

result = opt.minimize(fun = costFunction, x0 = ini_theta, args = (X, y),  method='Newton-CG',jac = gradient)
print(result)
print('Cost at theta found by minimize: ', result.fun)
print('Expected cost (approx): 0.203')
print('theta: ')
print('  ', result.x)
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201')

print('Plot the decision boundary')
plotDecisionBoundary(result.x, X, y)

p = predict(result.x, X)
py = (p == y)
acc = len(py[(py == True)]) * 100.0 / len(y)
print('Train Accuracy: ', acc)
print('Expected accuracy (approx): 89.0')



plt.show()







