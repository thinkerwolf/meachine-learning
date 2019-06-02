# -*- coding: utf-8 -*-
'''
Created on 2019 5-31

@author: wukai
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import loadtxt, dtype, double
from calcu import *
import os


# os.chdir(os.path.realpath("."))
os.chdir(os.path.split(os.path.abspath(__file__))[0])
data = loadtxt("ex1data2.txt", delimiter=",")
X = data[:, 0 : 2]
y = data[:, 2 : 3]
m = len(y)
n = X.shape[1]
print("First 5 examples from the dataset:")
print("X = ")
print(X[0 : 5, :])
print("y = ")
print(y[0 : 5, :])


print('Normalizing Features ...')
(X, mu, sigma) = featureNormalize(X)
print(X)

X = np.hstack((np.ones((m, 1)), X))

print('\nRunning gradient descent ...')
alpha = 0.01
iterations = 400
theta = np.zeros((n + 1, 1))
(theta, J_history) = gradientDescent(X, y, theta, alpha, iterations)
# Plot the convergence graph
end = len(J_history)
plt.plot(np.linspace(1, end, end), J_history, color='green', linestyle='dashed', linewidth=2, markersize=12)
plt.xlabel("number of iterateration")
plt.ylabel("cost J")

# Display gradient decent'result
print("Theta compute from gradient decent : ")
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = 0
P = np.array([[1650., 3.]])
for j in range(P.shape[1]):
    P[0][j] = (P[0][j] - mu[j]) / sigma[j]
P = np.hstack((np.ones((1, 1)), P))
print(P)
price = np.matmul(P, theta)[0][0]
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $', price)

print("\nSolveing with normal equations .......... ")

X = data[:, 0 : 2]
X = np.hstack((np.ones((m, 1)), X))
theta = normalEqn(X, y)
print("Theta compute from normal equations : ")
print(theta)

price = 0
P = np.array([[1.0, 1650., 3.]])
price = np.matmul(P, theta)[0][0]
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ', price);


plt.show()
