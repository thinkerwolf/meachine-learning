
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import loadtxt
import os
from calcu import * 

print("Plotting data....")
os.chdir(os.path.realpath("."))
data = loadtxt("ex1data1.txt", delimiter=",")
X = data[:, 0]
y = data[:, 1]
plt.scatter(X, y, marker="x")


m = len(y)
n = 1
X = np.reshape(X, (m, n))
y = np.reshape(y, (m, 1))
X = np.hstack((np.ones((m, 1)), X))
theta = np.zeros((n + 1, 1))

print("\nTesting the cost function.....")
J = computeCost(X, y, theta)
print('With theta = [0 ; 0], Cost computed = ', J)
print('Expected cost value (approx) 32.07');
J = computeCost(X, y, np.array([[-1], [2]]))
print('With theta = [-1 ; 2], Cost computed = ', J)
print('Expected cost value (approx) 54.24');


print("\nRunning Gradient Descent ...")
iterations = 1500;
alpha = 0.01;
(theta, J_history) = gradientDescent(X, y, theta, alpha, iterations)
print("Theta found by gradient descent:", theta)
print("Expected theta values (approx)")
print("  -3.6303\n  1.1664\n\n")


print("\nVisualizing J(theta_0, theta_1, ...)")
theta0_values = np.linspace(-10, 10, 100)
theta1_values = np.linspace(-1, 4, 100)
len0 = len(theta0_values)
len1 = len(theta1_values)
J_vals = np.zeros((len0, len1))

for i in range(len0):
    for j in range(len1):
        t = np.array([ [ theta0_values[i] ], [ theta1_values[j] ] ])
        cost = computeCost(X, y, t)
        J_vals[i][j] = cost
            
theta0_values, theta1_values = np.meshgrid(theta0_values, theta1_values)

J_vals = np.transpose(J_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title("J", fontsize=20)
ax.set_xlabel('theta_0', fontsize=14)
ax.set_ylabel('theta_1', fontsize=14)
ax.plot_surface(theta0_values, theta1_values, J_vals, cmap="jet",
                       linewidth=0, antialiased=False)

plt.figure()
plt.contourf(theta0_values, theta1_values, J_vals, 100, alpha=.75, cmap="hot")
plt.contour(theta0_values, theta1_values, J_vals, 100, colors='black', linewidth=.5)

plt.show()




