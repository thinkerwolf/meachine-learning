# -*- coding: utf-8 -*-
'''
Created on 2019年6月1日

@author: wukai
'''
import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    pos = np.nonzero(y == 1)[0] # tuple(ndarray...)[0]
    neg = np.nonzero(y == 0)[0]
    posX = X[pos, :]
    negX = X[neg, :]
    ax = plt.gca()
    ax.scatter(x=posX[:, 0], y=posX[:, 1], marker="+", label="Admitted")
    ax.scatter(x=negX[:, 0], y=negX[:, 1], marker="o", label="Not admitted")
    ax.legend(loc="upper right")

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g
    
def costFunction(theta : np.ndarray, X : np.ndarray, y : np.ndarray):
    m, n = X.shape
    J = 0
    # YOUR CODE HERE
    theta = np.reshape(theta, (n, 1))
    z = X @ theta
    g = sigmoid(z) # m * 1
    J = 1.0 / m * np.sum(-1 * y * np.log(g) - (1.0 - y) * np.log(1.0 - g))
    return J

def gradient(theta : np.ndarray, X : np.ndarray, y : np.ndarray):
    m, n = X.shape
    theta = np.reshape(theta, (n, 1))
    z = X @ theta
    g = sigmoid(z) # m * 1
    grad = (1.0 / m * (np.dot(X.T, (g - y))))[:, 0] # n * m * m * 1
    # print(grad)
    return grad








