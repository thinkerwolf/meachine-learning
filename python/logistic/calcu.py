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
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(x=posX[:, 0], y=posX[:, 1], marker="+", label="Admitted")
    ax.scatter(x=negX[:, 0], y=negX[:, 1], marker="o", label="Not admitted")
    ax.legend(loc="upper right")
    return ax

'''
预测函数
'''
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g
'''
代价函数
'''
def costFunction(theta : np.ndarray, X : np.ndarray, y : np.ndarray):
    m, n = X.shape
    J = 0
    # YOUR CODE HERE
    theta = np.reshape(theta, (n, 1))
    z = X @ theta
    g = sigmoid(z) # m * 1
    J = 1.0 / m * np.sum(-1 * y * np.log(g) - (1.0 - y) * np.log(1.0 - g))
    return J

'''
梯度
'''
def gradient(theta : np.ndarray, X : np.ndarray, y : np.ndarray):
    m, n = X.shape
    theta = np.reshape(theta, (n, 1))
    z = X @ theta
    g = sigmoid(z) # m * 1
    grad = (1.0 / m * (np.dot(X.T, (g - y))))[:, 0] # n * m * m * 1
    # print(grad)
    return grad

'''
绘制决策边界
'''
def plotDecisionBoundary(theta : np.ndarray, X : np.ndarray, y : np.ndarray):
    ax = plotData(X[:, 1 : 3], y)
    n = X.shape[1]
    if n <= 3:
        # 只需要两个点决定一条线，选择两个终点
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2] )
        plot_y = (-1.0 / theta[2]) * ( theta[1] * plot_x + theta[0])
        ax.plot(plot_x, plot_y, label="Decision boundary")
        ax.legend(loc="upper right")

def predict(theta : np.ndarray, X : np.ndarray):
    m, n = X.shape
    p =  np.zeros((m, 1))
    theta = np.reshape(theta, (n, 1))
    h = X @ theta
    g = sigmoid(h)[:, 0]
    for i in range(m):
        if g[i] >= 0.5:
            p[i][0] = 1
        else:
            p[i][0] = 0
    return p
    