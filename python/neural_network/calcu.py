# -*- coding: utf-8 -*-
'''
Created on 2019年6月4日

@author: wukai
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
import math
from IPython.core.pylabtools import figsize

'''
Parameters
----------
image : np.ndarray
    
'''
def displayOneData(image : np.ndarray):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

def displayMultiData(X : np.ndarray, num = 100):
    display_width = 10 # 显示宽度
    images = X[0 : 101, :]
    m, n = images.shape  # 100 * 400 
    size = int(np.sqrt(n)) # 每个图片大小 20 * 20
    display_height = math.ceil(m / display_width) # 显示高度 
    fig, axs = plt.subplots(display_width, display_height, sharex = True, sharey = True, figsize=(8, 8))
    for i in range(display_width):
        for j in range(display_height):
            image = np.reshape(images[(i * display_width + j), : ], (size, size))
            axs[i, j].matshow(image, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def lrCostFunction(theta : np.ndarray, X : np.ndarray, y : np.ndarray, lam=1.):
    m, n = X.shape
    J = 0
    # YOUR CODE HERE
    theta = np.reshape(theta, (n, 1))
    z = X @ theta
    g = sigmoid(z) # m * 1
    J = 1.0 / m * np.sum(-1 * y * np.log(g) - (1.0 - y) * np.log(1.0 - g))
    J += lam / (2 * m) * np.sum(theta[1 : ] ** 2)
    return J

'''
梯度
'''
def gradient(theta : np.ndarray, X : np.ndarray, y : np.ndarray):
    m, n = X.shape
    theta = np.reshape(theta, (n, 1))
    z = X @ theta
    g = sigmoid(z) # m * 1
    grad = (1.0 / m * (X.T @ (g - y)))[:, 0] # n * m * m * 1
    # print(grad)
    return grad

'''
正则化梯度
'''
def gradientReg(theta : np.ndarray, X : np.ndarray, y : np.ndarray, lam = 1.):
    grad = gradient(theta, X, y)
    th = (1 * lam / X.shape[0]) * (theta[1 : ])
    grad = grad + np.concatenate([np.array([0]), th])
    return grad

def oneVsAll(X : np.ndarray, y : np.ndarray, num_labels, lam = 1.0):
    m = X.shape[0]
    X = np.column_stack((np.ones(m), X))
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n)) # 10 * 400
    for i in range(num_labels):
        label = i + 1
        yt = y.copy()
        yt[yt != label] = 0
        yt[yt == label] = 1
        result = opt.minimize(fun = lrCostFunction, x0 = all_theta[i, :], args = (X, yt, lam),  method='TNC', jac = gradientReg
                              , options={'disp': True})
        all_theta[i, :] = result.x
    return all_theta

def predictOneVsAll(all_theta : np.ndarray, X : np.ndarray):
    m = X.shape[0]
    X = np.column_stack((np.ones(m), X))
    # num_labels = all_theta.shape[0]
    H = X @ all_theta.T
    prob_matrix = sigmoid(H)
    y_pred = np.argmax(prob_matrix, axis=1) + 1
    print(y_pred)
    return y_pred
    