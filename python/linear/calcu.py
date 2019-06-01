# -*- coding: utf-8 -*-
'''
Created on 2019年5月31日

@author: wukai
'''
import numpy as np


def computeCost(X : np.ndarray, y : np.ndarray, theta : np.ndarray):
    m = len(y)
    h = np.matmul(X, theta)     # h = m * n x n * 1 = m * 1 
    sv = np.matmul(np.transpose((h - y)), (h - y))
    J = 1 / (2 * m) * sv[0][0]
    return J

def gradientDescent(X : np.ndarray, y : np.ndarray, theta : np.ndarray, alpha, iterations):
    m = len(y)
    J_history = np.linspace(0, 0, iterations)
    for i in range(iterations):
        h = np.matmul(X, theta)
        theta = theta - np.matmul(np.transpose(X), (h - y)) * (alpha * 1 / m)
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history

def featureNormalize(X : np.ndarray):
    m = X.shape[0]
    n = X.shape[1]
    print("m = ", m , ", n = ", n)
    mu = np.linspace(0, 0, n)
    sigma = np.linspace(0, 0, n)
    X_norm = np.zeros((m, n))
    for j in range(n):
        x = X[:, j]
        sigma[j] = np.std(x, ddof=1)
        mu[j] = np.mean(x)
    for j in range(n):
        avg = mu[j];
        std = sigma[j];
        for i in range(m):
            X_norm[i][j] = (X[i][j] - avg) / std
    return X_norm, mu, sigma

def normalEqn(X : np.ndarray, y : np.ndarray):
    Xt = np.transpose(X)
    Xtp = np.linalg.pinv(np.matmul(Xt, X))
    theta = np.matmul(np.matmul(Xtp, Xt), y)
    return theta



