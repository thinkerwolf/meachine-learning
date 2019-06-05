# -*- coding: utf-8 -*-
'''
Created on 2019年6月5日

@author: wukai
'''
import numpy as np
    
'''
Use backpropagation to compute cost
Parameters
----------
ils : int
    Input Layer Size
hls : int
    Hidden Layer Size

'''
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

'''
Neural Network Cost function
'''
def nnCostFunction(nn_params : np.ndarray, ils, hls, num_labels, X : np.ndarray, y : np.ndarray, lam=0.1):
    # 还原Theta1，Theta2
    Theta1 = np.reshape(nn_params[0 : (ils + 1) * hls], (hls, ils + 1))
    Theta2 = np.reshape(nn_params[(ils + 1) * hls : nn_params.size ], (num_labels, hls + 1))
    
    
    m = np.size(X, axis = 0)
    Xt = np.column_stack((np.ones(m), X)) # 5000 * 401
    z2 = Xt @ Theta1.T  # 5000 * 401 x 401 * 25 
    a2 = sigmoid(z2)   # 5000 * 25
        
    z3 = np.column_stack((np.ones(m), a2)) @ Theta2.T # 5000 * 26 x 26 * 10 
    a3 = sigmoid(z3)   # 5000 * 10
    
    lj = 0
    for i in range(num_labels):
        l = i + 1
        yt = y.copy()
        yt[yt != l] = 0
        yt[yt == l] = 1
        a = a3[:, i : (i + 1)] # 5000 * 1
        lj += np.sum(yt * np.log(a) + (1.0 - yt) * np.log(1.0 - a))
    
    # print("lj = ", lj)
    
    # regulation
    TT1 = Theta1 * Theta1
    TT2 = Theta2 * Theta2
    TT1[ : 0] = 0
    TT2[ : 0] = 0
    rj = np.sum(TT1) + np.sum(TT2)
    
    # print('rj = ' ,lam / (2 * m) * rj)
    J = -1 / m * lj + lam / (2 * m) * rj
    return J    
        
    
    

'''
Use backpropagation to compute gradient
'''
def nnGradient(nn_params : np.ndarray, ils, hls, num_labels, X : np.ndarray, y : np.ndarray, lam=0.1):
    Theta1 = np.reshape(nn_params[0 : (ils + 1) * hls], (hls, ils + 1))
    Theta2 = np.reshape(nn_params[(ils + 1) * hls : nn_params.size ], (num_labels, hls + 1))
    m = np.size(X, axis = 0)
    
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    X = np.column_stack((np.ones(m), X))
    for i in range(m):
        a1 = X[i : i + 1, :]    # 1 * 401
        # 2. use forward propgation   
        z2 = a1 @ Theta1.T # 1 * 401 x 401 * 25
        a2 = np.column_stack((np.ones(1), sigmoid(z2))) # 1 * 26
        z3 = a2 @ Theta2.T # 1 * 26 x 26 * 10   10 * 26
        a3 = sigmoid(z3) # 1 * 10
        
        l = y[i, 0]  # label
        yk = np.zeros((1, num_labels))
        yk[0, l - 1] = 1
        
        delta3 = a3 - yk   # 1 * 10
        delta2 = (delta3 @ Theta2) * a2 * (1 - a2) 
        delta2 = delta2[:, 1 :] # 1 * 25
        
        Theta2_grad = Theta2_grad + delta3.T * a2  # 10 * 1 x 1 * 26  
        Theta1_grad = Theta1_grad + delta2.T * a1  # 25 * 1 x 1 * 401 
    
    # regulation
    # Theta1[:, 0] = 0    # this will change the value of nn_params
    # Theta2[:, 0] = 0    # this will change the value of nn_params
    Theta1_grad[:, 0:1] = (1 / m) * (Theta1_grad[:, 0:1])
    Theta1_grad[:, 1:] = (1 / m) * (Theta1_grad[:, 1:]) + lam * Theta1[:, 1:]
    
    Theta2_grad[:, 0:1] = (1 / m) * (Theta2_grad[:, 0:1])
    Theta2_grad[:, 1:] = (1 / m) * (Theta2_grad[:, 1:]) + lam * Theta2[:, 1:]
   
    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten()) 
    return grad


def randInitializeWeights(input_size, output_size):
    epsilon_init = 0.12
    # W = np.zeros((output_size, input_size + 1))
    W = np.random.random((output_size, input_size + 1)) * 2 * epsilon_init - epsilon_init
    return W


def nnPredict(nn_params : np.ndarray, ils, hls, num_labels, X : np.ndarray):
    Theta1 = np.reshape(nn_params[0 : (ils + 1) * hls], (hls, ils + 1))
    Theta2 = np.reshape(nn_params[(ils + 1) * hls : nn_params.size ], (num_labels, hls + 1))
    m = np.size(X, axis = 0)
    X = np.column_stack((np.ones(m), X))
    z2 = X @ Theta1.T  # 5000 * 400
    a2 = sigmoid(z2)   # 5000 * 25
    a2 = np.column_stack((np.ones(a2.shape[0]), a2))
    
    z3 = a2 @ Theta2.T
    h = sigmoid(z3)
    y_pred = np.argmax(h, axis=1) + 1
    return y_pred


























