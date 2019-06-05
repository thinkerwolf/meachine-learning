# -*- coding: utf-8 -*-
'''
Created on 2019年6月5日

@author: wukai
'''
import os
import scipy.io as sio
import scipy.optimize as opt
from calcu import *

os.chdir(os.path.split(os.path.abspath(__file__))[0])
data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y']
m, n = X.shape
input_layer_size = n     # Input feature nums
hidden_layer_size = 25   # 25 hidden unit   
num_labels = 10          # Label nums

weights = sio.loadmat('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
nn_params = np.append(Theta1.flatten(), Theta2.flatten())

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)
nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)
print('Cost at parameters (loaded from ex4weights): {}(this value should be about 0.287629)\n'.format(J))

print('\nChecking cost using Regularization ')
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)
print('Cost at parameters (loaded from ex4weights): {}(this value should be about 0.383770)\n'.format(J))


print('\nEvaluating sigmoid gradient...')
g = sigmoidGradient(np.array([[-1, -0.5, 0, 0.5, 1]]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:  ')
print(' ', g)


print('\nInitializing Neural Network Parameters ...')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# Unroll parameters
initial_nn_params = nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())

# Training NN
lam = 1;
result = opt.minimize(fun = nnCostFunction, 
                    x0 = initial_nn_params, 
                    args = (input_layer_size, hidden_layer_size, num_labels, X, y, lam), 
                    method='TNC', jac = nnGradient, options={'disp': True})
print(result)

y_pr = nnPredict(result.x, input_layer_size, hidden_layer_size, num_labels, X)
py = (y.flatten() == y_pr)
acc = len(py[(py == True)]) * 100.0 / len(y)
print('Accuracy = {}'.format(acc))



