'''
Created on 2019年6月4日

@author: wukai
'''
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from calcu import *


print('Loading data ...........')
os.chdir(os.path.split(os.path.abspath(__file__))[0])
data = sio.loadmat('ex3data1.mat')
m, n = data['X'].shape
y = data['y']
input_layer_size = n     # Input feature nums
hidden_layer_size = 25   # 25 hidden unit   
num_labels = 10          # Label nums
X = np.column_stack((np.ones(m), data['X']))

print('Loading Saved Neural Network Parameters')
params = sio.loadmat('ex3weights.mat')
print(params)

y_pr = nnPredict(params['Theta1'], params['Theta2'], X)
py = (y.flatten() == y_pr)
acc = len(py[(py == True)]) * 100.0 / len(y)
print('Accuracy = {}'.format(acc))



