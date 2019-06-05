'''
Created on 2019年6月4日

@author: wukai
'''
import numpy as np
import pandas as pd


y = np.linspace(1, 10, 10)
print(y[0 : 5])

theta1 = np.array([[2, 3, 4], [5, 6, 7]])
theta2 = np.array([[1, 1, 1], [2, 3, 1]])
print(theta1 * theta1)
print(np.row_stack((theta1, theta2)))
t= np.append(theta1.flatten(), theta2.flatten())
print(t)

print(theta1[:, 1:])


