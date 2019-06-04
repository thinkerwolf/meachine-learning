'''
Created on 2019年6月4日

@author: wukai
'''
import numpy as np
import pandas as pd


y = np.linspace(1, 10, 10)
y = np.random.permutation(y)
y = np.reshape(y, (10, 1))

yt = y.copy()
yt[yt != 4] = 0
yt[yt == 4] = 1
print(yt)
print(y)


pr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
print(pr.max(1))


