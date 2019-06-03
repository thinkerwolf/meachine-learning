import numpy as np
import pandas as pd
from calcu import *

print("\nTest 切片  筛选..................")
t4 = np.array([[1], [0], [1], [0], [1]])
t5 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
print("1-", t5[np.nonzero(t4 == 0)[0], :])
print("2-", t5[:, 1])
print("3-", np.sum(t5))
print("4-", np.log(t5))
print("5-", t5[(t5 > 4)])

print("\n Test map feature ...............")
X1 = np.array([[3], [5], [7], [10], [11]])
X2 = np.array([[4], [9], [8], [6], [20]])
print("1-", X1[1 :, :])
C = np.zeros(X1.shape)
print('2-', C[:,::])
print('3-', mapFeature(X1, X2, needNdArrary=True))


print("\n Test pandas ............")
print('1-', pd.Series([1, 3, 5, np.nan, 6, 8]))
dates = pd.date_range('20130101', periods=6)
print('2-', dates)

df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns=list('ABCD'))
print('3-\n', df)
df2 = pd.DataFrame({
        'A' : 1.,
        'B' : pd.Timestamp('20190603'),
        'C' : pd.Series(1, index=list(range(4)), dtype='float32'),
        'D' : np.array([3] * 4, dtype="int32"),
        'F' : 'Foo'
    })
print('4-\n', df2)
print('5-', df2.dtypes)

# Display the index, columns, and the underlying NumPy data:
print('6-', df.index)
print('7-', df.columns)
print('8-', df.values)
print('9-', df2.values)








