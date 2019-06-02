import numpy as np

A = np.arange(5).reshape(1, 5)
print(A)
print(A.shape)
print(A.ndim)
print(A.dtype.name)
print(A.itemsize)
print(A.size)
print(type(A))

B = np.eye(N=5, M=1)
print("arnage->", np.arange(0, 10, 1))
y = np.arange(5).reshape(5, 1)
y = np.transpose(y)
print("y", y)
print(np.dot(A, B))
print(A.shape[1])
print(type(A.shape))

print(A[:, 4])
print(A[:, 0:2])

t4 = np.array([[1], [0], [1], [0], [1]])
t5 = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
print(t5[np.nonzero(t4 == 0)[0], :])
print(t5[:, 1])

print(np.sum(t5))

print(np.log(t5))

print(t5[(t5 > 4)])


