import numpy as np

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(np.dot(A, B))


# 다차원 배열을 곱하려면 꼭 !!  두 행렬의 대응하는 차원의 원소 수를 일치 시켜야 합니다.ㅇ
C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[1,2],[3,4],[5,6]])

print(C.shape)
print(D.shape)

print(np.dot(C,D))


# 신경망에서의 행렬의 곱

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)
Y = np.dot(X, W)
print(Y)




