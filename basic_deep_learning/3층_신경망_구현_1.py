import numpy as np

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape) # (2,)
print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# A1 은 가중신호와 편향의 총합, (가중치의 합) 이며, 이를 활성화 함수로 h() 변환된 신호를 Z 로 표기한다.
Z1 = sigmoid(A1)
print(A1)
print(Z1)

# 다음 1 층에서 2층으로 가는 과정!

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3,2)
print(B2.shape) # (2,)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# 마지막 2층에서 출력층으로의 신호 전달! , 구현은 거의 다 똑같지만 활성화함수만 지금까지의 은닉층과는 다름!

# 이를 항등함수로 정의하고 , 이를 출력층의 활성화 함수로 이용함!
# 항등함수는 입력을 그대로 출력하는 함수임! ( 그냥 흐름의 통일성을 위해 생성함! )
def identity_function(x):
    return x


W3 = np.array([[0.1, 0.3],[0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # 혹은 Y = A3







