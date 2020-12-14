import sys, os

sys.path.append(os.pardir)

import numpy as np
from MNIST.mnist import load_mnist

(x_train, t_train), (x_test, t_text) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) #(60000, 784) 훈련데이터는 6만, 입력데이터는 784열 (28 x 28)
print(t_train.shape) #(60000, 10) 정답레이블은 10줄짜리 데이터!

# 여기서 훈련데이터에서 무작위로 10장만 빼내는것은 ! ( 미니배치 뽑기!!! )
train_size = x_train.shape[0] # 6만
batch_size = 10 # 10개만 무작위로
batch_mask = np.random.choice(train_size, batch_size) # 즉 6만, 10 -> 6만중에 10개 무작위
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# 미니배치 같은 배치 데이터를 지원하는 교차 엔트로피 오차 구현하는 방법!

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 정답 레이블이 원핫코딩이 아닌 2, 7등 의 숫자 레이블로 주어졌을 때 교차 엔트로피 오차 구하기!
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size




