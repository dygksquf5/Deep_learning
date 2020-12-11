# 신경망의 추론처리 with Mnist

import sys, os
sys.path.append(os.pardir)

import numpy as np
import pickle

from MNIST.mnist import load_mnist
from functions.func import sigmoid, softmax



def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# sample_weight.pkl 에는 '학습된 가중치 매개변수' 가 있음, 가중치와 편향 매개변수가 딕셔너리 변수로 저장 되어 있음
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# x, t = get_data()
# network = init_network()
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다
#     if p == t[i]:
#         accuracy_cnt += 1
#
#
# print("Accuracy : " + str(float(accuracy_cnt) / len(x)))


# 배치 처리를 구현 해 보기! ( 이미지 100장을 묶어서 동시에 판별해보기! )
x ,t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # axis=1 이라는 인수가 추가되면 주어진 배열중 1번째 차원을 구성하는 각 원소에서 최댓값을 인덱스로 반환하기 !!
    accuracy_cnt += np.sum(p == t[i: i+batch_size])
    # 이 마지막 부분 설명하자면 , 배치 단위로 분류한 결과를 실제 답과 비교하는건데 이를위해
    # == 연산자를 사용해서 넘파이 배열끼리 비교하면 T/F 인 bool 값이 나오는데,
    # 이결과 배열에서 True 가 몇개인지 세는것
    # 만약  y = np.array([1, 2, 1, 0])
    #      x = np.array([1, 2, 0, 0])
    # print(y == t) -> [True True False True] 이렇게 나옴 그러면
    # np.sum(y == x) -> 3 이라는 결과가 나옴! T 가 3 개 .

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))




