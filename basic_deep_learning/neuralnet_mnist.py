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


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다
    if p == t[i]:
        accuracy_cnt += 1


print("Accuracy : " + str(float(accuracy_cnt) / len(x)))


