# 2층 신경망 클래스 구현하기

import sys, os
sys.path.append(os.pardir)

from functions.func import *
from functions.gradient import numerical_gradient


class TowLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {} # 신경망의 매개변수를 보관하는 딕셔너리 변수(인스턴스 변수)
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size) # 1번째층 가중치
        self.params['b1'] = np.zeros(hidden_size) # 1번째층 편향
        self.params['W2'] = weight_init_std *\
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x): # x 는 이미지 데이터 ! 예칭을 수행함
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x: 입력데이터, t: 정답테이블
    def loss(self, x, t): # 손실함수 구하기, 인수 x 는 이미지데이터
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x 입력데이터 , t 정답테이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {} # 기울기를 부관하는 딕셔너리 변수
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 1번째층 가중치의 기울기
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 1번째층 편향의 기울기
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
