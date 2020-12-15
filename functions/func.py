import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c) # 오버플로 방지
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def relu(x):
    return np.maximum(0, x)


def sum_squares_error(y,t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


