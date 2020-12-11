# # AND
#
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1
#
#
# print(AND(0, 0)) # 0 출력
# print(AND(1, 0)) # 0 출력
# print(AND(0, 1)) # 0 출력
# print(AND(1, 1)) # 1 출력


import numpy as np

x = np.array([0, 1]) # 입력
w = np.array([0.5, 0.5]) # 가중치 (weight)
b = -0.7
print(w*x) # [0., 0.5]
print(np.sum(w*x)) #[0.5]
print(np.sum(w*x) + b) #[-0.19999999999999996] 대략 0.2 부동소수점 수에 의한 연산오차


# 가중치와 편향 구현하기 AND
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # AND 와는 가중치 (w 와 b) 만 다르다!
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 일반적은 퍼셉트론으론 XOR 을 구현할 수 없다.
# 그럴 때 다중 퍼셉트론을 활용하여 XOR 을 구현 할 수 있다.


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(0,0)) # 0을 출력
print(XOR(1,0)) # 1을 출력
print(XOR(0,1)) # 1을 출력
print(XOR(1,1)) # 0을 출력