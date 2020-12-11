import numpy as np

# a = np.array([0.3, 2.9, 4.0])
# exp_a = np.exp(a) # 지수함수
# print(exp_a)
#
# sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)
#
# y = exp_a / sum_exp_a
# print(y)


#  위의 소프트맥스 함수 논리 흐름대로 함수 정의해놓기
# 햐지만 이대로 쓰면 오버플로 일어날 수 있음 (입력값이 클 땐 )
# def softmax(a):
#     exp_a = np.ext(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y


# 하지만! 오퍼플로가 생길 수 있음 입력값이 커질수록!! 이걸 방지하기위해 C 를 식에 대입해서 임의정수를 분자와 분모 양쪽에 (더하던 빼든
# 결과는 바뀌지 않는다! 지수함수 계산할 때 ) 대입! 오버플로 막을 목적이기때문에 입력신호중 최댓값을 이용하는것이 일반적임!

# 예를들어보면 (오버플로)

# a = np.array([1010, 1000, 990])
# print ( np.exp(a) / np.sum(np.exp(a))) # [nan nan nan] 뜬다, 즉 Not a number 의 약자!
# c = np.max(a)
# print(a - c)  #c = 1010로 제공된 입력값의 최댓값
# print(np.exp(a - c) / np.sum(np.exp(a-c)))


# 오버플로를 방지하기위해 함수 다시 구현!

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y)) # 출력의 총 합이 1 이 된다는 점은 소프트맥스 함수의 중요한 성질!

# 즉 y 는 현재 [0.01821127 0.24519181 0.73659691] 라고 출력되는데, 소프트맥스함수가 총 합이 1 이 된다는건
# 결국 이를 확률로 해석할 수 있다.
# 0.01821127 는 1.8%  0.24519181는 24.5% 0.73659691는 73.7% 로 해석이 가능하다.
# 그리고 이 확률들로부터 2번째(인덱스) 원소의 확률이 가장 높으니 답은 2번째 클래스다 라고 할 수 있다.



