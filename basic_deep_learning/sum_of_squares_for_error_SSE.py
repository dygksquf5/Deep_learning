# 오차제곱합 !


import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 정답은 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예1 : 2 일 확률이 가장 높다고 추정함 (0.6)
y =[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t))) # 0.09750000000000003 출력이 작을수록 정답과의 오차가 적단얘기

# 이번에는 7 일 확률이 가장 높다고 추정 해 보기
y_2 =[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y_2), np.array(t))) # 0.5975 출력이 첫번째 출력문보다 훨 높음! 그만큼 오차가 크단이야기