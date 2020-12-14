import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
                # np.log 함수에 0 을입력하면 마이너스를 무한대로 뜻하는 -inf 되기때문에, 아주작은 앖을 더해서 절대 0 안되게
                    # 아주 작은 값인 delta 더해줌.


# 정답은 2
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 예1 : 2 일 확률이 가장 높다고 추정함 (0.6)
y =[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t))) # 0.510825457099338 교차엔트로피 오차가 약 0.51, 즉 정답에 더 가까움

# 이번에는 7 일 확률이 가장 높다고 추정 해 보기
y_2 =[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y_2), np.array(t))) # 2.302584092994546 교차엔트로피 오차가 2.30으로 정답이 너무 멈.