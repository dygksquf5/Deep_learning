# 시그모이드 함수

import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 이렇게 주면 넘파이 배열을 넘겨도 브로드캐스트 기능으로 배열을 전부 다 계산해줌!! 넘파이첵오


x = np.arange(-5.0, 5.0, 0.1)
print(x)
y = sigmoid(x)
print(y)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # 축 범위 지정
plt.show()



