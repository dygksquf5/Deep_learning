# Rectified Linear Unit 렐루 함수, Rectified 즉 정류된 선형함수이다! 근데 한국말로 너무 여러우니 .. 렐루함수라고 하자

import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x) # 즉 0보다 작거나 0과 같으면 0 출력, 아니면 해당 값 출력 !


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0 , 5.5)
plt.show()