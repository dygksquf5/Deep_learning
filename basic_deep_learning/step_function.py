# 계단함수 구현하기

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 넘파이 배열을 인수로 넣기위해 살짝 수정해보기

import numpy as np
import matplotlib.pylab as plt

#
# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)
#
# # astype 메서드는 원하는 자료형을 인수로 지정하면 됨!
#
#
# print(step_function(np.array([-1.0, 1.0])))


def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1) # -0.5 에서 5.0 까지 0.1 간격의 넘파이 배열을 생성함. 즉 [-0.5, -4.9, -4,8...]
y = step_function(x)
print(y)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()