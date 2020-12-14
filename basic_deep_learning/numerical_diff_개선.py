import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 수치 미분의 예


def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

# 그럼 만약 x = 5 일때랑 10일때 이 함수의 미분을 계산 해 보기!

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

# 이렇게 계산한 미분값이 x 에 대한 f(x)의 변화량이다. = 즉 함수의 기울기!


# 이 미분값을 기울기로 하는 직선도 함께 그려보기! 5, 10 에대한!
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
tf_2 = tangent_line(function_1, 10)
y2 = tf(x)
y3 = tf_2(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()
