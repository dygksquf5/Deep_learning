import numpy as np

A = np.array([1, 2, 3, 4])
print(np.ndim(A)) # 배열의 차원 확인가능함 !
print(A.shape) # 배열의 형상을 알 수 있음!  # 튜블로 반환함! 왜냐면 다차원 배열일 때도 통일된 형태로 결과를 반환하기 위함
print(A.shape[0])

B = np.array([[1,2], [3,4], [5,6]])
print(np.ndim(B))
print(B.shape) # 3x2 배열
