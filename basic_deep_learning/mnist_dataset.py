import sys, os

import numpy as np
from PIL import Image  # 파이썬 이미지 라이브러리로 이미지 불러와보장

from MNIST.mnist import load_mnist


sys.path.append(os.pardir)  # 부모 디렉터리 파일 가져올 수 있도록 하기!


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

# normarlize 는 입력이미지의 픽셀값을 0.1~1.0 사이 값으로 졍규화 할지 정함. False 는 입력이미지 픽셀은 원래값대로 0~255사이 값 유지
# flatten 은 입력이미지를 평한단 1차원배열만들지 정함. False 는 입력이미지를 1X28X28의 3차원 배열로, True 는 784개의 원소로 이루어진
# 1차원 배열로 저장함
# one_hot_label 은 원핫인코딩 형태로 저장할지 정함. ex ( [0,0,1,0,0,0,0,0,0,]) 처럼 원소만 1 이고(hot하고) 나머진 모두0인배열
# 이걸 False 로 지정ㅇ하면 7, 이나 2 같이 숫자 형태의 레이블 저장함!

# print(x_train.shape)  # (60000, 784)
# print(t_train.shape)  # (60000,)
# print(x_test.shape)  # (10000, 784)
# print(t_test.shape)  # (10000,)


img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)            # (784,)
img = img.reshape(28, 28)   # 원래 이미지의 모양으로 변형
print(img.shape)            # (28, 28)


img_show(img)