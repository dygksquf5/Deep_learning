# MulLayer 사용하여 순전파를 구현 해 봅시다

from basic_deep_learning.layer_native import MulLayer

apple = 100
apple_num = 2
tax = 1.1 # 10퍼

# 꼐층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_pirce = mul_apple_layer.forward(apple, apple_num)
price =  mul_tax_layer.forward(apple_pirce, tax)

print(price)
 # 220


# 역전파 구해보기!
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax) # 2.2, 110, 200


