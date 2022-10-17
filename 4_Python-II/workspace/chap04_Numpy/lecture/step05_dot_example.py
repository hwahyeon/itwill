# -*- coding: utf-8 -*-
"""
신경망에서 행렬곱 적용 예

 - 은닉층(h) = [입력(X) * 가중치(w)] + 편향(b)
"""

import numpy as np

# 1. ANN model
# input : image(28x28). hidden node(히든 레이어를 구성하는 노드의 개수) : 32개
# -> weight[?, ?]

# 2. input data : image data
28*28 #784
x_img = np.random.randint(0, 256, size = 784)
x_img.shape #(784,)
x_img.max() #255 당연히 범위설정을 255로 했으니까 이렇게 나오는 것.

# 이미지 정규화 : 0과 1사이로 정규화해줌.
x_img = x_img / 255
x_img.max() #1.0
x_img2d = x_img.reshape(28, 28)
x_img2d.shape #(28, 28)
# 이미지를 모델에 집어넣기 위한 전처리 과정.

# 3. weight data
'''
X = 28x28이고 h는 32니
X(28) * w = h (32) -> w = (28 x 32)이여야지 행렬곱연산이 가능하다.
'''
weight = np.random.randn(28, 32)
weight
weight.shape # (28, 32)

# 4. hidden layer
hidden = np.dot(x_img2d, weight)
hidden.shape # h(28, 32) = x(28, 28) * w(28, 32)


'''
이미지를 일정한 사이즈로 규격화해야지 이미지를 모델에 집어넣고 학습하게 할 수 있다.

'''


































