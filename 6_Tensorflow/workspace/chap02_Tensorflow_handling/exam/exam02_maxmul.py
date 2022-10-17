'''
텐서플로우 행렬 연산 실습 
  - ppt.16 두번째 이미지 구현 
'''

import tensorflow as tf
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error 

# x,y 행렬 정의 
x = tf.placeholder(dtype=tf.int32, shape = [None, 2])
y = tf.placeholder(dtype=tf.int32, shape = [None, 3])

# 공급 data 정의 
data1 = [[1,2], [3,4]]
data2 = [[1,2,3], [2,3,4]]

# 행렬곱 식 정의 
z = tf.matmul(x, y)

