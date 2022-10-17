# -*- coding: utf-8 -*-
"""
step04_@tf.function2.py

- Tensorflow2.0 특징
    3. @tf.function 함수 장식자(데코레이터)
        - 여러 함수들을 포함하는 main 함수 
"""

import tensorflow as tf

# model 생성 함수
def linear_model(x) :
    return x * 2 + 0.2 # 회귀방정식

# model 오차 함수
def model_err(y, y_pred) :
    return y - y_pred # 오차

# model 평가 함수 : main // main 함수에만 @tf.function을 붙여도 된다.
@tf.function
def model_evaluation(x, y) :
    y_pred = linear_model(x) # 함수 호출
    err = model_err(y, y_pred) # 함수 호출
    return tf.reduce_mean(tf.square(err)) # mse

# x, y data 생성
X = tf.constant([1, 2, 3], dtype = tf.float32)
Y = tf.constant([2, 4, 6], dtype = tf.float32)

MSE = model_evaluation(X, Y)
print("MSE = %.5f"%(MSE)) # MSE = 0.04000


# 1.0에선 @tf.function를 붙이면 실행이 안되며,
# 2.0에선 실행이 되긴 하나 이런 식으로 하면 그래프에서 보이지 않는다. 
def add(a, b):
    return a + b

a = tf.constant(1)
print(a)
print(add(a, 1))
# 이처럼 2버젼에서는 @tf.function을 안붙여도 실행은 된다.











