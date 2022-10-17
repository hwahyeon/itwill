# -*- coding: utf-8 -*-
"""
step02_function_replace.py

-Tnesorflow 2.0 특징
    2. 세션 대신 함수
        - ver2.0 : python 함수 사용 권장
        - API 정리 : tf.placeholder(dtype, shape)는 삭제됨 : 함수 인수로 대체
                    tf.random_uniform -> tf.random.uniform()
                    tf.random_normal -> tf.random.normal()
"""

import tensorflow as tf

''' step07_variable_feed.py -> ver2.0 '''

'''
# 변수 정의
a = tf.placeholder(dtype=tf.float32) # shape 생략 : 가변형 # float도 종류가 3가지가 있다. 비트로 구별된다. 32비트가 기본.
b = tf.placeholder(dtype=tf.float32) # shape 생략 : 가변형

c = tf.placeholder(dtype=tf.float32, shape = [5]) # 고정형 : 1d
d = tf.placeholder(dtype=tf.float32, shape = [None, 3]) # 고정형 : 2d
#고정형, 2d(행 가변) // 행의 길이를 모르거나 가변적일때

c_data = tf.random_uniform([5]) # 0 ~ 1 난수 

# 식 정의
mul = tf.multiply(a, b)
add = tf.add(mul, 10)
c_calc = c * 0.5 # 1d(vector) * 0d(scala)
'''

def mul_fn(a, b) : # tf.placeholder() -> 인수로 받는 방식으로 대체
    return tf.multiply(a, b)

def add_fn(mul):
    return tf.add(mul, 10)

def c_calc(c):
    return tf.multiply(c, 0.5) # c * 0.5로 해도 된다.

# data 생성
a_data = [1.0, 2.5, 3.5]
b_data = [2.0, 1.0, 4.0]

mul_re = mul_fn(a_data, b_data)
print("mul =", mul_re.numpy()) # mul = [ 2.   2.5 14. ]

print("add = {}".format(add_fn(mul_re))) # add = [12.  12.5 24. ]

# tf.random.uniform() # ver1.x
c_data = tf.random.uniform(shape=[3, 4], minval=0, maxval=1) # ver 2.x
print(c_data)

print("c_cala function : ", c_calc(c_data).numpy())

'''
c_cala function :  [[0.20585775 0.02047473 0.44582593 0.10116839]
 [0.12582666 0.19515878 0.11092913 0.4860201 ]
 [0.11182928 0.14683795 0.2067644  0.45787102]]
'''








