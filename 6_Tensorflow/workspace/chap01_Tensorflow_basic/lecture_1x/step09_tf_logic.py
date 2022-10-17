# -*- coding: utf-8 -*-
"""
step09_tf_logic.py
    - if, while 형식
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 1. if문
# 형식 : y = tf.cond(pred, true_fn, false_fn)

'''
pred : 조건식
true_fn : 조건식이 참인 경우 수행하는 함수(인수 없음)
false_fn : 조건식이 거짓인 경우 수행하는 함수(인수 없음)
'''

x = tf.constant(10) # x = 10

def true_fn():
    return tf.multiply(x, 10) # x * 10

def false_fn():
    return tf.add(x, 10) # x + 10

y = tf.cond(x > 100, true_fn, false_fn) # false

sess = tf.Session()
print("y =", sess.run(y)) # y = 20

# 2. while
i = tf.constant(0) # i = 0 : 반복변수 // 반복변수 : 리스트나 튜플로 표현됨

def cond(i):
    return tf.less(i, 100) # i < 100

def body(i):
    return tf.add(i, 1) # i += 1 // i = i + 1
    
loop = tf.while_loop(cond, body, (i,))

# 한개의 원소를 튜플로 정의할 때는 ,를 넣어줘야한다.
'''
cond : 조건식(호출가능한 함수)
body : 반복문(호출가능한 함수)
loop_vars : 반복변수(tuple or list 변수)
'''

sess = tf.Session()
print("loop =", sess.run(loop)) # loop = 100


# 1.ver는 if문이나 while문을 쓰더라도 함수를 다 지정해줘야하는 번거로움이 있다.
# 2부터는 직접 실행 환경으로 바뀌었다.








