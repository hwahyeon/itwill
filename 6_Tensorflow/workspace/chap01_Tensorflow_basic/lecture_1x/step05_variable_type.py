# -*- coding: utf-8 -*-
"""
step05_variable_type.py

Tensorflow 변수 유형
    1. 초기값을 갖는 변수 : Fetch
      변수 = tf.Variable(초기값) 초기값이 있어야 한다.
    
    2. 초기값이 없는 변수 : Feed 방식
      변수 = tf.placeholder(dtype, shape)
      나중에 향후에 들어올 자료에 대한 타입과 shape이 있어야 한다.
      shape는 생략할 수 있다.
"""
import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함

# 상수 정의
x = tf.constant(100.0)
y = tf.constant(50.0)

# 식 정의
add = tf.add(x, y) # 150 = 100 + 50

# 변수 정의
var1 = tf.Variable(add) # Fetch 방식 : 초기값
var2 = tf.placeholder(dtype = tf.float32) # Feed 방식 : 초기값(x)

# 변수 참조하는 식
mul1 = tf.multiply(x, var1)
mul2 = tf.multiply(x, var2) # var2는 연산하는 시점에서 값이 없기 때문에 연산할 수 없다.
print(mul2) # Tensor("Mul_9:0", dtype=float32)


with tf.Session() as sess:
    print("add = ", sess.run(add)) # 초기화 하기 전 먼저 식부터 실행 : add = 150
    sess.run(tf.global_variables_initializer()) # 변수 초기화(Fetch방식)
    print("var1 = ", sess.run(var1)) # 변수 생성 : var1 =  150
    # sess.run(var2, feed_dict = {변수 명 : 값})
    print("var2 = ", sess.run(var2, feed_dict = {var2 : 150}))
    # Feed 방식에서 데이터 공급하는 방법
    print("var2 = ", sess.run(var2, feed_dict = {var2 : [1.5, 2.5, 3.5]}))
    
    mul_re = sess.run(mul1) # 상수(100)와 변수(150) 참조
    print("mul_re = ", mul_re) # mul = 15000.0


    # feed 방식의 식 연산 수행
    mul_re2 = sess.run(mul2, feed_dict = {var2 : 150})
    mul_re2 = sess.run(mul2, feed_dict = {var2 : [1.5, 2.5, 3.5]}) # mul_re2 =  [150. 250. 350.]
    print("mul_re2 = ", mul_re2) # mul = 15000.0











