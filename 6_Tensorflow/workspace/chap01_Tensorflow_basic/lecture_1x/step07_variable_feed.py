# -*- coding: utf-8 -*-
"""
step07_variable_feed.py

 2. 초기값이 없는 변수 : Feed 방식
     변수 = tf.placeholder(dtype, shape)
     - dtype : 자료형(tf.float32, tf.int32, tf.string)
     - shape : 자료구조([n] : 1차원, [r, c] : 2차원, 생략 : 공급 data 결정)
"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함

# 변수 정의
a = tf.placeholder(dtype=tf.float32) # shape 생략 : 가변형 # float도 종류가 3가지가 있다. 비트로 구별된다. 32비트가 기본.
b = tf.placeholder(dtype=tf.float32) # shape 생략 : 가변형

c = tf.placeholder(dtype=tf.float32, shape = [5]) # 고정형 : 1d
d = tf.placeholder(dtype=tf.float32, shape = [None, 3]) # 고정형 : 2d
#고정형, 2d(행 가변) // 행의 길이를 모르거나 가변적일때

# 식 정의
mul = tf.multiply(a, b)
add = tf.add(mul, 10)
c_calc = c * 0.5 # 1d(vector) * 0d(scala)

c_data = tf.random_uniform([5]) # 0~1 난수 5개 생성


with tf.Session() as sess:
    # 변수 초기화 생략
    
    # 식 실행
    mul_re = sess.run(mul, feed_dict = {a : 2.5, b : 3.5}) #data feed
    print("mul =", mul_re) # mul = 8.75
    
    # 공급 데이터
    a_data = [1.0, 2.0, 3.5]
    b_data = [0.5, 0.3, 0.4]
    feed_data = {a : a_data, b : b_data}
    
    mul_re2 = sess.run(mul, feed_dict = feed_data)
    print("mul_re2 =", mul_re2) #mul_re2 = [0.5 0.6 1.4]


    # 식 실행 : 식 참조
    # sess.run(add, feed_dict = {a : a_data, b : b_data})
    # 계속 공급 데이터를 공급해야한다.
    # multiply가 연산되도록 a와 b가 공급되어야 한다.
    # 똑같은 데이터를 여러군데서 반복적으로 사용하기 위해서 feed_data를 만들어놓고
    # 계속 사용하는 것이다.
    
    add_re = sess.run(add, feed_dict = feed_data) # mul + 10
    print("add =", add_re) # add = [10.5 10.6 11.4]


    c_data_re = sess.run(c_data) # 상수 생성
    print(c_data_re)
    print("c_calc =", sess.run(c_calc, feed_dict = {c : c_data_re}))
    # c_calc를 실행할 것이고, 그러려면 c에 c_data_re를 제공한다.
    # c_calc = [0.3329826  0.4121703  0.04045826 0.4485628  0.22904634]



# 주의 : 프로그램 정의 변수와 리턴 변수명은 다르게 지정함



