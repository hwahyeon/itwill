# -*- coding: utf-8 -*-
"""
step02_tf_variable_init.py
     - 변수 정의와 초기화
   
     - 상수 vs 변수 
       상수 : 수정 불가, 초기화 필요 없음 
       변수 : 수정 가능, 초기화 필요함 
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


''' 프로그램 정의 영역 '''
# 상수 정의
x = tf.constant([1.5, 2.5, 3.5], name = 'x') # 1차원 : 상수는 수정 불가
print("x :", x)
#x : Tensor("x:0", shape=(3,), dtype=float32)

# 변수 정의
y = tf.Variable([1.0, 2.0, 3.0], name='y') # 1차원 : 수정 불가
print("y :", y)
#y : <tf.Variable 'y_5:0' shape=(3,) dtype=float32_ref>

# 식 정의
mul = x * y # 상수 * 변수
# graph = node(연산자:+-*/) + edge(데이터:x,y)
# tensor : 데이터의 자료구조(scala(0), vector(1), matrix(2), array(3), n-array)

sess = tf.Session()
# 변수 초기화 객체
init = tf.global_variables_initializer()

''' 프로그램 실행 영역 '''
print("x=", sess.run(x)) # 상수 할당 : x= [1.5 2.5 3.5]
sess.run(init) # 참조 -> 변수의 값이 초기화된다.
print("y=", sess.run(y)) # 변수 할당 : 상수는 상관없지만 변수는 먼저 초기화를 해줘야한다.
#y= [1. 2. 3.]

# 식 할당
mul_re = sess.run(mul) # 식 할당(연산)
print("mul=", mul_re) # mul= [ 1.5  5.  10.5]
type(mul_re) # numpy.ndarray

print("sum =", mul_re.sum()) # sum = 17.0
# tensorflow와 numpy는 호환이 가능하다. 잘 어울린다.

sess.close()













