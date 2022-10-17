# -*- coding: utf-8 -*-
"""
python code vs tensorflow code
"""

# python : 직접 실행 환경 / tensorflow : 간접 실행 환경
x = 10
y = 20
z = x + y
print("z=", z) # z= 30

#import tensorflow as tf # ver 2.0
import tensorflow.compat.v1 as tf
# ver 2.x -> ver 1.x 현재 2.0이 설치되어있으나 1.대의 버젼을 사용하기 위해서

tf.disable_v2_behavior() # ver2.x 사용

print(tf.__version__)

'''프로그램 정의 영역'''
x = tf.constant(10) # 상수 정의
y = tf.constant(20) # 상수 정의
print(x, y)
'''
Tensor("Const_4:0", shape=(), dtype=int32) // shape=() 0차원 -> scala
Tensor("Const_5:0", shape=(), dtype=int32) // Const는 순서를 의미하는데 크게 중요한 것은 아님
'''

# 식 정의
z = x + y
print("z = ", z)
# z =  Tensor("add_1:0", shape=(), dtype=int32)

# 30이란 결과를 얻으려면 session을 통해야 한다.

# session 객체 생성 
sess = tf.Session() # 프로그램에서 정의한 상수, 변수, 식 -> device(CPU, GPU, TPU)에 할당
# sess : 사용자의 개입 없이 자료에 따라 알맞은 디바이스에 할당하는 역할이다.

'''프로그램 실행 영역'''
print("x=", sess.run(x)) #x= 10
print("y=", sess.run(y)) #y= 20
# sess.run(x, y) error

x_val, y_val = sess.run([x, y])
# x, y = sess.run([x, y]) 기존 상수의 값이 변경될 수 있다.

print(x_val, y_val) #10 20

print("z=",sess.run(z)) # x, y 상수 참조 -> 연산 z= 30

# 객체 닫기
sess.close()

# 이건 1.대의 텐서플로우고 2에선 이렇게 하지 않지만, 1을 통해 역사를 알아야하기에 수업함.












