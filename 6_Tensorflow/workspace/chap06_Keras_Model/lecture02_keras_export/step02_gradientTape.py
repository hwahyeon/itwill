# -*- coding: utf-8 -*-
"""
step02_gradientTape.py

자동 미분계수
  - tf.GadientTape() 클래스 이용 (저수준API에서 사용)
  - 역방향 step 이용 (cf : 순방향 : 연산과정 -> loss)
  - 딥러닝 모델 최적화 핵심 기술
  - 가중치(w)에 대한 오차(loss)의 미분값 계산
    -> x(w)에 대한 y(loss)의 기울기 계산
"""

import tensorflow as tf


'''
한 점 A(2, 3)를 지나는 접선의 기울기
2차 방정식 : y = x^2 + x
'''

# [실습1] x = 2
x = tf.Variable(2.0) # x = 2

with tf.GradientTape() as tape:
   y = tf.math.pow(x,2) + x # y = x ^ 2 + x
   
grad = tape.gradient(y, x) # x에 대한 y의 기울기
print("기울기 = ", grad.numpy()) # 기울기 = 5.0


# [실습2.1] x = 1.0
x = tf.Variable(1.0) # x = 1

with tf.GradientTape() as tape:
   y = tf.math.pow(x,2) + x # y = x ^ 2 + x
   
grad = tape.gradient(y, x) # x에 대한 y의 기울기
print("기울기 = ", grad.numpy()) # 기울기 = 3.0
# [정리] 미분값(기울기) 양수 -> x(w) 감소 -> 최솟점(0) 하강 

# [실습2.2] x = 0.1
x = tf.Variable(0.1) # x = 0.1

with tf.GradientTape() as tape:
   y = tf.math.pow(x,2) + x # y = x ^ 2 + x
   
grad = tape.gradient(y, x) # x에 대한 y의 기울기
print("기울기 = ", grad.numpy()) # 기울기 = 0.2
# [정리] 미분값(기울기) 양수 -> x(w) 감소 -> 최솟점 하강 


# [실습3] 미분값(기울기) 음수
x = tf.Variable(-0.5) # x = -0.5

with tf.GradientTape() as tape:
   y = tf.math.pow(x,2) + x # y = x ^ 2 + x
   
grad = tape.gradient(y, x) # x에 대한 y의 기울기
print("기울기 = ", grad.numpy()) # 기울기 = 0.0
# [정리] 미분값(기울기) 양수 -> x(w) 증가 -> 최솟점 하강











