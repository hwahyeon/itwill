# -*- coding: utf-8 -*-
"""
step06_variable_assign.py

난수 상수 생성 함수 : 정규분포난수, 균등분포 난수
tf.Variable(난수 상수) -> 변수 값 수정

초기값을 난수로 갖게 해보는 것이 이번 스텝 목표
"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함

# 상수
num = tf.constant(10.0)

# 0차원(scala) 변수
var = tf.Variable(num + 20.0) # 상수 + 상수 =scala
print("var = ", var)
#var =  <tf.Variable 'Variable_9:0' shape=() dtype=float32_ref>
# shape=() 아무 차수가 없으니 0차원이다.

# 1차원 변수
var1d = tf.Variable(tf.random_normal([3])) # 1차원 : [n]
print("var1d = ", var1d)
#var1d =  <tf.Variable 'Variable_11:0' shape=(3,) dtype=float32_ref>

# 2차원 변수
var2d = tf.Variable(tf.random_normal([3, 2])) # 2차원 : [row,col]
print("var2d = ", var2d)
#var2d =  <tf.Variable 'Variable_14:0' shape=(3, 2) dtype=float32_ref>

# 3차원 변수
var3d = tf.Variable(tf.random_normal([3,2,4])) # 3차원 : [side,row,col]
print("var3d = ", var3d)
#var3d =  <tf.Variable 'Variable_33:0' shape=(3, 2, 4) dtype=float32_ref>


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) # 변수 초기화(초기값 할당) : var, var1d, var2d
    
    print("var =", sess.run(var)) # var = 30.3
    print("var1d =", sess.run(var1d)) # var1d = [-0.62235045 -1.4218352  -0.29592878]
    print("var2d =", sess.run(var2d))
    '''
    var2d = [[ 0.21181095  1.1174587 ]
             [-0.04767229  0.02041698]
             [-0.12762658  0.25054032]]
    '''
    
    # 변수의 값 수정
    var1d_data = [0.1, 0.2, 0.3]
    print("var1d assign_add =", sess.run(var1d.assign_add(var1d_data)))# 같은 형식을 넣어준다.
    print("var1d assign =", sess.run(var1d.assign(var1d_data)))
    # var1d assing = [0.1 0.2 0.3]

    print("var3d = ", sess.run(var3d))

    '''
    var3d =  [[[-1.4438963  -0.00833097  0.1302376   0.27404603]
               [-1.1530439  -0.6031041   0.719889   -0.6803237 ]]
           
              [[-0.5115865  -0.05755749 -2.3865855  -0.71170545]
               [ 0.41710764 -0.6555791  -0.43875965 -0.1543008 ]]
           
              [[ 0.5162361  -1.6854509   2.36436    -1.4342089 ]
               [ 0.43553814 -0.18124762  0.18138908 -0.27668837]]]
    '''

    var3d_re = sess.run(var3d)

    print(var3d_re[0]) # 첫번째 면
    print(var3d_re[0, 0]) # 첫번째 면, 첫번째 행
    print(var3d_re[0].sum()) # 첫번째 면 : 합계
    print(var3d_re[0, 0].mean()) # 첫번째 면, 첫번째 행 : 평균



    # 24개 균등분포난수를 생성하여 var3d 변수에 값을 수정하시오.
    
    var3d_data = tf.random_uniform([3,2,4])
    print("var3d assign =", sess.run(var3d.assign(var3d_data)))


'''
var3d assign = [[[0.66382587 0.22439933 0.59727454 0.304116  ]
  [0.12089884 0.36170733 0.47032547 0.35157204]]

 [[0.8652736  0.5869484  0.98661125 0.8932574 ]
  [0.25464988 0.8652648  0.80728865 0.12328196]]

 [[0.7285123  0.4569447  0.62088835 0.89311886]
  [0.80890036 0.94601834 0.47263098 0.3069222 ]]]
'''







