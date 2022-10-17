# -*- coding: utf-8 -*-
"""
1. 축(axis) : 행축, 열축
2. 행렬곱 연산 : np.dot()
   - 회귀방정식 = [X * a] + b
     X1, X2, -> a1, a2
     model = [X1 * a1 + X2 * a2] + b
     model = np.dot(X, a) + b
   
   - 신경망에서 행렬곱 예
     [X * w] + b
"""

import numpy as np

28*32 # 896
x_img = np.random.randint(0,256, 896)
x_img.shape #  (896,)

x_img = x_img.reshape(28,32)
x_img.shape
x_img.max() # 255
x_img = x_img / 255   # 정규화
x_img

# 1. 축(axis) : 행축, 열출
'''
행 축 : 동일한 열의 모음(axis=0) -> 열 단위
열 축 : 동일한 행의 모음(axis=1) -> 행 단위
'''

arr2d = np.random.randn(5, 4)
arr2d

print('전체 원소 합계 :', arr2d.sum())
print('행 단위 합계 :', arr2d.sum(axis=1))
#행 단위 합계 : [-0.48786343  5.58314224 -0.47179439  0.97190458 -4.73873225]
print('열 단위 합계 :', arr2d.sum(axis=0))
#열 단위 합계 : [ 4.22438501 -0.37066529 -3.14661723  0.14955427]

# 2. 행렬곱 연산 : np.dot()
X = np.array([[2,3], [2.5, 3]])
X # 입력 x
'''
array([[2. , 3. ],
       [2.5, 3. ]])
'''
X.shape # (2, 2)

a = np.array([[0.1],[0.05]]) # (2,1) // 기울기
a.shape # (2, 1)

b = 0.1 #아무 상수값을 일단 넣어봄 // 절편

y_pred = np.dot(X, a) + b
y_pred
'''
array([[0.45],
       [0.5 ]])
'''

'''
np.dot(X, a) 전제조건
1.X, a : 행렬 구조
2. 수일치 : X열 차수 = a의 행차수
'''


# [실습] p.60
X = np.array([[0.1, 0.2],[0.3, 0.4]])
X.shape # (2, 2)
X
'''
array([[0.1, 0.2],
       [0.3, 0.4]])
'''

w = np.array([[1,2,3], [2,3,4]])
w
'''
array([[1, 2, 3],
       [2, 3, 4]])
'''
w.shape #(2, 3)

# 행렬곱
h = np.dot(X, w)
h
'''
        h1   h2    h3 (히든 레이어)
array([[0.5, 0.8, 1.1],
       [1.1, 1.8, 2.5]])
'''
h.shape #(2, 3) = X(2,2) * w(2,3)

















