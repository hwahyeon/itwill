# -*- coding: utf-8 -*-
"""
공분산 vs 상관계수 (correlation)
 - 공통점 : 변수 간의 상관성 분석
 
 
1. 공분산 :두 확률변수 간의 분산(평균에서 퍼진 정도)을 나타내는 통계
 - 두 확률변수 : X, Y -> X 표본평균(ux), Y 표본평균(uy)
 Cov(X, Y) = sum((X - ux) * (Y - uy)) / n
 
 Cov(X, Y) > 0 : X증가 -> Y증가
 Cov(X, Y) < 0 : X증가 -> Y감소
 Cov(X, Y) = 0 : 두 변수는 선형관계가 아님
 문제점 : 값이 큰 변수에 영향을 받는다. 이런 문제를 해결하기 위해 상관계수를 사용한다.

2. 상관계수 : 공분산을 각각의 표준편차로 나누어서 정규화한 통계
 - 부호는 공분산과 동일하고, -1 ~ +1(절대값을 1을 넘지 않음)
 - Cor(X, Y) = Cov(X, Y) / std(X) * std(Y)
"""

import numpy as np
import pandas as pd

score_iq = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\score_iq.csv")
score_iq.info()

cor = score_iq.corr()
cor

'''
              sid     score        iq   academy      game        tv
sid      1.000000 -0.014399 -0.007048 -0.004398  0.018806  0.024565
score   -0.014399  1.000000  0.882220  0.896265 -0.298193 -0.819752
iq      -0.007048  0.882220  1.000000  0.671783 -0.031516 -0.585033
academy -0.004398  0.896265  0.671783  1.000000 -0.351315 -0.948551
game     0.018806 -0.298193 -0.031516 -0.351315  1.000000  0.239217
tv       0.024565 -0.819752 -0.585033 -0.948551  0.239217  1.000000
'''
#특정변수와 특정변수의 상관관계를 확인하기 위해서
score_iq['score'].corr(score_iq['iq']) #0.88222034461347


'''
1. 공분산
 - score vs iq
 - score vs academy
 
 Cov(X, Y) = sum((X - ux) * (Y - uy)) / n
'''

X = score_iq['score']
Y1 = score_iq['iq']
Y2 = score_iq['academy']

# 공분산을 계산하는 함수
def Cov(X, Y):
    ux = X.mean()
    uy = Y.mean()
    cov_re = sum((X - ux) * (Y - uy)) / len(X) # X와 Y의 길이는 어차피 같으니 둘 중 하나의 길이를 사용.
    return cov_re

cov1 = Cov(X, Y1) # score vs iq # 50.99528888888886
cov2 = Cov(X, Y2) # score vs academy # 7.072444444444438
print(cov1, cov2)
'''
50.99528888888886 7.072444444444438
여기서 문제점은 공분산은 값이 큰 변수의 영향을 받기 때문에 이렇게 큰 차이가 나는 것이다.

     sid  score   iq  academy  game  tv
0  10001     90  140        2     1   0
1  10002     75  125        1     3   3
2  10003     77  120        1     0   4
3  10004     83  135        2     3   2
4  10005     65  105        0     4   4

'''
score_iq.head()

score_iq.cov() #pandas의 공분산 행렬
'''
                 sid      score         iq   academy      game        tv
sid      1887.500000  -4.100671  -2.718121 -0.231544  1.208054  1.432886
score      -4.100671  42.968412  51.337539  7.119911 -2.890201 -7.214586
iq         -2.718121  51.337539  78.807338  7.227293 -0.413691 -6.972975
academy    -0.231544   7.119911   7.227293  1.468680 -0.629530 -1.543400
game        1.208054  -2.890201  -0.413691 -0.629530  2.186309  0.474899
tv          1.432886  -7.214586  -6.972975 -1.543400  0.474899  1.802640
'''

#특정변수만 확인하기 위해서
score_iq['score'].cov(score_iq['iq']) # 51.33753914988811
score_iq['score'].cov(score_iq['academy']) # 7.11991051454138

'''
2. 상관계수

Cor(X, Y) = Cov(X, Y) / std(X) * std(Y)
'''

def Cor(X, Y) :
    cov = Cov(X, Y)
    std_x = X.std()
    std_y = Y.std()
    cor_re = cov / (std_x * std_y)
    return cor_re

cor1 = Cor(X, Y1) # score vs iq
cor2 = Cor(X, Y2) # score vs academy
print(cor1, cor2)

'''
0.8763388756493802 0.8902895813918037
정규화를 한 값을 토대로 한 것이기 때문에 두 변수에 대한 상관관계가 고르게 나온다.
즉, 비슷한 상관성을 보인다.
'''





























