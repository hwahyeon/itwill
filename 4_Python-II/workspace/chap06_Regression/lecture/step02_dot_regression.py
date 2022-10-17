# -*- coding: utf-8 -*-
"""
행렬곱 함수(np.dot) 이용 y 예측치 구하기
  y_pred = np.dot(X, a) + b

np.dot(X, a) 전제조건
 1. X, a : 행렬 구조
 2. 수일치 : X열 차수 = a행 차수
"""

from scipy import stats # 단순회귀모델 - dot(x)
from statsmodels.formula.api import ols # 다중회귀모델 - dot(o) 
import pandas as pd
import numpy as np # array()

# 1. dataset load
score = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\score_iq.csv")
score.info()
'''
 0   sid      150 non-null    int64
 1   score    150 non-null    int64 -> y
 2   iq       150 non-null    int64 -> x1
 3   academy  150 non-null    int64 -> x2
 4   game     150 non-null    int64
 5   tv       150 non-null    int64
'''

formula = "score ~ iq + academy"
model = ols(formula, data = score).fit()

# 회귀계수 : 기울기, 절편 
model.params
'''
Intercept    25.229141
iq            0.376966
academy       2.992800
'''

# model 결과 확인 
model.summary()

# model 예측치 
model.fittedvalues

# y_pred = (X1 * a1 + X2 * a2) + b
# y_pred = np.dot(X, a) + b

X = score[['iq', 'academy']]
X.shape # (150, 2) # x1, x2

'''
np.dot(X, a) 전제조건 
 1. X, a : 행렬 구조 
 2. 수일치 : X열 차수 = a행 차수
'''

# list -> numpy
a = np.array([[0.376966],[2.992800]]) # (2, 1)
a.shape # (2, 1)

matmul = np.dot(X, a) # 행렬곱 
matmul.shape # (150, 1) = X(150, 2) . a(2, 1)

b = 25.229141 # 절편 
y_pred = matmul + b # boardcast(2차원 + 0차원)
y_pred.shape # (150, 1)

# 2차원 -> 1차원 
y_pred1d = y_pred.reshape(150)
y_pred1d.shape # (150,)

y_true = score['score']
y_true.shape # (150,)

# 데이터프레임 생성 
df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred1d})
df.head()
df.tail()

cor = df.corr()
cor # 0.972779

# 상관계수 비교 
cor = df['y_true'].corr(df['y_pred'])
cor # 0.9727792069594754
























