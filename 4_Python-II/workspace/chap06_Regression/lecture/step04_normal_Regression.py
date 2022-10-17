# -*- coding: utf-8 -*-
"""
data scaling(정규화, 표준화) : 이물질 제거
  - 용도 : 특정 변수의 값에 따라서 model에 영향을 미치는 경우
    ex) 범죄율(-0.1 ~ 0.99), 주택가격(99 ~ 999)
  - 정규화 : 변수의 값을 일정한 범위로 조정(0 ~ 1, -1 ~ 1) - (X변수)
            정규화 공식 nor = (x - min) / (max - min)
  - 표준화 : 평균=0과 표준편차=1를 이용 (Y변수)
          표준화 공식 z = (x - mu) / sd
"""

from sklearn.datasets import load_boston #dataset
from sklearn.linear_model import LinearRegression #model 생성
from sklearn.model_selection import train_test_split #split
from sklearn.metrics import mean_squared_error, r2_score # model 평가

import numpy as np
import pandas as pd

# 1. dataset load
X, y = load_boston(return_X_y = True)
X.shape #(506, 13)
y.shape #(506,)

# 2. data scaling
'''
X : 정규화(0~1)
y : 표준화(평균=0, 표준편차=1)
'''
X.max() #711.0
X.mean() #70.07396704469443
y.max() #50.0
y.mean() #22.532806324110677

# 정규화 함수
def normal(x):
    nor = (x - np.min(x)) / (np.max(x) - np.min(x))
    return nor

# 표준화 함수
def zscore(y):
    mu = y.mean()
    z = (y - mu) / y.std()
    return z

# X변수 정규화
x_nor = normal(X)
x_nor.mean() #0.09855691567467571

# Y변수 표준화(mu=0, st=1)
y_nor = zscore(y)
y_nor.mean() #-5.195668225913776e-16 -> 0에 수렴하고 있다.
y_nor.std() #0.9999999999999999 -> 1 수렴

# 3. dataset split(15 : 25)
x_train, x_test, y_train, y_test = train_test_split(x_nor, y_nor,
                                                    random_state = 123)
#test_size = 0.25가 default

x_train.shape #(379, 13)
x_test.shape #(127, 13)


# 4. model 생성
lr = LinearRegression() # object
model = lr.fit(X=x_train, y=y_train)
model

# 5. model 평가
#모델 평가 전 예측치 먼저 계산
y_pred = model.predict(X=x_test) #검정데이터를 넣어 Y를 예측해보는 것.

mse = mean_squared_error(y_test, y_pred) #(정답, 예측치)
score = r2_score(y_test, y_pred)
print('MSE =', mse) #MSE = 0.2933980240643525(오류율 : 30%)
print('r2 score =', score) #r2 score = 0.6862448857295749(정확률 : 70%)















