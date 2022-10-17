# -*- coding: utf-8 -*-
"""
Pipeline vs Grid Search
  1. SVM model
  2. Pipeline : model workflow라고도 표현함. 모델을 생성할 수 있는 일련의 작업 흐름
                (dataset 전처리 -> model 생성 -> model test)
  3. Grid Search를 통해 가장 적절한 파라미터 찾기
              모델을 turning할 때 사용함. 최적의 모델을 만들어준다.
"""

from sklearn.datasets import load_breast_cancer # dataset
from sklearn.svm import SVC # model class
from sklearn.model_selection import train_test_split # split
from sklearn.preprocessing import MinMaxScaler # scaling(0~1)
from sklearn.pipeline import Pipeline # model workflow
import numpy as np


# 1. SVM model

# 1) dataset load
X, y = load_breast_cancer(return_X_y=True) # X와 y변수를 동시에 반환해주는 것
X.shape # (569, 30) 관측치는 569개

# 열 평균
X.mean(axis = 0) #0 행단위, 1 열단위
# 첫번째것과 네번째 것 추출
# 1.41272917e+01,  6.54889104e+02

X.min() #0.0
X.max() #4254.0
# 정규화가 필요하다. -> MinMaxScaler를 사용.

# 2) X변수 정규화 : 전처리
scaler = MinMaxScaler().fit(X) # 1) scaler 객체
X_nor = scaler.transform(X) # 2) 정규화 X_nor와 X 둘다 행렬
X_nor.mean(axis = 0)
X_nor.min() #0.0
X_nor.max() #1.0000000000000002


x_train, x_test, y_train, y_test = train_test_split(X_nor, y, test_size=0.3)
# X_nor : 정규화된 데이터

# 3) SVM model 생성 
svc = SVC(gamma='auto') # default SVM 모델
model = svc.fit(x_train, y_train)

# 4) model 평가
score = model.score(x_test, y_test)
score #0.9590643274853801


# 2. Pipeline : model workflow

# 1) pipeline step : [ (step1:scaler), (step2:model), ... ]
pipe_svc = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC(gamma='auto'))])

# 2) pipeline model
model = pipe_svc.fit(x_train, y_train)

# 3) pipeline model test
score = model.score(x_train, y_train)
score #0.949748743718593



# 3. Grid Search : model turning
# Pipeline -> Grid Search -> model turning
from sklearn.model_selection import GridSearchCV

help(SVC)
# C=1.0, kernel='rbf', degree=3, gamma='auto'

# 1) params 설정
params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] # 파라미터의 범위

# dict 형식 = {'object__C' : params_range} # __C 는 파이프라인의 객체를 쓰겠다는 뜻
params_grid = [{'svc__C' : params, 'svc__kernel':['linear']}, # 선형 설정
   {'svc__C' : params, 'svc__gamma': params, 'svc__kernel':['rbf']}] # 비선형 설정

# 2) GridSearchCV 객체
gs = GridSearchCV(estimator=pipe_svc, param_grid=params_grid, 
                  scoring = 'accuracy', cv=10, n_jobs=1)
# scoring : 평가방법, cv : 교차검정, n_jobs : cpu 수 




model = gs.fit(x_train, y_train)

# best score 
acc = model.score(x_test, y_test) # 검정데이터로 확인
acc #0.9824561403508771 베스트 파라미터로 얻어진 베스트 스코어

# best parameter
model.best_params_
# {'svc__C': 10.0, 'svc__gamma': 1.0, 'svc__kernel': 'rbf'}













