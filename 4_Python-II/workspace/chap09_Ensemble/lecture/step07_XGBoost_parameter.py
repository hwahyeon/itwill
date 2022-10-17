# -*- coding: utf-8 -*-
"""
1. XGBoost Hyper Parameter
2. model 학습 조기 종료 : early stopping rounds
3. Best Hyper Parameter : Grid Search
"""

from xgboost import XGBClassifier # model
from xgboost import plot_importance # 중요변수 시각화
from sklearn.datasets import load_breast_cancer # y가 이항분류로 되어있는 셋. 이항분류에 적합한 셋.
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import accuracy_score, classification_report # 모델 평가 도구

# 1. dataset load
X, y = load_breast_cancer(return_X_y = True)

X.shape #(569, 30)
y # 0 or 1

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                        test_size = 0.3)
xgb = XGBClassifier(colsample_bylevel=1,
                    learning_rate=0.3,
                    max_depth=6,
                    min_child_weight=1,
                    n_estimators=100,
                   objective='binary:logistic')

'''
예를 들어서, n_estimators=100로 학습하는데
1번째 트리 0.9, 2번째 트리 0.91, 3번째 트리 0.93....식으로 증가하다가,
50번째부터 정확도가 올라가지 않는다면, 조기종료를 실행함.
'''

model = xgb.fit(x_train, y_train) # 1~100개의 트리 학습
model
'''
colsample_bylevel=1 : 트리 모델 생성시 훈련셋의 샘플링 비율(0.6 ~ 1)
learning_rate=0.300000012 : 학습률(0.01 ~ 0.1)
max_depth=6 : 트리 깊이, 과적합 영향
min_child_weight=1 : 자식 노드 분할 결정하는 최소 가중치 합, 과적합 영향
n_estimators=100 : 학습하고자 하는 트리의 수, 트리 모델 수
objective='binary:logistic' : 이항과 다항분류 결정
'''


# 2. model 학습 조기 종료 : early stopping rounds

eval_set = [(x_test, y_test)] # 평가셋

model_early = xgb.fit(x_train, y_train, eval_set = eval_set,
                      eval_metric='error',
                      early_stopping_rounds=50,
                      verbose = True)
#최대치는 100갠데 50개해보고 조기종료하라.
'''
x_train, y_train : 훈련셋
eval_set : 평가셋
eval_metric : 평가방법(이항분류 : error, 다항분류 : merror, 회귀 : rmse)
early_stopping_rounds : 학습조기종료
verbose : 학습 -> 평가 출력여부 (True를 넣으면 출력한다.)
'''

#Stopping. Best iteration:
#[11]     validation_0-error:0.00000

score = model_early.score(x_test, y_test)
score

# 3. Best Hyper Paramger : Grid Search
from sklearn.model_selection import GridSearchCV

# default model object
xgb = XGBClassifier()

params = {'colsample_bylevel' : [0.6, 0.8, 1],
          'learning_rate':[0.01, 0.1, 0.5],
          'max_depth' : [3, 5, 7],
          'min_child_weight' : [1, 3, 5],
          'n_estimators' : [100, 300, 500]}

gs = GridSearchCV(estimator=xgb, param_grid=params, cv=5)
# cv 교차검정(cross checking)


# 훈련셋 : x_train, y_train 평가셋 : x_test, y_test
model = gs.fit(x_train, y_train, eval_set=eval_set,
               eval_metric='error', verbose=True)

# best score
model.best_score_ #0.9597151898734175

# best parameter
model.best_params_
'''
{'colsample_bylevel': 0.8,
 'learning_rate': 0.5,
 'max_depth': 3,
 'min_child_weight': 1,
 'n_estimators': 100}
'''






















