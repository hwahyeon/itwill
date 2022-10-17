# -*- coding: utf-8 -*-
"""
1. XGBoost Hyper Paramger
2. model 학습 조기 종료 : early stopping rounds
3. Best Hyper Paramger : Grid Search
"""

from xgboost import XGBClassifier # model
from sklearn.datasets import make_blobs # 다항분류
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import accuracy_score, classification_report # 모델 평가 도구

# 1. XGBoost Hyper Paramger
X, y = make_blobs(n_samples = 2000, n_features = 4,
                  centers = 3, cluster_std = 2.5)
X.shape #(2000, 4)
y.shape #(2000,)
y # [1, 0, 0, ..., 0, 0, 0]


x_train, x_test, y_train, y_test = train_test_split(X, y,
                                        test_size = 0.3)

xgb = XGBClassifier(colsample_bylevel=1,
                    learning_rate=0.3,
                    max_depth=3,
                    min_child_weight=1,
                    n_estimators=200,)


model = xgb.fit(x_train, y_train) # 1~200개의 트리 학습
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

model_early = xgb.fit(x_test, y_test, eval_set = eval_set,
                      eval_metric='merror',
                      early_stopping_rounds=100,
                      verbose = True)
#'merror': multi error
'''
X, y : 훈련셋
eval_set : 평가셋
eval_metric : 평가방법
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

params = {'colsample_bylevel' : [0.7, 0.9],
          'learning_rate':[0.01, 0.1],
          'max_depth' : [3, 5],
          'min_child_weight' : [1, 3],
          'n_estimators' : [100, 200]}

gs = GridSearchCV(estimator=xgb, param_grid=params, cv=5)

# 훈련셋 : x_train, y_train 평가셋 : x_test, y_test
model = gs.fit(x_train, y_train, eval_set=eval_set,
               eval_metric='merror', verbose=True)
model
# best score
model.best_score_ #0.9597151898734175

# best parameter
model.best_params_
'''
{'colsample_bylevel': 0.7,
 'learning_rate': 0.01,
 'max_depth': 3,
 'min_child_weight': 1,
 'n_estimators': 200}
'''






















