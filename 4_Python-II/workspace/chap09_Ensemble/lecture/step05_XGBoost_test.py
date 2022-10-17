# -*- coding: utf-8 -*-
"""
XGBoost model

> pip install xgboost
"""

# import test
from xgboost import XGBClassifier, XGBRegressor # model
from xgboost import plot_importance # 중요변수 시각화
from sklearn.datasets import make_blobs # 클러스터 데이터셋 생성
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import accuracy_score, classification_report # 모델 평가 도구
import matplotlib.pyplot as plt # dataset 시각화

# 1. dataset load
X, y = make_blobs(n_samples = 2000, n_features = 4,
                  centers = 3, cluster_std = 2.5)
'''
n_samples : 데이터셋 크기
n_features : X 변수
centers : Y변수 범주
cluster_std : 클러스터 표준편차(값이 클수록 오분류 커)
'''
X.shape #(2000, 4)
y.shape #(2000,)
y # [1, 0, 0, ..., 0, 0, 0]

plt.scatter(x=X[:,0], y=X[:,1], s=100, c=y, marker='o')
plt.show()

# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# 3. model 생성
xgb = XGBClassifier()
model = xgb.fit(x_train, y_train)
model
#objective = 'binary:Logistic'-> 이항분류
#objective = 'multi:softprob' -> 다항분류 (y가 여러개)
'''
n_estimators=100 : tree의 수
max_depth=6 : tree의 깊이
learning_rate=0.300000012 :작으면 작을 수록 정밀하게 학습하는 것(0.001~0.9)
'''


# 4. model 평가
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
acc

report = classification_report(y_test, y_pred)
print(report)


# 5. 중요변수 시각화
fscore = model.get_booster().get_fscore()
fscore

plot_importance(model)
plt.show()













