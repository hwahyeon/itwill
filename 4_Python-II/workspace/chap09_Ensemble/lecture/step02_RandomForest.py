# -*- coding: utf-8 -*-
"""
Random Forest Ensemble model
"""

from sklearn.ensemble import RandomForestClassifier # model
from sklearn.datasets import load_wine # dataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# 1. dataset load
wine = load_wine()
wine.feature_names # X변수명
wine.target_names # y범주 이름 : ['class_0', 'class_1', 'class_2']

X = wine.data
y = wine.target

X.shape # (178, 13)

# 2. RF model
rf = RandomForestClassifier()
'''
n_estimators : integer, optional (default=100) : 트리 수
criterion : string, optional (default="gini") or "entropy" : 중요변수 선정
max_depth : integer or None, optional (default=None) : 트리 깊이
min_samples_split : int, float, optional (default=2) : 노드 분할 최소 샘플수
min_samples_leaf : int, float, optional (default=1) : 단노드 분할 최소 샘플수
max_features : int, float, string or None, optional (default="auto") : 최대 x변수 사용 수
n_jobs : int or None, optional (default=None) : cpu 수
random_state : int, RandomState instance or None, optional (default=None)
'''
rf

import numpy as np

idx = np.random.choice(a=X.shape[0], size=int(X.shape[0]*0.7),
                       replace=False)
len(idx) #124
X_train = X[idx] #X[idx, :]와 같은 뜻.
y_train = y[idx]

model = rf.fit(X=X_train, y=y_train)


idx_test = [i for i in range(len(X)) if not i in idx]
len(idx_test) #54

X_test = X[idx_test]
y_test = y[idx_test]

X_test.shape # (54, 13)
y_test.shape # (54,)

y_pred = model.predict(X_test)
y_true = y_test


con_mat = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred)

con_mat
acc
print(report)

# 중요변수
print('중요변수 :', model.feature_importances_)
'''
중요변수 : [0.12306207 0.02308869 0.01322073 0.02187131 0.02886348 0.05718473
 0.16700951 0.00426587 0.02574447 0.14959062 0.07009833 0.10652767
 0.20947253]
'''

len(model.feature_importances_) #13

# 중요변수 시각화
import matplotlib.pyplot as plt

x_size = X.shape[1]
plt.barh(range(x_size), model.feature_importances_) #(y, x)
plt.yticks(range(x_size), wine.feature_names)
plt.xlabel('importance')
plt.show()






























