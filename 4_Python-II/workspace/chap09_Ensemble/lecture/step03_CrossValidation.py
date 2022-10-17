# -*- coding: utf-8 -*-
"""
교차검정(CrossValidation)
"""

from sklearn.datasets import load_digits # dataset
from sklearn.ensemble import RandomForestClassifier # model
from sklearn.model_selection import train_test_split, cross_validate # split
from sklearn.metrics import accuracy_score

# 1. dataset load
digit = load_digits()

X = digit.data
y = digit.target

X.shape #(1797, 64)
y #array([0, 1, 2, ..., 8, 9, 8])

# 2. model
rf = RandomForestClassifier()
model = rf.fit(X, y)

pred = model.predict(X) # class 예측치
pred # array([0, 1, 2, ..., 8, 9, 8])


pred2 = model.predict_proba(X) # 확률 예측치
pred2
'''
array([[1.  , 0.  , 0.  , ..., 0.  , 0.  , 0.  ],
       [0.01, 0.96, 0.01, ..., 0.  , 0.  , 0.  ],
       [0.  , 0.04, 0.82, ..., 0.  , 0.12, 0.01],
       ...,
       [0.01, 0.08, 0.02, ..., 0.01, 0.8 , 0.02],
       [0.01, 0.01, 0.01, ..., 0.01, 0.03, 0.89],
       [0.  , 0.  , 0.02, ..., 0.  , 0.9 , 0.  ]])
'''

# 확률을 index(10진수)로 반환받기
pred2_dit = pred2.argmax(axis = 1) #어떤 특정모델은 확률로 예측해야하기에 이런 식의 방법도 필요.
pred2_dit #array([0, 1, 2, ..., 8, 9, 8], dtype=int64)

acc = accuracy_score(y, pred)
acc #1.0

acc = accuracy_score(y, pred2_dit)
acc #1.0


# 3. CrossValidation
score = cross_validate(model, X, y, scoring='accuracy', cv=5) # 5번 크로스 체킹

score
test_score = score['test_score']

# 산술평균
test_score.mean() # 0.9365784586815227



















