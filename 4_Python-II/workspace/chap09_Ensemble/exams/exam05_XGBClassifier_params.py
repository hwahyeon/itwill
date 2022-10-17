# -*- coding: utf-8 -*-
"""
문) wine dataset을 이용하여 다음과 같이 다항분류 모델을 생성하시오. 
 <조건1> tree model 200개 학습
 <조건2> tree model 학습과정에서 조기 종료 100회 지정
 <조건3> model의 분류정확도와 리포트 출력   
"""
from xgboost import XGBClassifier # model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine # 다항분류
from sklearn.metrics import accuracy_score, classification_report


#################################
## 1. XGBoost Hyper Parameter
#################################

# 1. dataset load

print(wine.feature_names) # 13개 
print(wine.target_names) # ['class_0' 'class_1' 'class_2']


X, y = load_wine(return_X_y = True)
X.shape #(178, 13)
y # 0 or 1



# 2. train/test 생성 
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                        test_size = 0.3)

type(X_train) # numpy.ndarray

# 3. model 객체 생성
obj = XGBClassifier()
print(obj) # parameter 확인 
obj = XGBClassifier(colsample_bytree=1,
                    learning_rate=0.1,
                    max_depth=3, 
                    min_child_weight=1,
                    n_estimators=200, 
                    objective="multi:softprob",
                    num_class=3)

model = xgb.fit(x_train, y_train) # 1~200개의 트리 학습
model

# 4. model 학습 조기종료 
evals = [(X_test, y_test)] # 평가셋
model = obj.fit(X_train, y_train,
                eval_metric='merror',
                early_stopping_rounds=100,
                eval_set=evals,
                verbose=True)

model # objective='multi:softprob'

# 5. model 평가 
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy =', acc) # accuracy = 0.944444444444444444

report = classification_report(y_test, y_pred)
print(report)

score = model_early.score(x_test, y_test)
score




# Best Hyper Paramger : Grid Search
from sklearn.model_selection import GridSearchCV

# default model object
xgb = XGBClassifier()

params = {'colsample_bylevel' : [0.6, 0.8, 1],
          'learning_rate':[0.01, 0.1, 0.5],
          'max_depth' : [3, 5, 7],
          'min_child_weight' : [1, 3, 5],
          'n_estimators' : [100, 300, 500]}

gs = GridSearchCV(estimator=xgb, param_grid=params, cv=5)


model = gs.fit(x_train, y_train, eval_set=eval_set,
               eval_metric='error', verbose=True)

# best score
model.best_score_ #0.9597151898734175

# best parameter
model.best_params_















