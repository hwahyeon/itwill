# -*- coding: utf-8 -*-
"""
XGBoost 회귀트리 : XGBRegressor
"""
# import test
from xgboost import XGBRegressor # model
from xgboost import plot_importance # 중요변수 시각화
from sklearn.datasets import load_boston # 주택가격
from sklearn.model_selection import train_test_split # split
from sklearn.metrics import mean_squared_error, r2_score # model 평가

# 1. dataset load
boston = load_boston()
x_names = boston.feature_names # x변수명
x_names
'''
['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT']
'''

X, y = load_boston(return_X_y = True)
X.shape #(506, 13)

y # 비율척도, 비정규화


# 2. split
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                        test_size = 0.3)

# 3. model
xgb = XGBRegressor()
model = xgb.fit(x_train, y_train)
model #objective='reg:squarederror'


# 4. 중요변수 시각화
fscore = model.get_booster().get_fscore()
fscore # 'f5' : 414

x_names[5] # 'RM'
x_names[0] # 'CRIM'

plot_importance(model)
plt.show()


# 5. model 평가
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

mse # 18.4113743503027
score # 0.7864910580517387














