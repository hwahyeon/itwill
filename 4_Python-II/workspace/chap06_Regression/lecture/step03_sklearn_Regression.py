# -*- coding: utf-8 -*-
"""
sklearn 관련 Linear Regression
"""

from sklearn.linear_model import LinearRegression # model object
from sklearn.model_selection import train_test_split # train/test split
from sklearn.metrics import mean_squared_error, r2_score # model 평가 도구
from sklearn.datasets import load_diabetes #dataset
import numpy as np # 숫자 처리
import pandas as pd # data.frame, 상관계수

# 왜 불러왔는지 주석으로 표시해놓고 한번에 끌어서 일목요연하게 해놓는 것이 좋다.

#############################
### diabetes
#############################
# 단순선형회귀 : x(1) -> y

# 1. dataset load
X, y = load_diabetes(return_X_y = True)
X.shape #(442, 10)

y.shape #(442,)
y.mean() #152.13348416289594

# 2. x, y 변수 
# x(bmi : 비만도 지수) -> y

x_bmi = X[:,2]
x_bmi.shape #(442,)
x_bmi = x_bmi.reshape(442, 1) # 1d -> 2d reshape

# 3. model 생성 : object -> training -> model
# object를 먼저 만들고 학습시키고 모델을 최종 만듦
obj = LinearRegression() # 생성자 -> object
model = obj.fit(x_bmi, y) #(X, y) -> model
'''
Reshape your data either using array.
reshape(-1, 1) if your data has a single feature or array.
reshape(1, -1) if it contains a single sample.

X 부분이 vector값이면 위처럼 에러가 난다.
'''

# y 예측치
y_pred = model.predict(x_bmi) # predict(X)
y_pred.shape #(442,)
y.shape #(442,)

# 4. model 평가 : NSE(정규화), r2_score(비정규화)
MSE = mean_squared_error(y, y_pred) #(y에 대한 정답, y에 대한 예측치)
score = r2_score(y, y_pred) #r2_score(y에 대한 정답, y에 대한 예측치)
print('mse =', MSE) #mse = 3890.4565854612724
print('r2 score =', score) #r2 score = 0.3439237602253803

# 5. dataset split(70 : 30)
x_train, x_test, y_train, y_test = train_test_split(x_bmi, y, test_size = 0.3)

x_train.shape #(309, 1)
x_test.shape #(133, 1)

x_train = x_train.reshape(309, 1)

# model 생성
obj = LinearRegression() # 생성자 -> object
model = obj.fit(x_train, y_train) #training dataset

y_pred = model.predict(x_test) # test dataset

# model 평가 : MSE(정규화), r2_score(비정규화)
MSE = mean_squared_error(y_test, y_pred)
score = r2_score(y_test, y_pred) 
print('mse =', MSE) #mse = 4275.373402623988
print('r2 score =', score) # r2 수치가 1에 가까울수록 예측을 잘했다고 볼 수 있다.
#r2 score = 0.2874415662187214
y_test[:10]
y_pred[:10]

import pandas as pd # 상관계수

df = pd.DataFrame({'y_true' : y_test, 'y_pred' : y_pred})
cor = df['y_true'].corr(df['y_pred'])
cor #0.5291641791214962

import matplotlib.pyplot as plt # 회귀선 시각화
plt.plot(x_test, y_test, 'ro') #산점도
plt.plot(x_test, y_pred, 'b-') #회귀선
plt.show()

#############################
### iris.csv
#############################
# 다중회귀모델 : y(1) <- x(2~4)

iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\iris.csv")
iris.info()
'''
 0   Sepal.Length  150 non-null    float64
 1   Sepal.Width   150 non-null    float64
 2   Petal.Length  150 non-null    float64
 3   Petal.Width   150 non-null    float64
 4   Species       150 non-null    object
'''

# 2. x, y 변수 선택
cols = list(iris.columns)
cols #['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

y_col = cols[0] # 'Sepal.Length'
x_cols = cols[1:-1] # 'Sepal.Width', 'Petal.Length', 'Petal.Width'
x_cols

# 3. dataset split(75 : 25) or (70 : 30)
train_test_split(iris, test_size=0.25) # default size = 0.25
'''
test_size : 검정 데이터셋 비율(default = 0.25)
random_state : sampling seed 값
'''

iris_train, iris_test = train_test_split(iris,
                         test_size=0.3, random_state=123)
iris_train.shape # (105, 5)
iris_test.shape # (45, 5)

iris_train.head()
iris_test.head()

# 4. model 생성 : train data
lr = LinearRegression()
model = lr.fit(X=iris_train[x_cols], y=iris_train[y_col])
model #object info


# 5. model 평가 : test data
y_pred = model.predict(X=iris_test[x_cols]) #예측치
y_pred.shape #(45,)

y_true = iris_test[y_col] #관측치(정답) = label

y_true.min() #4.3
y_true.max() #7.9 어느정도 정규화가 되었다고 볼 수 있다.

# 평균제곱오차 : mean((y_true - y_pred)**2)
mse = mean_squared_error(y_true, y_pred)
# 결정계수 : 1을 기준으로 함. 1이 나오면 100% 예측한 것이다.
score = r2_score(y_true, y_pred)
print('MSE =', mse) #MSE = 0.11633863200224723(오차)
print('r2 score =', score) #r2 score = 0.8546807657451759 # 예측률이 대략 85%정도 된다는 것.

# y_true vs y_pred 시각화
type(y_true) # pandas.core.series.Series

# pandas -> numpy
y_true = np.array(y_true) #그래프 그릴 것들의 타입을 갖게 만든다.
type(y_pred) # numpy.ndarray 


# y_true vs y_pred 시각화
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10, 5))
chart = fig.subplots()
chart.plot(y_true, color = 'b', label = 'real values')
chart.plot(y_pred, color = 'r', label = 'fitted values')
plt.legend(loc = 'best')
plt.show()





















