# -*- coding: utf-8 -*-
"""
sklearn 로지스틱 회귀모델
  - y변수가 범주형인 경우
"""

from sklearn.datasets import load_breast_cancer, load_iris #dataset
from sklearn.linear_model import LogisticRegression # model 생성
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가
import pandas as pd

##################################
### 1. 이항분류 모델
##################################

# 1. dataset load
breast = load_breast_cancer()

X = breast.data # x 변수 (X 인 이유 : 차원이 큰 쪽을 대문자로 표현한다.)
y = breast.target # y 변수 반환
y
X.shape #(569, 30)
y.shape #(569,)

# 2. model 생성
help(LogisticRegression)
'''
random_state=None, solver='lbfgs', max_iter=100, multi_class='auto'

random_state=None : 난수 seed값 지정
solver='lbfgs' : 알고리즘 (lbfgs가 기본 알고리즘default이란 뜻)
    - 'sag', 'saga' and 'newton-cg' solvers.
max_iter=100 : 반복학습 횟수
multi_class='auto' : 다항분류
    - multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'

적용 예)
일반 데이터, 이항분류 : default
일반 데이터, 다항분류 : solver='lbfgs', multi_class='multinomial'
빅 데이터, 이항분류 : solver='sag' or 'saga'
빅 데이터, 다항분류 : solver='sag' or 'saga', multi_class='multinomial'
'''

lr = LogisticRegression(random_state=123)
model = lr.fit(X = X, y = y) #전체 모든 데이터로 모델 생성
model #multi_class = 'auto' -> sigmoid 활용함수 -> 이항분류

# 3. model 평가
acc = model.score(X, y)
print('accuracy =', acc) # accuracy = 0.9472759226713533

y_pred = model.predict(X)

acc - accuracy_score(y, y_pred)
print('accuracy =', acc) #accuracy = 0.9472759226713533

#위 방법과 아래 방법은 동일한 값이 나온다.

con_mat = confusion_matrix(y, y_pred)
print(con_mat)
type(con_mat) #numpy.ndarray
'''
     0    1
0 [[193  19]
1  [ 11 346]]
'''

acc = (con_mat[0,0] + con_mat[1,1]) / con_mat.sum()
print('accuracy =', acc) #accuracy = 0.9472759226713533

import pandas as pd
tab = pd.crosstab(y, y_pred, rownames=['관측치'], colnames=['예측치'])
tab # pandas 활용
'''
예측치    0    1
관측치          
0    193   19
1     11  346
'''

acc = (tab.iloc[0,0] + tab.loc[1,1]) / len(y)
print('accuracy =', acc) #accuracy = 0.9472759226713533

###########################
### 2. 다항분류 모델
###########################

# 1. dataset load
iris = load_iris()
iris.target_names #['setosa', 'versicolor', 'virginica']

X, y = load_iris(return_X_y=True)

X.shape #(150, 4)
y.shape #(150,)
y # 0~2

# 2. model
# 일반 데이터, 다항분류 : solver='lbfgs', multi_class='multinomial'

LogisticRegression(random_state=123,
                   solver='lbfgs',
                   multi_class='multinomial')

# multi_class='multinomial' : softmax 활용함수 이용 -> 다항분류
'''
활성함수
sigmoid function : 0 ~ 1 확률값 -> cutoff = 0.5 -> 이항분류
softmax function : 0 ~ 1 확률값 -> 전체합 = 1(c0:0.1, c1:0.1, c2:0.8) -> 다항분류
'''

model = lr.fit(X, y)

y_pred = model.predict(X) # class
y_pred2 = model.predict_proba(X) # 확률값

y_pred # 0 ~ 2
y_pred2.shape #(150, 3)

#['setosa', 'versicolor', 'virginica']
#[9.81797141e-01, 1.82028445e-02, 1.44269293e-08]

import numpy as np
arr = np.array([9.81797141e-01, 1.82028445e-02, 1.44269293e-08])
arr.max() # 0.981797141
arr.min() # 1.44269293e-08
arr.sum() # 0.9999999999269293

# 3. model 평가
acc = accuracy_score(y, y_pred)

print('accuracy =', acc)
# accuracy = 0.9733333333333334

con_mat = confusion_matrix(y, y_pred)
con_mat
'''
array([[50,  0,  0],
       [ 0, 47,  3],
       [ 0,  1, 49]], dtype=int64)
'''

acc = (con_mat[0,0] + con_mat[1,1] + con_mat[2,2]) / con_mat.sum()
print('accuracy =', acc) #accuracy = 0.9733333333333334

# 히트맵 : 시각화
import seaborn as sn # heatmap - Accuracy Score

# confusion matrix heatmap 
plt.figure(figsize=(6,6)) # chart size
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square = True);# , cmap = 'Blues_r' : map »ö»ó 
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 18)
plt.show()


############################
### digits : multi class ###
############################

from sklearn.datasets import load_digits

# 1. dataset load
digits = load_digits()
digits.target_names # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X = digits.data
y = digits.target 
X.shape # (1797, 64) -> 1797장 images
y.shape # (1797,) 1797장 images 10진수 정답

# 2. split
img_train, img_test, label_train, label_test = train_test_split(
    X, y, test_size = 0.25)

import matplotlib.pyplot as plt

# 훈련셋 image -> reshape
img2d = img_train.reshape(-1, 8, 8) #(전체image, 세로픽셀, 가로픽셀) -1로 전체를 표시한다.
img2d.shape #(1347, 8, 8)
img2d[0]

plt.imshow(img2d[0])
label_train[0] #3

img_test.shape #(450, 64)


# 3. model 생성
lr = LogisticRegression(solver='lbfgs',
                   multi_class='multinomial')
model = lr.fit(img_train, label_train)

y_pred = model.predict(img_test)

# 4. model 평가
acc = accuracy_score(label_test, y_pred)
print(acc)

con_mat = confusion_matrix(label_test, y_pred)
con_mat

result = label_test == y_pred
result #False인 부분이 오분류된 부분이다.
# True -> 1, False -> 0
result.mean() 

result
len(result) #450


# 틀린 image
false_img = img_test[result==False]
false_img.shape # (12, 64)
false_img3d = false_img.reshape(-1, 8, 8)

false_img3d.shape

# Preferences -> IPython console -> Graphics -> Inline
for idx in range(false_img3d.shape[0]): #row
    #print(img.shape)
    print(idx)
    plt.imshow(false_img3d[idx])
    plt.show()

# 히트맵 : 시각화
import seaborn as sn # heatmap - Accuracy Score

# confusion matrix heatmap 
plt.figure(figsize=(6,6)) # chart size
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square = True);# , cmap = 'Blues_r' : map »ö»ó 
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title(all_sample_title, size = 18)
plt.show()






























