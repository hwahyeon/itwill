# -*- coding: utf-8 -*-
"""
SVM model 
 - 선형 SVM, 비선형 SVM
 - Hyper paramter(kernel, C, gamma)
"""

import pandas as pd # csv file read
from sklearn.model_selection import train_test_split # split 
from sklearn.svm import SVC  # model class
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score # model 평가  


# 1. dataset load
iris = pd.read_csv("C:/ITWILL/4_Python-II/data/iris.csv")
iris.info()


# 2. x,y변수 선택 
cols = list(iris.columns)
cols

x_cols = cols[:4] # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
y_col = cols[-1] # 'Species'


# 3. train(60)/test(40) split
train, test = train_test_split(iris, test_size=0.4)

# 4. SVM model 생성 
svc = SVC(C=1.0, gamma='auto', kernel='rbf') # 비선형 SVM model 
# default : C= 1.0, kernel='rbf'

svc2 = SVC(C=1.0, kernel='linear') # 선형 SVM model (선형/비선형은 데이터의 특성에 맞게 사용한다.)

model = svc.fit(X=train[x_cols], y=train[y_col])
model2 = svc2.fit(X=train[x_cols], y=train[y_col])

# 5. model 평가 
y_pred = model.predict(X = test[x_cols])
y_true = test[y_col]

# 비선형 SVM
acc = accuracy_score(y_true, y_pred)
print(acc)  # 0.9666666666666667 -> 0.95

# 선형 SVM
y_pred2 = model2.predict(X = test[x_cols])
y_true2 = test[y_col]

acc2 = accuracy_score(y_true2, y_pred2)
print(acc2) # 0.9666666666666667 -> 0.96


##############################
### Grid Search
##############################
#  Hyper paramter(kernel, C, gamma) C는 결정경계, gamma는 결정경계의 얼마나 피팅률을 높일것이냐를 정하는 파라미터

# Cost, gamma
params = [0.001, 0.01, 0.1, 1, 10, 100] # e-3 ~ e+2
kernel = ['linear', 'rbf'] # kernel 
best_score = 0 # best score 
best_parameter = {} # dict 

for k in kernel : 
    for gamma in params : 
        for C in params : 
            svc = SVC(kernel = k, gamma = gamma, C = C)
            model = svc.fit(train[x_cols], train[y_col])
            score = model.score(test[x_cols], test[y_col])
            
            if score > best_score :
               best_score = score 
               best_parameter = {'kernal':k,'gamma':gamma,'C':C}
               

print('best score =', best_score)       
print('best parameter=', best_parameter)        
# best score = 0.9666666666666667
# best parameter= {'kernal': 'linear', 'gamma': 0.001, 'C': 1}
# 이 데이터는 선형으로도 구별이 가능하다는 뜻이다.







