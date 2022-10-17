'''
 문) load_breast_cancer 데이터 셋을 이용하여 다음과 같이 Decision Tree 모델을 생성하시오.
 <조건1> 75:25비율 train/test 데이터 셋 구성 
 <조건2> y변수 : cancer.target, x변수 : cancer.data 
 <조건3> 중요변수 확인 

'''
import pandas as pd
from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

# 데이터 셋 load 
cancer = load_breast_cancer()
print(cancer)
print(cancer.DESCR)

# 변수 선택 
X = cancer.data
y = cancer.target

names = cancer.feature_names
names

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25)

dtc = DecisionTreeClassifier(criterion = 'gini',
                             random_state = 123,
                             max_depth = 2) # 기본depth = 6
model = dtc.fit(X = x_train, y = y_train)

tree.plot_tree(model)
names[22] #'worst perimeter' y를 구별하는데 가장 중요한 변수
print(export_text(model))



#teacher

# 데이터 셋 load 
cancer = load_breast_cancer()
names = cancer.feature_names
names
len(names) # 30

print(cancer)
print(cancer.DESCR)

# 변수 선택 
X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.25)

dt = DecisionTreeClassifier(max_depth=3)
model = dt.fit(x_train, y_train)

tree.plot_tree(model)
names[22] # 'worst perimeter'







