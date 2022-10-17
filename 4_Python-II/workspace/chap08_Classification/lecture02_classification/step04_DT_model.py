# -*- coding: utf-8 -*-
"""
Decision Tree 모델
  - 중요변수 선정 기준 : GINI, Entropy
  - GINI : 불확실성을 개선하는데 기여하는 척도
  - Entropy : 불확실성을 나타내는 척도

"""

from sklearn.model_selection import train_test_split # split
from sklearn.datasets import load_iris, load_wine # dataset
from sklearn.tree import DecisionTreeClassifier # tree model

# DecisionTreeClassifier y가 범주형 -> 분류트리를 만들 때
# DecisionTreeRegressor y가 연속형  -> 회귀트리를 만들 때

from sklearn.metrics import accuracy_score, confusion_matrix # model 평가

# tree 시각화 관련
from sklearn.tree.export import export_text # tree 구조 텍스트
from sklearn import tree
from sklearn.tree import export_graphviz


iris = load_iris()
names = iris.feature_names
names
iris.target_names #['setosa', 'versicolor', 'virginica']


X = iris.data
y = iris.target

X.shape # (150, 4)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3)

help(DecisionTreeClassifier)

dtc = DecisionTreeClassifier(criterion = 'gini',
                             random_state = 123,
                             max_depth = 3)
# random_state 실행할 때마다 동일한 모델이 만들어짐.
model = dtc.fit(X = x_train, y = y_train)

tree.plot_tree(model)

print(export_text(model))
'''
|--- feature_2 <= 2.45 : 3번 칼럼 분류조건(왼쪽 노드)
|   |--- class: 0  ->  'setosa' 100% 분류
|--- feature_2 >  2.45 : 3번 칼럼 분류조건(오른쪽 노드)
|   |--- feature_3 <= 1.75
|   |   |--- feature_2 <= 4.95
|   |   |   |--- class: 1
|   |   |--- feature_2 >  4.95
|   |   |   |--- class: 2
|   |--- feature_3 >  1.75
|   |   |--- feature_2 <= 4.85
|   |   |   |--- class: 1
|   |   |--- feature_2 >  4.85
|   |   |   |--- class: 2
'''

print(export_text(model, names))
'''
|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.75
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: 1
|   |   |--- petal length (cm) >  4.95
|   |   |   |--- class: 2
|   |--- petal width (cm) >  1.75
|   |   |--- petal length (cm) <= 4.85
|   |   |   |--- class: 1
|   |   |--- petal length (cm) >  4.85
|   |   |   |--- class: 2
'''

y_pred = model.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc # 0.9555555555555556

confusion_matrix(y_true, y_pred)
'''
array([[13,  0,  0],
       [ 0, 17,  1],
       [ 0,  1, 13]], dtype=int64)
'''


dtc2 = DecisionTreeClassifier(criterion = 'entropy',
                             random_state = 123,
                             max_depth = 6)
model2 = dtc2.fit(X = x_train, y = y_train)
tree.plot_tree(model2)

y_pred = model2.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc # 0.9555555555555556

confusion_matrix(y_true, y_pred)
'''
array([[13,  0,  0],
       [ 0, 18,  0],
       [ 0,  2, 12]], dtype=int64)
'''


#######################
## model overfitting ##
#######################

wine = load_wine()
x_name = wine.feature_names # x변수 이름


X = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 123)

# default model
dt = DecisionTreeClassifier() # default model
model = dt.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
train_score #1.0
test_score = model.score(x_test, y_test)
test_score #0.9259259259259259
# 훈련데이터만큼 분류정확도가 높게 나오진 않는다.
# (이것을 오버피팅이라고 하며 일반화되지 않았다고 표현한다.)
# 오버피팅엔 tree의 depth가 영향을 준다.

tree.plot_tree(model) # max_depth = 5


# max_depth = 3
dt = DecisionTreeClassifier(max_depth = 3)
model = dt.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
train_score #0.9838709677419355
test_score = model.score(x_test, y_test)
test_score #0.9259259259259259

tree.plot_tree(model) # max_depth = 3


export_graphviz(model, out_file='DT_tree_graph.dot',
                feature_names=x_name,
                max_depth = 3,
                class_names=True)


















