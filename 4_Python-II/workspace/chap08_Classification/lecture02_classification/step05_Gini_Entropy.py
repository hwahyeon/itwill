# -*- coding: utf-8 -*-
"""
Gini 불순도(Impurity), Entropy
   - tree model에서 중요변수 선정 기준
   - 정보이득 = base 지수 - Gini 불순도 or entropy
   - 정보이득이 클수록 중요변수로 본다
   - Gini impurity = sum(p * (1 - p))
   - Entropy = -sum(p * log(p))
"""

import numpy as np #log

# 1. 불확실성 큰 경우
x1 = 0.5; x2 = 0.5 # 전체 확률 = 1

gini = sum([x1 * (1-x1), x2 * (1-x2)])
print('gini =', gini) # gini = 0.5

entropy = -sum([x1 * np.log2(x1),x2 * np.log2(x2)])
print('entropy =', entropy) # entropy = 1.0

'''
cost(loss) function : 정답과 예측치 -> 오차 변환 함수
x1 -> y_true, x2 -> y_pred
y_true = x1 * np.log2(x1)
y_pred = x2 * np.log2(x2)
'''
y_true = x1 * np.log2(x1)
y_pred = x2 * np.log2(x2)
cost = -sum([y_true, y_pred])
print('cost =', cost) # cost = 1.0


# 2. 불확실성 작은 경우
x1 = 0.9; x2 = 0.1 # 전체 확률 = 1

gini = sum([x1 * (1-x1), x2 * (1-x2)])
print('gini =', gini) # gini = 0.18

entropy = -sum([x1 * np.log2(x1),x2 * np.log2(x2)])
print('entropy =', entropy) # entropy = 0.4689955935892812

y_true = x1 * np.log2(x1)
y_pred = x2 * np.log2(x2)
cost = -sum([y_true, y_pred])
print('cost =', cost) # cost = 0.4689955935892812


#####################
###  dataset 적용  ###
#####################

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    columns = ['dark_clouds', 'gust'] # X1, X2, label
    return dataSet, columns


dataSet, columns = createDataSet()
type(dataSet) #list
dataSet = np.array(dataSet) # list -> numpy
columns #['dark_clouds', 'gust']
dataSet.shape #(5, 3)

X = dataSet[:, :2]
X
'''
array([['1', '1'],
       ['1', '1'],
       ['1', '0'],
       ['0', '1'],
       ['0', '1']], dtype='<U11')
'''

y = dataSet[:, 2]
y #array(['yes', 'yes', 'no', 'no', 'no'], dtype='<U11')

# dummy
y = [1 if d == 'yes' else 0 for d in y]
y #[1, 1, 0, 0, 0]

from sklearn.tree import DecisionTreeClassifier, export_graphviz 
from sklearn.metrics import accuracy_score
from sklearn import tree

dt = DecisionTreeClassifier('entropy') #default (gini)
model = dt.fit(X, y)


pred = model.predict(X)
model.pred
acc = accuracy_score(y, pred)
acc # 1.0

# 중요변수 
tree.plot_tree(model)

export_graphviz(model, out_file='dataset_tree.dot',max_depth=3,
                feature_names = columns)











