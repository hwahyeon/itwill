# -*- coding: utf-8 -*-
"""
step03_SMS_spam_classficaition

NB vs SVM : 희소행렬(고차원)
  - 가중치 적용 : Tfidf방식

"""

from sklearn.naive_bayes import MultinomialNB # NB model 
from sklearn.svm import SVC # SVM model 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import accuracy_score, confusion_matrix # model 평가 
import numpy as np # npy file load # chap07/data -> spam_data.npy 


# 1. dataset load
x_train, x_test, y_train, y_test = np.load('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/spam_data.npy',
                                           allow_pickle=True)

x_train.shape #(3901, 4000) numpy객체
x_test.shape #(1673, 4000) numpy객체
y_train.shape # numpy객체가 아니라 list객체라 shape을 할 수 없다.
y_test.shape

# list -> numpy
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train.shape #(3901,)
y_test.shape #(1673,)
# 리스트는 호출, 연산에 제한이 있어 numpy로 변경하는 것이 다양한 작업하기에 유리함.


# 2. NB model : 연산이 빠른 편
nb = MultinomialNB()
model = nb.fit(X = x_train, y = y_train)

y_pred = model.predict(X = x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc 

con_mat = confusion_matrix(y_true, y_pred)
con_mat
'''
array([[1442,    0],
       [  40,  191]], dtype=int64)
'''
print(191 / (40 + 191)) #0.8268398268398268

# 3. SVM model : 가상의 초평면에 사상을 시켜 결과를 내는 것이라 연산이 느린 편. 그러나 성능이 좋음.
svc = SVC(gamma = 'auto')
svc = SVC(kernel='linear')
svc = SVC()

model_svc = svc.fit(X = x_train, y = y_train)

y_pred2 = model_svc.predict(X = x_test)
y_true2 = y_test

acc2 = accuracy_score(y_true2, y_pred2)
acc2
con_mat2 = confusion_matrix(y_true2, y_pred2)
con_mat2
'''
SVC()
array([[1439,    3],
       [  46,  185]], dtype=int64)
'''
print(185 / (46 + 185)) #0.8008658008658008

'''
SVC(kernel='linear') 속도가 빠르다.
array([[1437,    5],
       [  36,  195]], dtype=int64)
'''
print(195 / (36+195)) #0.8441558441558441














