'''
문) weatherAUS.csv 파일을 시용하여 NB 모델을 생성하시오
  조건1> NaN 값을 가진 모든 row 삭제 
  조건2> 1,2,8,10,11,22,23 칼럼 제외 
  조건3> 7:3 비율 train/test 데이터셋 구성 
  조건4> formula 구성  = RainTomorrow ~ 나머지 변수(16개)
  조건5> GaussianNB 모델 생성 
  조건6> model 평가 : accuracy, confusion matrix, f1 score
'''
import pandas as pd
from sklearn.model_selection import train_test_split # split 
from sklearn.naive_bayes import GaussianNB, MultinomialNB # model class
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score # model 평가  


data = pd.read_csv('C:/ITWILL/4_Python-II/data/weatherAUS.csv')
print(data.head())
print(data.info())

# 조건1> NaN 값을 가진 모든 row 삭제
data=data.dropna()
print(data.head())

# 조건2> 1,2,8,10,11,22,23 칼럼 제외 
col = list(data.columns)
for i in [1,2,8,10,11,22,23] :    
    col.remove(list(data.columns)[i-1])
print(col)
len(col) # 17

# dataset 생성 
new_data = data[col]
print(new_data.head())

# 조건3> 7:3 비율 train/test 데이터셋 구성
train_set, test_set = train_test_split(
     new_data, test_size=0.3, random_state=0) # seed값 

# 조건4> formula 구성  = RainTomorrow ~ 나머지 변수(16개)
cols= list(new_data.columns)
x_cols = cols[:-1]
y_col = cols[-1] # 'RainTomorrow'

# 조건5> GaussianNB 모델 생성 
nb = GaussianNB()
model  = nb.fit(X=train_set[x_cols], y=train_set[y_col])
model

# 조건6> model 평가 : accuracy, confusion matrix, f1 score
y_pred = model.predict(X = test_set[x_cols])
y_true = test_set[y_col]

acc = accuracy_score(y_true, y_pred) # 분류정확도 
con_mat = confusion_matrix(y_true, y_pred) # 교차분할표 
help(f1_score)
f1_score = f1_score(y_true, y_pred, average='micro') # 불균형인 경우 

acc # 0.8051400076716533
con_mat
'''
array([[3391,  676],
       [ 340,  807]], dtype=int64)
'''
f1_score # 0.8051400076716534 





