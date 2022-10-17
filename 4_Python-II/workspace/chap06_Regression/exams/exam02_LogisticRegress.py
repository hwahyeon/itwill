'''
문) load_wine() 함수를 이용하여 와인 데이터를 다항분류하는 로지스틱 회귀모델을 생성하시오. 
  조건1> train/test - 7:3비울
  조건2> y 변수 : wine.data 
  조건3> x 변수 : wine.data
  조건4> 모델 평가 : confusion_matrix, 분류정확도[accuracy]
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 1. wine 데이터셋 
wine = load_wine()
wine.target_names # ['class_0', 'class_1', 'class_2']


# 2. 변수 선택 
wine_x = wine.data # x변수 
wine_y = wine.target # y변수

# 3. train/test split(7:3)
wine_x.shape #(178, 13)
wine_y.shape #(178,)

x_train, x_test, y_train, y_test = train_test_split(
    wine_x, wine_y,test_size=0.3)

x_train.shape # (124, 13)
x_test.shape # (54, 13)

# 4. model 생성  : solver='lbfgs', multi_class='multinomial'

lr = LogisticRegression(random_state=123,
                   solver='lbfgs',
                   multi_class='multinomial',
                   max_iter=200, #반복횟수
                   n_jobs=1, # 병렬처리 cpu수
                   verbose=1) #학습과정 출력여부

model = lr.fit(x_train, y_train)
model


# 5. 모델 평가 : accuracy, confusion matrix
acc = model.score(x_test, y_test) #predict 연산 -> 분류정확도 // 예측치를 넣지 않고 평가하는 방법
print('accuracy =', acc) #accuracy = 0.9444444444444444

y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred) #(정답, 예측치)로 평가하는 방법
print('accuracy =', acc)

con_mat = metrics.confusion_matrix(y_test, y_pred)
print(con_mat)
'''
[[16  1  0]
 [ 0 21  2]
 [ 0  0 14]]
'''
acc = (16+21+14) / con_mat.sum()
print('accuracy =', acc) #accuracy = 0.9444444444444444



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






