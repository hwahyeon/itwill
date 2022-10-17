'''
문) bmi.csv 데이터셋을 이용하여 다음과 같이 softmax classifier 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : height, weight 칼럼 
       -> y변수 : label(3개 범주) 칼럼
    조건2> 5,000번 학습, 500 step 단위로 loss, Accuracy 출력 
    조건3> 분류정확도(Accuracy) 출력  
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
from sklearn.preprocessing import minmax_scale # x data 정규화(0~1)
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
 
bmi = pd.read_csv('c:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# 칼럼 추출 
col = list(bmi.columns)
print(col) 

# x,y 변수 추출 
x_data = bmi[col[:2]] # x변수

# x_data 정규화 
x_data = minmax_scale(x_data)


# label one hot encoding 
label_map = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
bmi["label"] = bmi["label"].apply(lambda x : np.array(label_map[x]))

y_data = list(bmi["label"]) # 중첩list : [[1,0,0], [1,0,0]]
print(y_data[:5])

# numpy 객체 변환 
x_data = np.array(x_data)
y_data = np.array(y_data) 

print(x_data.shape) # (20000, 2)
print(y_data.shape) # (20000, 3)

# x, y변수 선언
X  = tf.placeholder(tf.float32, [None, 2]) # 키와 몸무게  
Y = tf.placeholder(tf.float32, [None, 3]) # 정답 레이블

# 가중치 : [입력층, 출력층)] 정의 
w = tf.Variable(tf.random_normal([2, 3]))
# 절편 : [출력수]
b = tf.Variable(tf.zeros([3]))

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
model = tf.matmul(X, w) + b # 회귀모델 

# softmax(예측치)
softmax = tf.nn.softmax(model) # 활성함수 적용(0~1) : y1:0.8,y2:0.1,y3:0.1

# (2) loss function : Cross Entropy 이용 : -sum(Y * log(model)) 
'''
1차 방법 
'''
loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))

# 2차 방법 : Softmmax + CrossEntropy 
'''
loss = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y, logits = model))
'''
# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2) -> decoding(10)
y_pred = tf.argmax(softmax, axis = 1) # y1:0.8,y2:0.1,y3:0.1
y_true = tf.argmax(Y, axis = 1)


# 5. model 학습 
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # w, b 초기화 
    
    feed_data = {X : x_data, Y : y_data}
    
    # 반복학습 : 5000회 
    for step in range(5000) :
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        
        # 500 step 단위 손실값과 분류정확도 출력 
        if (step+1) % 500 == 0 :            
            y_pred_re = sess.run(y_pred, feed_dict = {X : x_data}) #예측치 
            y_true_re = sess.run(y_true, feed_dict = {Y : y_data}) # 정답
            acc = accuracy_score(y_true_re, y_pred_re)
            print("setp = {}, loss = {:.6f}, acc = {}".format(
                step+1, loss_val, acc))
            
    
    # model result
    y_pred_re = sess.run(y_pred, feed_dict = {X : x_data}) #예측치 
    y_true_re = sess.run(y_true, feed_dict = {Y : y_data}) # 정답

    #print("y pred =", y_pred_re)
    #print("y ture =", y_ture_re) 
    print("-"*40)   
    acc = accuracy_score(y_true_re, y_pred_re)
    print("accuracy =", acc) # accuracy = 0.98









