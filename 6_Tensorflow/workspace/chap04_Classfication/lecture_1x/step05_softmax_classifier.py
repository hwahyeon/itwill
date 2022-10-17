# -*- coding: utf-8 -*-
"""
step05_softmax_classifier.py

 - 활성함수 : Softmax(model)
 - 손실함수 : Cross Entropy
"""
import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용 안함
from sklearn.metrics import accuracy_score # 분류정확도 
import numpy as np


# 1. x, y 공급 data 
# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 1], [1, 1]]) # [6, 2]

# [기타, 포유류, 조류] : [6, 3] -> one hot encoding
y_data = np.array([
    [1, 0, 0],  # 기타[0]       10진수로 확인하고 다시 2진수로 변환해준다.
    [0, 1, 0],  # 포유류[1]
    [0, 0, 1],  # 조류[2]
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 2. X, Y 변수 정의
X = tf.placeholder(dtype=tf.float32, shape = [None,2] ) #[관측치, 입력수]
Y = tf.placeholder(dtype=tf.float32, shape = [None,3] ) #[관측치, 출력수]

# 3. w, b 변수 정의 : 초기값 난수 이용
w = tf.Variable(tf.random_normal([2,3])) # [입력수, 출력수]
b = tf.Variable(tf.random_normal([3])) # [출력수]

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
model = tf.matmul(X, w) + b # 회귀모델 

# softmax(예측치)
softmax = tf.nn.softmax(model) # 활성함수 적용(0~1) : y1, y2, y3 전체 값의 합이 1이 되게끔

# (2) loss function 
# 1차 방법 : Cross Entropy 이용 : -sum(Y * log(model)) 
#loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))

# 2차 방법 : Softmmax + CrossEntropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y, logits = model))

# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2) -> decoding(10) 즉, 2진수를 10진수로
y_pred = tf.argmax(softmax, axis = 1)
y_true = tf.argmax(Y, axis = 1)
# argmax로 10진수로 변경하고서 10진수와 10진수 끼리 비교할 수 있도록 하는 것.



# 5. model 학습 
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # w, b 초기화 
    
    feed_data = {X : x_data, Y : y_data}
    
    # 반복학습 : 500회 
    for step in range(500) :
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        
        if (step+1) % 50 == 0 :
            print("setp = {}, loss = {}".format(step+1, loss_val))
    
    # model result
    y_pred_re = sess.run(y_pred, feed_dict = {X : x_data}) #예측치 
    y_ture_re = sess.run(y_true, feed_dict = {Y : y_data}) # 정답

    print("y pred =", y_pred_re)
    print("y ture =", y_ture_re) 
       
    acc = accuracy_score(y_ture_re, y_pred_re)
    print("accuracy =", acc)


'''
y pred = [0 1 2 0 0 2]
y ture = [0 1 2 0 0 2]
accuracy = 1.0
'''



















