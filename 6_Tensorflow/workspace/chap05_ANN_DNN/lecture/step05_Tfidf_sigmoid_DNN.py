# -*- coding: utf-8 -*-
"""
step05_Tfidf_sigmoid_DNN.py

 1. Tfidf 가중치 기법 - sparse matrix
 2. Sigmoid 활성함수 : ham(0)/spam(1)
 3. Hyper parameters
    max features = 4000 // 입력의 수 input node에 해당함
    lr = 0.01
    epochs = 50
    batch size = 500
    iter size = 10
      -> 1epoch = 500 * 10 = 5000
"""

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.x 사용 안함 
from sklearn.metrics import accuracy_score # model 평가
import numpy as np

# file load
x_train, x_test, y_train, y_test = np.load('C:/ITWILL/6_Tensorflow/data/spam_data.npy',
                                           allow_pickle=True)
print(x_train.shape) # (3901, 4000)
print(x_test.shape) # (1673, 4000)
type(x_train) # numpy.ndarray

type(y_train) # list

# list -> numpy
y_train = np.array(y_train)
y_test = np.array(y_test)
type(y_train) #  numpy.ndarray
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

#Hyper parameters
max_features = 4000 #(input node)
lr = 0.01
epochs = 50
batch_size = 500
iter_size = 10
    
# X,Y 변수 정의 
X = tf.placeholder(dtype=tf.float32, shape = [None, max_features])
Y = tf.placeholder(dtype=tf.float32, shape = [None, 1]) # ham/spam

#############################
## DNN network
#############################


hidden1_nodes = 6 
hidden2_nodes = 3

# hidden layer1 : 1층 : relu()
w1 = tf.Variable(tf.random_normal([max_features, hidden1_nodes]))#[input, output]
b1 = tf.Variable(tf.random_normal([hidden1_nodes])) # [output]
hidden1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden layer2 : 2층 : relu()
w2 = tf.Variable(tf.random_normal([hidden1_nodes, hidden2_nodes]))#[input, output]
b2 = tf.Variable(tf.random_normal([hidden2_nodes])) # [output]
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, w2) + b2)

# output layer : 3층 : sigmoid()
w3 = tf.Variable(tf.random_normal([hidden2_nodes, 1]))#[input, output]
b3 = tf.Variable(tf.random_normal([1])) # [output]


# 4. sigmoid 분류기 
# 1) 회귀방정식 : 예측치 
model = tf.matmul(hidden2_output, w3) + b3  

# sigmoid(예측치)
sigmoid = tf.sigmoid(model) # 활성함수 적용(0~1) 

# (2) loss function : 

# 1차 방법 : Cross Entropy 이용 : -sum(Y * log(model))  
#loss = -tf.reduce_mean(Y * tf.log(sigmoid) + (1 - Y) * tf.log(1 - sigmoid))

# 2차 방법 : Sigmoid + CrossEntropy
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels = Y, logits = model))

# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) cuf off
y_pred = tf.cast(sigmoid > 0.5, tf.float32)

# 6. model training
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # w, b 초기화    
    
    # epochs = 50 -> 5000 * 50 = 250,000
    for epoch in range(epochs) : # 1세대 
        tot_loss = 0
        
        # 1epoch = 500 * 10
        for step in range(iter_size) :  # 10반복 학습 
            idx = np.random.choice(a=x_train.shape[0], 
                                   size=batch_size, replace=False)
            # Mini batch dataset 
            feed_data = {X : x_train[idx], Y : y_train[idx]}
            _, loss_val = sess.run([train, loss], feed_dict = feed_data)
            
            tot_loss += loss_val
            
        # 1epoch 종료 
        avg_loss = tot_loss / iter_size
        print("epoch = {}, loss = {}".format(epoch+1, avg_loss))         
        
    # model 최적화 : test 
    feed_data2 = {X : x_test, Y : y_test}
    y_pred_re = sess.run(y_pred, feed_dict = feed_data2)
    y_true = sess.run(Y, feed_dict = feed_data2)
    
    acc = accuracy_score(y_true, y_pred_re)
    print("accuracy =", acc)






