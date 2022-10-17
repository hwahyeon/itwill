# -*- coding: utf-8 -*-
"""
step03_MNIST_DNN.py

DNN model + NMIST + Hyper parameters + Mini batch
  - Network layer
    input nodes : 28 x 28 = 784
    hidden1 nodes : 128
    hidden2 nodes : 64
    output nodes : 10개의 결과(0~9)

  - Hyper parameters
    lr : 학습률
    epoch : 전체 dataset 재사용 횟수
    batch size : 1회 data 공급 횟수(균등분할해서 mini batch하겠다는 소리)
    iter size : 반복횟수
     -> 1epoch(60,000) : batch size(200) * iter size(300)
"""

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.x 사용 안함 
from sklearn.preprocessing import OneHotEncoder # y data
from sklearn.metrics import accuracy_score # model 평가
import matplotlib.pyplot as plt
import numpy as np

# 1. MNIST dataset load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape #images(픽셀) #(60000, 28, 28) - 전체 60000개에 세로 28, 가로 28이란 뜻
y_train.shape #Labels(10진수) #(60000,)


# 첫번째 image 확인
plt.imshow(x_train[0]) # 5
plt.show()

x_train[0]
y_train[0] # 5

# 2. images 전처리 

# 1) image 정규화
x_train_nor, x_test_nor = x_train / 255.0, x_test / 255.0

x_train_nor[0] # 0~1사이로 정규화된 것을 볼 수 있다.
x_test_nor[0]

# (60000, 28, 28) -> (60000, 784) 이런 형태(2랑 3번째 것을 곱해서)로
# 3차원의 자료를 2차원으로 reshape하겠다

# 2) 3차원 -> 2차원
x_train_nor = x_train_nor.reshape(-1, 784)
x_test_nor = x_test_nor.reshape(-1, 784)

x_train_nor.shape # (60000, 784)
x_test_nor.shape # (10000, 784)

# labels 전처리
# 1) 1차원 -> 2차원
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 2) one-hot encoding
obj = OneHotEncoder()
y_train_one = obj.fit_transform(y_train).toarray()
y_test_one = obj.fit_transform(y_test).toarray()
y_train_one.shape #(60000, 10)
y_test_one.shape #(10000, 10)


# 4. 변수 정의
X = tf.placeholder(dtype = tf.float32, shape = [None, 784]) # x data
Y = tf.placeholder(dtype = tf.float32, shape = [None, 10]) # y data

# Hyper parameters
lr = 0.01 # 학습률
epochs = 20 # 전체 dataset 재사용 횟수
batch_size = 200 # 1회 data 공급 횟수(균등분할해서 mini batch하겠다는 소리)
iter_size = 300 # 반복횟수 // 총 6만이고 batch size를 200으로 뒀으니 300으로 둔다.

#############################
## DNN network
#############################

hidden1_node = 128
hidden2_node = 64

# hidden layer1 : 1층 : relu()
w1 = tf.Variable(tf.random_normal([784, hidden1_node])) #[input, output]
# 784 대신 x_train_nor.shape[1]을 넣어주는 편이 낫다.
b1 = tf.Variable(tf.random_normal([hidden1_node])) #[output]
hidden1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden layer2 : 2층 : relu()
w2 = tf.Variable(tf.random_normal([hidden1_node, hidden2_node])) #[input, output]
b2 = tf.Variable(tf.random_normal([hidden2_node])) #[output]
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, w2) + b2)

# output layer : 3층 : softmax()
w3 = tf.Variable(tf.random_normal([hidden2_node, 10])) #[input, output]
b3 = tf.Variable(tf.random_normal([10])) #[output]

# 4. softmax 분류기 
# (1) 회귀방정식 : 예측치
model = tf.matmul(hidden2_output, w3) + b3 # 최종 모델 생성

# (2)softmax(예측치)
softmax = tf.nn.softmax(model) # 활성함수 적용(0~1) : y1:0.8,y2:0.1,y3:0.1

# (3) loss function : Softmax + Cross Entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels = Y, logits = model))

# (4) optimizer 
train = tf.train.AdamOptimizer(lr).minimize(loss) 

# (5) encoding -> decoding
y_pred = tf.argmax(softmax, axis = 1)
y_true = tf.argmax(Y, axis = 1)


# 6. model training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # w, b 초기화
    
    feed_data = { X : x_train_nor, Y : y_train_one}
    
    # epoch = 20
    for epoch in range(epochs): # 한 번 돌아갈 때마다 1세대가 끝난다.
        tot_loss = 0

        # 1epoch = 200 * 300
        for step in range(iter_size): # 300번 반복 학습
            idx = np.random.choice(a=y_train_one.shape[0],
                                   size=200, replace = False)
            # size = batch size를 의미
            # Mini batch dataset
            feed_data = {X : x_train_nor[idx], Y : y_train_one[idx]}
            _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        
            tot_loss += loss_val
        
        # 1epoch 종료
        avg_loss = tot_loss / iter_size
        print("epoch = {}, loss = {}".format(epoch+1, avg_loss))
                
        
    # model test
    feed_data2 = {X : x_test_nor, Y : y_test_one}
    y_pred_re = sess.run(y_pred, feed_dict = feed_data2)
    y_true_re = sess.run(y_true, feed_dict = feed_data2)
    
    acc = accuracy_score(y_true_re, y_pred_re)
    print("accuracy = ", acc)













