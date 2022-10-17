# -*- coding: utf-8 -*-
"""
step01_softmax_classifier_ANN.py

ANN model 
 - hidden layer : relu 활성함수 
 - output layer : Softmax 활성함수 
 - 1개 은닉층을 갖는 분류기 
 - node : 5개 
 - dataset : iris
"""
import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.x 사용 안함 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import OneHotEncoder, minmax_scale # y data 
from sklearn.metrics import accuracy_score # model 평가

# 1. x, y 공급 data 
iris = load_iris()

# x변수 : 1~4칼럼 
x_data = iris.data 
x_data.shape # (150, 4)

x_data = minmax_scale(x_data)

# y변수 : 5컬럼 
y_data = iris.target 
y_data.shape # (150,)

# reshape 
y_data = y_data.reshape(-1, 1)
y_data.shape # (150, 1)

'''
0 -> 1 0 0
1 -> 0 1 0
2 -> 0 0 1
'''

obj = OneHotEncoder()
# sparse -> numpy
y_data = obj.fit_transform(y_data).toarray()
y_data.shape # (150, 3)


# 2. X,Y변수 정의 
X = tf.placeholder(dtype=tf.float32, shape =[None,4]) # [관측치,입력수]
Y = tf.placeholder(dtype=tf.float32, shape =[None,3]) # [관측치,출력수]

#################
## ANN network ##
#################

hidden_node = 5

# hidden layer(히든레이어가 여러개면 계속 추가하면 된다. 지금은 1개)
w1 = tf.Variable(tf.random_normal([4, hidden_node])) #[input, output]
# X와 행렬곱해야하니 수일치 해야함.
b1 = tf.Variable(tf.random_normal([hidden_node])) #[output]

# output layer
w2 = tf.Variable(tf.random_normal([hidden_node, 3])) #[input, output]
b2 = tf.Variable(tf.random_normal([3])) #[output]

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
#model = tf.matmul(X, w1) + b1 # 회귀모델 // Relu 전까지 해놓은 것.
hidden_output = tf.nn.relu(tf.matmul(X, w1) + b1) # 활성함수(relu) 사용한 것

# output layer 결과
model = tf.matmul(hidden_output, w2) + b2 # 최종 모델 생성

# softmax(예측치)
softmax = tf.nn.softmax(model) # 활성함수 적용(0~1) : y1:0.8,y2:0.1,y3:0.1

# (2) loss function : 

# 1차 방법 : Cross Entropy 이용 : -sum(Y * log(model))  
# loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))

# 2차 방법 : Softmmax + CrossEntropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y, logits = model))

# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2) -> decoding(10)
y_pred = tf.argmax(softmax, axis = 1) # y1:0.8,y2:0.1,y3:0.1
y_true = tf.argmax(Y, axis = 1)



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
    y_true_re = sess.run(y_true, feed_dict = {Y : y_data}) # 정답

    #print("y pred =", y_pred_re)
    #print("y ture =", y_ture_re) 
       
    acc = accuracy_score(y_true_re, y_pred_re)
    print("accuracy =", acc) # accuracy = 0.98








