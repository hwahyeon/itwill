'''
문) wine data set을 이용하여 다음과 같이 DNN 모델을 생성하시오.
  <조건1>   
   - Hidden layer : relu()함수 이용  
   - Output layer : softmax()함수 이용 
   - 2개의 은닉층을 갖는 DNN 분류기
     hidden1 : nodes = 6
     hidden2 : nodes = 3
  
  <조건2> hyper parameters
    learning_rate = 0.01
    iter_size = 1,000
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
from sklearn.datasets import load_wine # data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
import numpy as np

tf.set_random_seed(1234)  # w,b random seed

# 1. wine data load
wine = load_wine()

# 2. 변수 선택/전처리  
x_data = wine.data # 178x13
y_data = wine.target # 3개 domain
#print(y_data) # 0-2
#print(x_data.shape) # (178, 13)

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding : 0=[1,0,0] / 1=[0,1,0] / 2=[0,0,1]
num_class = np.max(y_data)+1 # 2+1

y_data = np.eye(num_class)[y_data]
print(y_data.shape) # (178, 3)

# 4. train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=123)

# 5. X,Y 변수 정의  
X = tf.placeholder(tf.float32, shape=[None, 13]) # [n, 13개 원소]
Y = tf.placeholder(tf.float32, shape=[None, 3]) # [n, 3개 원소]

# 6. Hypter parameters
lr = 0.1
iter_size = 1000
    
##############################
### DNN network
##############################
hideen1_nodes = 6
hideen2_nodes = 3

# Hidden1 layer : 1층
w1 = tf.Variable(tf.random_normal([13, hideen1_nodes])) # [X_in, h1]
b1 = tf.Variable(tf.random_normal([hideen1_nodes]))# [h1]
hidden1_out = tf.nn.relu(tf.matmul(X, w1) + b1) 

# Hidden2 layer : 2층 
w2 = tf.Variable(tf.random_normal([hideen1_nodes, hideen2_nodes]))#[h1, h2] 
b2 = tf.Variable(tf.random_normal([hideen2_nodes]))# [h2]
hidden2_out = tf.nn.relu(tf.matmul(hidden1_out, w2) + b2)

# Output layer : 3층 
w3 = tf.Variable(tf.random_normal([hideen2_nodes, 3])) #[h2, Y_out] 
b3 = tf.Variable(tf.random_normal([3])) #[Y_out]

# Softmax 알고리즘(1~3)

# 1.model 
model = tf.matmul(hidden2_out, w3) + b3 # output 계산 
# softmax(예측치)
softmax = tf.nn.softmax(model) # 0~1 확률값(전체 합=1)

# 2. cost function : sotfmax + entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=model))

# 3. Optimizer
train = tf.train.AdamOptimizer(lr).minimize(cost)

# 4. accuracy 
pred = tf.math.argmax(softmax, 1)
label = tf.math.argmax(Y, 1)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) 
    
    # 1000번 학습 
    for step in range(iter_size) :
        feed_data = {X : x_train, Y : y_train} # train data 
        _, cost_val = sess.run([train, cost], feed_dict = feed_data)
        
        if (step+1) % 100 == 0 :
            print('step=', step+1, 'cost =', cost_val)
            
    # model test : test data 
    feed_data2 = {X : x_test, Y : y_test} 
    y_pred, y_true = sess.run([pred, label], feed_dict = feed_data2)
    
    acc = accuracy_score(y_true, y_pred)
    print('test set accuracy =', acc)
    
    print(y_pred[:10])
    print(y_true[:10])






