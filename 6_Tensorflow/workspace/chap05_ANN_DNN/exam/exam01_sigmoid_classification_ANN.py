'''
문) bmi.csv 데이터셋을 이용하여 다음과 같이 ANN 모델을 생성하시오.
  <조건1>   
   - 1개의 은닉층을 갖는 ANN 분류기
   - hidden nodes = 4
   - Hidden layer : relu()함수 이용  
   - Output layer : sigmoid()함수 이용 
     
  <조건2> hyper parameters
    최적화 알고리즘 : AdamOptimizer
    learning_rate = 0.0 ~ 0.01
    반복학습 : 300 ~ 500
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score # model 평가

import numpy as np
import pandas as pd
 
bmi = pd.read_csv('C:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# label에서 normal, fat 추출 
bmi = bmi[bmi.label.isin(['normal','fat'])]
print(bmi.head())

# 칼럼 추출 
col = list(bmi.columns)
print(col) 

# x,y 변수 추출 
x_data = bmi[col[:2]] # x변수
y_data = bmi[col[2]] # y변수

# y변수(label) 로짓 변환 dict
map_data = {'normal': 0,'fat' : 1}
y_data = y_data.map(map_data) # dict mapping

# x_data 정규화 함수 
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

x_data = x_data.apply(normalize)

# numpy 객체 변환 
x_data = np.array(x_data)
y_data = np.transpose(np.array([y_data]))# (1, 15102) -> (15102, 1)

print(x_data.shape) # (15102, 2)
print(y_data.shape) # (15102, 1)


# x,y 변수 정의 
X = tf.placeholder(tf.float32, shape=[None, 2]) # x 데이터 수
Y = tf.placeholder(tf.float32, shape=[None, 1]) # y 데이터 수 

tf.set_random_seed(1234)

##############################
### ANN network  
##############################

hidden_node = 4

# hidden layer(히든레이어가 여러개면 계속 추가하면 된다. 지금은 1개)
w1 = tf.Variable(tf.random_normal([2, hidden_node])) #[input, output]
b1 = tf.Variable(tf.random_normal([hidden_node])) #[output]

# output layer
w2 = tf.Variable(tf.random_normal([hidden_node, 1])) #[input, output]
b2 = tf.Variable(tf.random_normal([1])) #[output]

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
#model = tf.matmul(X, w1) + b1 # 회귀모델 // Relu 전까지 해놓은 것.

hidden_output = tf.nn.relu(tf.matmul(X, w1) + b1) # 활성함수(relu) 사용한 것

# output layer 결과
model = tf.matmul(hidden_output, w2) + b2 # 최종 모델 생성

# sigmoid 알고리즘(1~4)
# 1. model
sigmoid = tf.sigmoid(model)

# 2. cost function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = Y, logits = model))

# 3. 최적화 객체 : 비용함수 최소화
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 4. cut-off : 0.5
cut_off = tf.cast(sigmoid > 0.5, tf.float32)


# 5. model training
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # w, b 초기화
    
    feed_data = {X : x_data, Y : y_data} # 공급 data 
    
    # 반복학습 : 500회 
    for step in range(500) : 
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        
        if (step+1) % 50 == 0 :
            print("step = {}, loss = {}".format(step+1, loss_val))
            
    # model 최적화 
    y_true = sess.run(Y, feed_dict = feed_data)    
    y_pred = sess.run(cut_off, feed_dict = feed_data)    
    
    acc = accuracy_score(y_true, y_pred)
    print("accuracy = ", acc) 

'''
step = 50, loss = 0.6906813383102417
step = 100, loss = 0.588219940662384
step = 150, loss = 0.49513494968414307
step = 200, loss = 0.39326420426368713
step = 250, loss = 0.308067262172699
step = 300, loss = 0.24688027799129486
step = 350, loss = 0.20436085760593414
step = 400, loss = 0.17410346865653992
step = 450, loss = 0.15183371305465698
step = 500, loss = 0.1348360925912857
accuracy =  0.9888094292146735
'''














