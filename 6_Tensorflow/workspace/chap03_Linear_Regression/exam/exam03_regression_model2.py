'''
문3) iris.csv 데이터 파일을 이용하여 선형회귀모델  생성하시오.
     [조건1] x변수 : 2,3칼럼,  y변수 : 1칼럼
     [조건2] 7:3 비율(train/test set)
         train set : 모델 생성, test set : 모델 평가  
     [조건3] learning_rate=0.01
     [조건4] 학습 횟수 1,000회
     [조건5] model 평가 - MSE출력 
'''

import pandas as pd
import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 
from sklearn.metrics import mean_squared_error # model 평가 
from sklearn.preprocessing import minmax_scale # 정규화 
from sklearn.model_selection import train_test_split # train/test set

iris = pd.read_csv('C:/ITWILL/6_Tensorflow/data/iris.csv')
print(iris.info())
cols = list(iris.columns)
iris_df = iris[cols[:3]] 

# 1. x data, y data
x_data = iris_df[cols[1:3]] # x train
y_data = iris_df[cols[0]] # y tran

# 2. x,y 정규화(0~1) 
x_data = minmax_scale(x_data)


# 3. train/test data set 구성 
x_train, x_test, y_train, y_test = train_test_split(
     x_data, y_data, test_size=0.3, random_state=0) # seed값 

print(x_train.shape) # (105, 2) -> y(1),x(2)
print(x_test.shape) # (45, 2) -> y(1),x(2)s


# 4. X,y변수 정의 
X = tf.placeholder(dtype=tf.float32, shape=[None, 2]) # 입력 2개 
y = tf.placeholder(dtype=tf.float32, shape=[None]) # 출력 1개 

# a,b변수 정의 : 초기값 = 0
a = tf.Variable(tf.zeros([2, 1])) # 2차원(2,1) 
b = tf.Variable(tf.zeros([1])) 

# 5. model 생성 
y_pred = tf.matmul(X, a) + b # a=weight, b=bias
# model = tf.matmul(X, w) + b

# cost function
cost = tf.reduce_mean(tf.square(y - y_pred))

# optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = opt.minimize(cost)

# 6. model training
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # a,b 초기화     
    
    # x변수 정규화 공급 
    feed_data = {X : x_train, y : y_train}
    
    # 1,000번 학습 
    for step in range(1000) :        
        # model 생성 : train set 이용
        _, cost_val = sess.run([train, cost], feed_dict = feed_data)
        
        print('step=', step+1, 'cost=', cost_val)
        
    
    # 최적화 model 평가 : test set 이용 
    feed_data2 = {X : x_test, y : y_test}
    
    y_pred_re = sess.run(y_pred, feed_dict = feed_data2)
    y_true = sess.run(y, feed_dict = feed_data2)
    
    # model 평가 
    mse = mean_squared_error(y_true, y_pred_re)
    print('MSE = ', mse)
        
# MSE =  0.41485003