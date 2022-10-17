'''
문) bmi.csv 데이터셋을 이용하여 다음과 같이 logistic regression의 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : 1,2번째 칼럼(height, weight) 
       -> y변수 : 3번째 칼럼(label)의  normal, fat 관측치 대상
   조건2> label 더미변수 변환(normal -> 0, fat -> 1) 
   조건3> 2,000번 학습, 200 step 단위로 Cost 출력 
   조건4> 분류정확도 리포트(Accuracy report) 출력  
'''
import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
 
bmi = pd.read_csv('c:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# label에서 normal, fat 추출 
bmi = bmi[bmi.label.isin(['normal','fat'])]
print(bmi.head())

# 칼럼 추출 
col = list(bmi.columns)
print(col) 

# x,y 변수 추출 
x_data = bmi[col[:2]] # x변수(1,2칼럼)
y_data = bmi[col[2]] # y변수(3칼럼)
print('정규화 전 ')
print(x_data) 

# y변수(label) 로짓 변환 dict
map_data = {'normal': 0,'fat' : 1}
y_data= y_data.map(map_data) # dict mapping
print(y_data) # 0/1

# x_data 정규화 함수 
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

x_data = x_data.apply(normalize)
print('정규화 후 ')
print(x_data)

# numpy 객체 변환 
x_data = np.array(x_data)
y_data = np.transpose(np.array([y_data])) # (1, 15102) -> (15102, 1)

print(x_data.shape) # (15102, 2)
print(y_data.shape) # (15102, 1)

# 공급data
print(x_data)
print(y_data)

# X,y 변수 정의 
X = tf.placeholder(tf.float32, shape=[None, 2]) # [관측치, 입력수]
y = tf.placeholder(tf.float32, shape=[None, 1]) # [관측치, 출력수]  

# w,b 변수 정의 : 초기값(정규분포 난수 )
w = tf.Variable(tf.random_normal([2, 1]))# [입력수,출력수]
b = tf.Variable(tf.random_normal([1])) # [출력수] 

# 이항분류 알고리즘(1~4단계)
# 1. model 
model = tf.sigmoid(tf.matmul(X, w) + b)

# 2. cost function
cost = -tf.reduce_mean(y * tf.log(model) + (1 - y) * tf.log(1 - model))

# 3. GradientDesent algorithm : 비용함수 최소화
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# 4. cut-off 적용   
cut_off = tf.cast(model > 0.5, dtype=tf.float32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_data = {X : x_data, y : y_data}
    for step in range(2000): # 2000번 
        _, cost_val = sess.run([train, cost], feed_dict=feed_data)
        if (step+1) % 200 == 0:
            print('step = %d, cost value = %.6f'%(step, cost_val))

    # Accuracy report
    y_pred = sess.run(cut_off, feed_dict= feed_data)
       
    acc = accuracy_score(y_data, y_pred)
    print('accuracy=', acc)
    #accuracy= 0.9817242749304728
    
    
    
    
    
    
    
    