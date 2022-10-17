'''
exam_mnist_cnn_layer
 ppt-p.31 내용으로 CNN model를 설계하시오.
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함

from tensorflow.keras.datasets.mnist import load_data # ver2.0 dataset
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np

# minst data read
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 
print(x_test.shape) # (10000, 28, 28)
print(y_test.shape) # (10000,) : 10진수 

# image data reshape : [s, h, w, c]
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)

# x_data : int -> float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train[0]) # 0 ~ 255

# x_data : 정규화 
x_train /= 255 # x_train = x_train / 255
x_test /= 255
print(x_train[0])

# y_data : 10 -> 2(one-hot)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# hyper parameters
learning_rate = 0.001
epochs = 15
batch_size = 100
iter_szie = int(60000 / batch_size) # 600

# X, Y 변수 정의 
X_img = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) # input image
Y = tf.placeholder(tf.float32, shape=[None, 10]) # (?, 10)


# 1. Conv layer1(Conv->relu->pool)
Fiter1 = tf.Variable(tf.random_normal([5,5,1,32])) # [h,w,c,map]
conv2d = tf.nn.conv2d(X_img, Fiter1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(conv2d) #활성함수 
L1_out = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L1_out)
# Tensor("MaxPool_21:0", shape=(?, 14, 14, 32), dtype=float32)

# 2. Conv layer2(Conv->relu->pool)
Fiter2 = tf.Variable(tf.random_normal([5,5,32,64])) # [h,w,수일치,map]
conv2d = tf.nn.conv2d(L1_out, Fiter2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(conv2d) #활성함수 
L2_out = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2_out)
# Tensor("MaxPool_23:0", shape=(?, 7, 7, 64), dtype=float32)

# 3. Flatten layer : 합성곱(4차원) -> 행렬곱(2차원)
n = 7 * 7 * 64
L2_flat = tf.reshape(L2_out, [-1, n]) # 3D -> 1D 
print(L2_flat)
# Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)

# 4. Output layer 
W = tf.Variable(tf.random_normal([n, 10]))
b = tf.Variable(tf.random_normal([10]))

# 1) model
model = tf.matmul(L2_flat, W) + b

# softmax(예측치)
softmax = tf.nn.softmax(model) # 0~1 확률값(전체 합=1)

# 2) cost function : softmax + entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=Y, logits=model))
# 3) optimizer
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 4) pred vs label
pred = tf.math.argmax(softmax, 1)
label = tf.math.argmax(Y, 1)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())       
    
    # 15 epochs
    for epoch in range(epochs) :
        tot_cost = 0       
        
        # 1 epochs
        for step in range(iter_szie) :
            idx = np.random.choice(a=60000, size=batch_size, replace=False)
            feed_data = {X_img : x_train[idx], Y : y_train[idx]}
 
            _,cost_val = sess.run([train, cost], feed_dict = feed_data)
            tot_cost += cost_val
        
        avg_cost = tot_cost / iter_szie 
        print('epoch=', epoch+1, 'cost =', avg_cost)
            
    print('learning Finished...')
    feed_data = {X_img : x_test, Y : y_test}    
    y_pred, y_true = sess.run([pred, label], feed_dict = feed_data)
    acc = accuracy_score(y_true, y_pred)
    
    print('accuracy =', acc)
    
'''
epoch= 1 cost = 116.80276211236914
epoch= 2 cost = 8.936960047963039
epoch= 3 cost = 4.837352913850321
epoch= 4 cost = 3.0458111735239597
epoch= 5 cost = 2.027737020205254
epoch= 6 cost = 1.5025953391609654
epoch= 7 cost = 1.3923245415022771
epoch= 8 cost = 1.0934512373790006
epoch= 9 cost = 0.8984113232773827
epoch= 10 cost = 0.7390167595412699
learning Finished...
accuracy = 0.9776
'''    














