'''
문) 다음과 같이 Convolution layer와 Max Pooling layer를 정의하고, 실행하시오.
  <조건1> input image : volcano.jpg 파일 대상    
  <조건2> Convolution layer 정의 
    -> Filter : 6x6
    -> featuremap : 16개
    -> strides= 1x1, padding='SAME'  
  <조건3> Max Pooling layer 정의 
    -> ksize= 3x3, strides= 2x2, padding='SAME' 
'''
import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# 1. image load
img = imread('C:/ITWILL/6_Tensorflow/workspace/chap07_CNN/images/volcano.jpg') # 이미지 읽어오기
plt.imshow(img)
plt.show()
print(img.shape) #(405, 720, 3)


# 2. image 전처리  : 실수형 -> 정규화 
img = img.astype('float32') 
img = img / 255
img

# 3. image reshape 
inputImg = img.reshape(-1, 405, 720, 3)

Filter = tf.Variable(tf.random_normal([6,6,3,16]))#[r,c,color,fmap]

 

# 1. Convolution layer : 특징 추출 
conv2d = tf.nn.conv2d(inputImg, Filter, strides=[1,1,1,1], padding='SAME')
print(conv2d) # Tensor("Conv2D:0", shape=(1, 405, 720, 16), dtype=float32)

# 2. Pool layer : down sampling
pool = tf.nn.max_pool(conv2d, ksize=[1,3,3,1],strides=[1,2,2,1],
            padding = 'SAME')
print(pool) # Tensor("MaxPool:0", shape=(1, 203, 360, 16), dtype=float32)


with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # filter 초기화 
    
    # 합성곱 연산 
    conv2d_img = sess.run(conv2d)    
    conv2d_img = np.swapaxes(conv2d_img, 0, 3) # 축 교환 : 1, 256, 384, 8
    print(conv2d_img.shape) # (8, 256, 384, 1)
    
    for i, img in enumerate(conv2d_img) :
        plt.subplot(1, 16, i+1) # 1행16열,1~16 
        plt.imshow(img.reshape(405, 720), cmap='gray') # 
    plt.show()
    
    # 폴링 연산 
    pool_img = sess.run(pool)
    pool_img = np.swapaxes(pool_img, 0, 3) # 1, 64, 96, 8 -> 8, 64, 96, 1
    
    for i, img in enumerate(pool_img) :
        plt.subplot(1,16, i+1) # 1행16열,1~16 
        plt.imshow(img.reshape(203, 360), cmap='gray') 
    plt.show()
    

























