# -*- coding: utf-8 -*-
"""
lecture02_keras_export

step01_if_Dataset.py

Dataset 클래스
  - Dataset으로부터 사용가능한 데이터를 메모리에 로딩 기능
  - batch size 지정
"""

import tensorflow as tf

from tensorflow.python.data import Dataset

# member 확인
dir(Dataset)
'''
batch()
from_tensor_slices()
shuffle()
'''

# 1. from_tensro_slices() : 입력 tensor로부터 slice 생성
# ex) MNIST(60000, 28, 28) -> 60000개 image를 각각 1개씩 slice

# 1) x,y변수 생성
x = tf.random.normal([5,2]) # 2차원
y = tf.random.normal([5]) # 1차원

# 2) Dataset : 5개 slice
train_ds = Dataset.from_tensor_slices( (x,y) )
train_ds #<DatasetV1Adapter shapes: ((2,), ()), types: (tf.float32, tf.float32)>

# 5개 관측치 -> 5개 slices
for train_x, train_y in train_ds :
    print("x={}, y={}".format(train_x.numpy(), train_y.numpy()))

'''
x=[-0.6026611  1.6857167], y=-1.8974390029907227
x=[-1.0588142  -0.29367962], y=0.08373463153839111
x=[-1.1722308 -0.974713 ], y=0.9287455081939697
x=[0.5868069 0.8814635], y=0.3892480432987213
x=[-0.871248    0.12599912], y=-1.35860013961792
첫번째가 들어가서 트레인, 그 다음 것이 들어가서 트레인하는 식.
'''

# 2. from_tensor_slices(x, y).shuffle(buffer size).batch(size)
# 일정한 데이터를 섞고, 일정한 batch씩 학습시키는 방법
'''
shuffle(buffer size) : tensor 행단위 셔플링
    - buffer size : 전체 dataset에서 서플링 size
batch : model에 1회 공급할 dataset size
ex) 60,000(mnist) -> shuffle(10000).batch(100)
    1번째 slice data : 10000개 서플링 -> 100개씩 추출 다 소진이 되면 next 10000개 서플링 -> 100개씩 추출
    2번째 slice data : -> 100개씩 추출 
'''

# 1) x,y변수 생성
x2 = tf.random.normal([5,2]) # 2차원
y2 = tf.random.normal([5]) # 1차원

# 2) Dataset : 5개 관측치 -> 3 slice
train_ds2 = Dataset.from_tensor_slices( (x, y) ).shuffle(5).batch(2)
train_ds # <DatasetV1Adapter shapes: ((2,), ()), types: (tf.float32, tf.float32)>

# 3) 3 slice -> 1 slice
for train_x, train_y in train_ds2:
    print("x={}, y={}".format(train_x.numpy(), train_y.numpy()))

'''
x=[[ 0.13558073  0.8378942 ]
 [ 0.9045787  -1.408785  ]], y=[-0.55749136 -1.010345  ]
x=[[-2.768717  -1.006527 ]
 [ 1.3701516 -1.4122744]], y=[-0.18200299  0.9087064 ]
x=[[ 0.13244231 -0.40068492]], y=[-0.64981323]
'''

# 3. keras dataset 적용
from tensorflow.keras.datasets.cifar10 import load_data

# 1. dataset load
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # images : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # (50000, 1)

import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()

y_train[0] #array([6], dtype=uint8)

# train set batch size = 100 image
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(100)
# 50000장 중 10000장을 우선 가져와서 100장씩으로 나누는 것. 그 후에 다시 10000장을 가져와서 반복.

cnt = 0
for image_x, label_x in train_ds :
    print("image = {}, label = {}".format(image_x.shape, label_x.shape))
    cnt += 1
    
print("slice 개수 =", cnt) # slice 개수 = 500
# epochs = iter size(500) * batch size(100)

# val set batch size = 100 image
test_ds = Dataset.from_tensor_slices((x_val, y_val)).shuffle(10000).batch(100)

cnt = 0
for image_x, label_x in test_ds :
    print("image = {}, label = {}".format(image_x.shape, label_x.shape))
    cnt += 1
    
print("slice 개수 =", cnt) # slice 개수 = 100 


'''
문) MNIST 데이터셋을 이용하여 train_ds, val_ds 생성하기
    train_ds : shuffle = 10,000, batch size = 32
    val_ds : batch size = 32
'''

from tensorflow.keras.datasets.mnist import load_data

# 1. dataset load
(x_train, y_train), (x_val, y_val) = load_data()
x_train.shape #(60000, 28, 28)
x_val.shape #(10000, 28, 28)
#label : 10진수
y_train #array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)


# 2. Dataset 생성
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
train_ds # ((None, 28, 28), (None,)), types: (tf.uint8, tf.uint8)>
test_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(32)
test_ds # ((None, 28, 28), (None,)), types: (tf.uint8, tf.uint8)>

# 3. slices 확인
cnt = 0
for image_x, label_x in train_ds :
    print("image = {}, label = {}".format(image_x.shape, label_x.shape))
    cnt += 1
    
print("slice 개수 =", cnt) #slice 개수 = 1875
1875 * 32 #60000

cnt = 0
for image_x, label_x in test_ds :
    print("image = {}, label = {}".format(image_x.shape, label_x.shape))
    cnt += 1
    
print("slice 개수 =", cnt) # slice 개수 = 313
313 * 32 #10016





















