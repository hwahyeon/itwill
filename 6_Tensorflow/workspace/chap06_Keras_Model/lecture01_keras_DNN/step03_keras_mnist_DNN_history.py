# -*- coding: utf-8 -*-
"""
step03_keras_mnist_DNN_history.py

Tensorflow2.0 Keras + MNIST(0~9) + Flatten layer + History

1차 : 1차원 : (28x28) -> 784
2차 : 2차원 : 28x28 -> Flatten 적용 
"""

import tensorflow as tf # ver2.x
from tensorflow.keras.datasets.mnist import load_data # ver2.x dataset
from tensorflow.keras.utils import to_categorical # y변수 전처리 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense, Flatten # layer 생성 
from tensorflow.keras.models import load_model # model load
from sklearn.metrics import accuracy_score

# 1. x, y 공급 data 
(x_train, y_train), (x_val, y_val) = load_data()
x_train.shape # images : (60000, 28, 28)
y_train.shape # labels : (60000,)

# x변수 전처리 : 정규화 
x_train[0] # 0 ~ 255
x_train = x_train / 255.
x_val = x_val / 255.

# 2d -> 1d : 1차 
'''
x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)
'''

# y변수 전처리 : one hot encoding
y_train = to_categorical(y_train) 
y_val = to_categorical(y_val)
y_train.shape # (60000, 10)


# 2. keara model 생성 
model = Sequential()
model # object info 

# 3. model layer
'''
model.add(Dense(node수, input_shape, activation)) # hidden layer1
model.add(Dense(node수, activation)) # hidden layer1 ~ n
'''
input_shape = (28, 28) # 2차원 

# Flatten lauer : 2d(28,28) -> 1d(784)
model.add(Flatten(input_shape = input_shape)) # 0층 

# hidden layer1 = [784, 128] 
model.add(Dense(128, activation='relu')) # 1층 
# hidden layer2 = [128, 64]
model.add(Dense(64, activation='relu')) # 2층 
# hidden layer3 = [64, 32]
model.add(Dense(32, activation='relu')) # 3층
# output layer = [32, 10]
model.add(Dense(10, activation='softmax')) # 4층 

# 4. model compile : 학습환경 설정 
model.compile(optimizer = 'adam',  # 최적화 알고리즘(lr 생략) 
              loss='categorical_crossentropy', # 손실 
              metrics=['accuracy']) # 평가 방법 

# layer 확인 
model.summary() 
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_24 (Dense)             (None, 128)               100480    
_________________________________________________________________
dense_25 (Dense)             (None, 64)                8256      
_________________________________________________________________
dense_26 (Dense)             (None, 32)                2080      
_________________________________________________________________
dense_27 (Dense)             (None, 10)                330       
=================================================================
'''

# 5. model training : train(112) vs val(38)
model_fit = model.fit(x=x_train, y=y_train, # 학습용 
           epochs=15, # 10 -> 15 수정 
           verbose=1,
           validation_data=(x_val, y_val)) # 평가용 


# 6. model history
print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

train_loss = model_fit.history['loss']
train_acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss']
val_acc = model_fit.history['val_accuracy']

import matplotlib.pyplot as plt 

# train vs val loss
plt.plot(train_loss, color = 'y', label = 'train loss')
plt.plot(val_loss, color='r', label = 'val loss')
plt.legend(loc='best')
plt.xlabel("epochs")
plt.show()

# train vs val accuracy
plt.plot(train_acc, color = 'y', label = 'train acc')
plt.plot(val_acc, color='r', label = 'val acc')
plt.legend(loc='best')
plt.xlabel("epochs")
plt.show()













