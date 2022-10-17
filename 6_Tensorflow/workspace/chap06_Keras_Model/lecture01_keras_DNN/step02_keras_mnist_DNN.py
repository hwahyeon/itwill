# -*- coding: utf-8 -*-
"""
step02_keras_mnist_DNN.py

Tensorflow2.0 Keras + MNIST(0~9) + Flatten layer

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
model.fit(x=x_train, y=y_train, # 학습용 
          epochs=10,
          verbose=1,
          validation_data=(x_val, y_val)) # 평가용 


# 6. model evaluation : 모델 검증  
model.evaluate(x=x_val, y=y_val) # accuracy: 0.9763


# 7. model save/load 
model.save("keras_model_mnist.h5")
print("model saved")

new_model = load_model("keras_model_mnist.h5")
print(new_model) # object info 


# 8. model test : new dataset
y_pred = new_model.predict(x_val) #예측치 : 확률값 
y_true = y_val # 관측치 # one hot 

# 확률값 or one hot -> decoding
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.argmax(y_true, axis=1)

acc = accuracy_score(y_true, y_pred)
print("accuracy = ", acc)
# accuracy =  0.9763



