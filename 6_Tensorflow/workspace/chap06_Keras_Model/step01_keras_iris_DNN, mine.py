# -*- coding: utf-8 -*-
"""
step01_keras_iris_DNN.py

- Tensorflow2.x keras + iris
- Keras : DNN model 생성을 위한 고수준 API
"""

import tensorflow as tf # ver2.x
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale # x변수 전처리
from tensorflow.keras.utils import to_categorical # y변수 전처리
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense


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

# [수정한 부분]
y_data = to_categorical(y_data) # y변수 전처리 : one hot encoding : 이렇게 전처리 해놔야 나중에 엔트로피를 쓸 수 있다.
y_data.shape # (150, 3)

# 75 vs 25
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data) 
print(x_train.shape) # (112, 3)
print(x_val.shape) # (38, 3)


# 2. keras model 생성
model = Sequential()
model # object info

# 3. model layer
'''
model.add(Dense(node수, input_shape, activation)) # hidden layer1
model.add(Dense(node수, activation)) #hidden layer1 ~ n
'''
# hidden layer1 = [4, 12]
model.add(Dense(12, input_shape=(4,),activation='relu')) # 1층
# hidden layer2 = [12, 6]
model.add(Dense(6, activation='relu')) # 2층
# hidden layer3 = [6, 3]
model.add(Dense(3, activation='softmax')) # 3층

# 4. model compile : 학습환경 설정
model.compile(optimizer = 'adam', # 최적화 알고리즘(버젼1에 있던 lr 생략, 여기선 자동 조절)
              loss = 'categorical_crossentropy', # 손실
              metrics=['accuracy']) # 평가 방법

# Layer 확인
model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (None, 12)                60 = w(4*12) + b(12)        
_________________________________________________________________
dense_7 (Dense)              (None, 6)                 78 = w(12*6)+b(6)       
_________________________________________________________________
dense_8 (Dense)              (None, 3)                 21 = w(6*3)+b(3)       
=================================================================
Total params: 159
Trainable params: 159
Non-trainable params: 0
'''

# 5. model training : train(112) vs val(38)
model.fit(x=x_train, y=y_train, # 학습용
          epochs = 300,
          verbose = 1,
          validation_data=(x_val, y_val)) # 평가용
















