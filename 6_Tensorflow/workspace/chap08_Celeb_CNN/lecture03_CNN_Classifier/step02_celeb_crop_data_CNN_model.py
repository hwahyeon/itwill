# -*- coding: utf-8 -*-
"""
step02_celeb_crop_data_CNN_model.py

1. file load
2. CNN layer
3. CNN model
4. CNN model save
"""
import tensorflow as tf # ver2.x
from tensorflow.keras.datasets.cifar10 import load_data # dataset
from tensorflow.keras import Sequential # model
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras.layers import Conv2D, MaxPooling2D # Conv layer
from tensorflow.keras.layers import Dense, Flatten, Dropout # DNN layer
import numpy as np


# 1. file load

# lecture03_CNN_Classifier
x_train, y_train, x_val, y_val = np.load(file="./create_file/image_train_val.npy", 
        allow_pickle=True)

x_train.shape # images : (740, 150, 150, 3) - (size, h, w, c)
y_train.shape # labels : (740, 5)

x_val.shape # (250, 150, 150, 3)


# 2. keras CNN model layer
model = Sequential()

input_shape = (150, 150, 3)

# conv layer1 : 1층 [5,5,3,32] : kernel_size -> Filter
model.add(Conv2D(32, kernel_size=(5,5), input_shape = input_shape,
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# conv layer2 : 2층 [5,5,32,64]
model.add(Conv2D(64, kernel_size=(5,5), activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Flatten : 3d -> 1d
model.add(Flatten())

# DNN hidden layer : 3층 
model.add(Dense(256, activation="relu"))

# DNN output layer : 4층 
model.add(Dense(5, activation="softmax"))


# 3. model compile : 학습환경 설정 
model.compile(optimizer = 'adam',  # 최적화 알고리즘(lr 생략) 
              loss='categorical_crossentropy', # 손실 - one-hot
              metrics=['accuracy']) # 평가 방법 

# layer 확인 
model.summary() 


# 4. model training
model_fit = model.fit(x=x_train, y=y_train, # 학습용       
          #batch_size = 100 이미지 분량이 많지 않아 batchsize는 쓰지 않음.
          epochs=15, # image 재학습 size
          verbose=1,
          validation_data=(x_val, y_val)) # 평가용 

# 5. model evaluation : 모델 검증  
loss, acc = model.evaluate(x=x_val, y=y_val) # accuracy: 0.9763
print("loss ={:.7f}, accuracy = {:.7f}".format(loss, acc))
# loss =0.9008456, accuracy = 0.6857000


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

# 7. model save
model.save("./create_file/celeb_CNN_model.h5")



























