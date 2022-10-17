# -*- coding: utf-8 -*-
"""
문) 다음과 같이 Celeb image의 분류기(classifier)를 생성하시오.  
   조건1> train image : train_celeb4
   조건2> validation image : val_celeb4
   조건3> image shape : 120 x 120
   조건4> Image Data Generator 이용 image 자료 생성 
   조건5> CNN model layer 
         1. Convolution layer1 : [4,4,3,32]
         2. Convolution layer2 : [4,4,32,64]
         3. Flatten layer
         4. DNN hidden layer1 : 64
         5. DNN hidden layer2 : 32
         6. DNN output layer : 10
   조건6> 기타 나머지는 step04 참고       
"""
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Conv2D, MaxPool2D,Activation
from tensorflow.keras.layers import Dense, Flatten, Dropout 
import os

# images dir 
base_dir = "./"
train_dir = os.path.join(base_dir, 'train_celeb4')
val_dir = os.path.join(base_dir, 'val_celeb4')


input_shape = (120, 120, 3) 

# 1. CNN Model layer 
print('model create')
model = Sequential()

# Convolution layer1 합성곱으로 특징 추출
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape = input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
Dropout(0.5) # Accuracy에 대해 오버피팅을 해결하는 방안으로 활용 // 별도로 정해진 것은 아니다.
# Val의 Accuracy와 Train의 Accuracy의 차이를 보고 맞춰나가야 한다.


# Convolution layer2 
model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
Dropout(0.5)

# Flatten layer : 3d -> 1d
model.add(Flatten()) 

# DNN hidden layer1(Fully connected layer)
model.add(Dense(64, activation = 'relu'))
Dropout(0.2)

# DNN hidden layer2(Fully connected layer)
model.add(Dense(32, activation = 'relu'))
Dropout(0.2)

# DNN Output layer
model.add(Dense(5, activation = 'softmax'))

# model training set : Adam or RMSprop 
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy', # Y=integer 
              metrics = ['sparse_categorical_accuracy'])

# 2. image file preprocessing : image 제너레이터 이용  // 이미지 생성 시작
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("image preprocessing")

# 특정 폴더의 이미지를 분류하기 위해서 학습시킬 데이터셋 생성
train_data = ImageDataGenerator(rescale=1./255) # 0~1 정규화 

# 검증 데이터도 정규화
validation_data = ImageDataGenerator(rescale=1./255) # 0~1 정규화 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(120,120), # image reshape
        batch_size=20, # batch size
        class_mode='binary') # binary label
# Found 609 images belonging to 4 classes.

validation_generator = validation_data.flow_from_directory(
        val_dir,
        target_size=(120,120),
        batch_size=20,
        class_mode='binary')
# Found 200 images belonging to 4 classes.

# 3. model training : image제너레이터 이용 모델 훈련 
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=100, # 20 *100
          epochs=5, 
          validation_data=validation_generator,
          validation_steps=50) 


# 4. model history graph
import matplotlib.pyplot as plt
 
print(model_fit.history.keys())
# dict_keys(['loss', 'sparse_categorical_accuracy', 'val_loss', 'val_sparse_categorical_accuracy'])

loss = model_fit.history['loss'] # train
acc = model_fit.history['sparse_categorical_accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_sparse_categorical_accuracy']

epochs = range(1, len(acc) + 1)

# acc vs val_acc   
plt.plot(epochs, acc, 'bo', label='train acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('Training vs validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='best')
plt.show()

# loss vs val_loss 
plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training vs validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()