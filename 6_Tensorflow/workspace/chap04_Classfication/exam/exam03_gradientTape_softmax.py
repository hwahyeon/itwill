# -*- coding: utf-8 -*-

'''
문) load_wine() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성하기 
  조건1> train/test - 7:3비율 
  조건2> y 변수 : wine.target
  조건3> x 변수 : wine.data
  조건4> learning_rate = 0.1 ~ 0.01
  조건5> optimizer = tf.keras.optimizers.Adam
  조건6> epoch(step) = 300회
  조건7> 모델 평가 : Accuracy
'''

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import minmax_scale # 정규화 
import tensorflow as tf 

# 1. data load
wine = load_wine()
print(wine) # "data", "target"

# 변수 선택  
X = wine.data  
X.mean() # 69.13366292091617

# X변수 정규화 
x_data = minmax_scale(X)

# y변수 one-hot
y = wine.target
X.shape # (178, 13)

obj = OneHotEncoder() # loss function -> cross entropy
y_data = obj.fit_transform(y.reshape([-1, 1])).toarray()
y_data.shape # (178, 3)

# train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=123)

x_train.shape # (124, 13)
y_train.shape # (124, 3)

# 2. Model 클래스 : model = input * w + b
class Model(tf.keras.Model): # keras Model class 상속 
  def __init__(self): # 생성자 
    super().__init__() 
    self.W = tf.Variable(tf.random.normal([13, 3])) 
    self.B = tf.Variable(tf.random.normal([3])) 
  def call(self, inputs): # 메서드 재정의 
    return tf.matmul(tf.cast(inputs, tf.float32), self.W) + self.B # model 생성 : 예측치  


# 3. 손실함수 : (예측치, 정답) -> 오차 
def loss(model, inputs, outputs): # (model input, output)    
    softmax = tf.nn.softmax(model(inputs)) # 확률값 
    return -tf.reduce_mean(outputs * tf.math.log(softmax) + (1-outputs) * tf.math.log(1-softmax))


# 4. 기울기 계산 함수 : 오차값 -> 기울기 반환  
def gradient(model, inputs, outputs):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, outputs) # 손실함수 호출 
  return tape.gradient(loss_value, [model.W, model.B])
  

# 5. 모델 및 최적화 객체   
model = Model() # 모델 객체 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)#최적화 

print("초기 손실값 : {:.6f}".format(loss(model, x_train, y_train)))
print('-'*30) 


# 6. 반복 학습 : Model 객체와 손실함수 이용
for i in range(300):
  grads = gradient(model, x_train, y_train) # 기울기 계산
  # 기울기에 따른 모델 최적화 : W, B 업데이트 : (gradient, variable)
  optimizer.apply_gradients(zip(grads, [model.W, model.B])) 
  # 기울기 -> 최적화
  
  # Step별 손실값 
  if (i+1) % 20 == 0:
    print("Step = {:03d} -> loss = {:.3f}".format(i+1, 
          loss(model, x_train, y_train) ))
    

# 7. 최적화된 model 
print('-'*30)
print("최종 손실값 : {:.6f}".format(loss(model, x_train, y_train)))

y_true = tf.argmax(y_test, 1) # 정답 
softmax = tf.nn.softmax(model.call(x_test)) # 활성함수 
y_pred = tf.argmax(softmax, 1) # 예측치     
      
print('분류정확도 : {} '.format(accuracy_score(y_true, y_pred)))  
 
'''
------------------------------
최종 손실값 : 0.021365
분류정확도 : 0.9629629629629629
'''