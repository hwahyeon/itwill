# -*- coding: utf-8 -*-

'''
문) load_boston() 함수를 이용하여 보스턴 시 주택 가격 예측 회귀모델 생성하기 
  조건1> train/test - 7:3비율 
  조건2> y 변수 : boston.target
  조건3> x 변수 : boston.data
  조건4> learning_rate=0.005
  조건5> optimizer = tf.keras.optimizers.Adam
  조건6> epoch(step) = 5000회
  조건7> 모델 평가 : MSE
'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale # 정규화
from sklearn.metrics import mean_squared_error
import tensorflow as tf 

# 1. data load
boston = load_boston()
print(boston) # "data", "target"

# 2. 변수 선택  
X = boston.data # 정규화 
y = boston.target
X.shape # (506, 13)
y_nor = minmax_scale(y)

# train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        X, y_nor, test_size=0.1, random_state=123)


tf.random.set_seed(123)

# 2. Model 클래스 : model = input * w + b
class Model(tf.keras.Model): # keras Model class 상속 
  def __init__(self): # 생성자 
    super(Model, self).__init__()
    self.w = tf.Variable(tf.random.uniform(shape=[13, 1]))
    #type(self.w) # dtype=float32
    self.b = tf.Variable(tf.zeros(shape=[1]))
  def call(self, inputs): # 메서드 재정의 -> model(inputs) 자동호출 메서드 
      # 행렬곱 사용 : type 일치 
      return tf.matmul(tf.cast(inputs,tf.float32), self.w)+self.b# 예측치    


# 3. 손실함수 : (예측치, 정답) -> 오차 
def loss(model, inputs, outputs): # (model object, input, output)
    error = model(inputs) - outputs # model(inputs) = call(inputs)
    return tf.reduce_mean(tf.square(error)) # mse


# 4. 기울기 계산 함수 : 오차값 -> 기울기 반환 
def gradient(model, inputs, outputs):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, outputs) # 손실함수 호출 
  return tape.gradient(loss_value, [model.w, model.b])
# 기울기(미분) 계산에 따른 W, B 업데이트 추적
  

# 5. 모델 및 최적화 객체   
model = Model() # 모델 객체 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)# 최적화


#print("초기 손실값 : {}".format(loss(model, x_train, y_train)))
print(loss(model, x_train, y_train))
print("w = {}, b = {}".format(model.w.numpy(), model.b.numpy()))
print('-'*30)


# 6. 반복 학습 : Model 객체와 손실함수 이용
for i in range(5000): # 5000번 학습 
  grads = gradient(model, x_train, y_train) # 기울기 계산
  # 기울기 -> 모델 최적화 : W, B 업데이트 : (gradient, variable)
  optimizer.apply_gradients(zip(grads, [model.w, model.b])) 
  
  # Step별 손실값 
  if (i+1) % 50 == 0:
    print("Step : {:03d} -> loss : {:.3f}".format(i+1, 
          loss(model, x_train, y_train) ))
    

# 7. 최적화된 model 
print('-'*30)
print("최종 손실값: {}".format(loss(model, x_train, y_train)))
print("w = {}, b = {}".format(model.w.numpy(), model.b.numpy()))

# model 예측치   
y_pred = model.call(x_test)
print('최적화된 model 예측치 : \n', y_pred.numpy())  

mse = mean_squared_error(y_test, y_pred)
print("MSE =", mse)
