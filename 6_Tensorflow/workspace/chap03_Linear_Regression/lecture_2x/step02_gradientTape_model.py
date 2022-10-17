# -*- coding: utf-8 -*-
"""
step02_gradientTape_model.py

tf.GradientTape + regression model
 -> 미분계수 자동 계산 -> model 최적화(최적의 기울기와 절편 update)
"""
import tensorflow as tf
tf.executing_eagerly() # 즉시 실행 test : ver2.x 사용중이라면 true값이 나와야 한다.

# 1. input/output 변수 정의 
inputs = tf.Variable([1.0, 2.0, 3.0]) # x변수 
outputs = tf.Variable([2.0, 4.0, 6.0]) # y변수 1.25->1.9->2.8
 

# 2. model : Model 클래스
class Model(tf.keras.Model): # 자식클래스(부모클래스)
    def __init__(self): # 생성자
        super().__init__() # 부모 생성자 호출
        self.W = tf.Variable(tf.random_normal([1])) # 기울기(가중치)
        self.B = tf.Variable(tf.random_normal([1])) # 절편
    def call(self, inputs): # 메서드 재정의
        return inputs * self.W + self.B # 회귀방정식(예측치)
    

# 3. 손실 함수 : 오차 반환 
def loss(model, inputs, outputs):
  err = model(inputs) - outputs # 예측치 - 정답
  return tf.reduce_mean(tf.square(err)) # MSE

# 4. 미분계수(기울기) 계산
def gardient(model, inputs, outputs):  
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs) # 손실함수 호출  
        grad = tape.gradient(loss_value, [model.W, model.B]) 
        # 미분 계수 -> 기울기와 절편 업데이트
    return grad # 업데이터 결과 변환

# 5. model 생성
model = Model() # 생성자

'''
mse = loss(model, inputs, outputs)
print("mse =", mse.numpy()) # mse = 0.259863

grad = gardient(model, inputs, outputs)
print("grad =", grad)
'''
# 6. model 최적화 객체
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

print("최초 손실값  : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))

# 7. 반복학습 
for step in range(300) :
    grad = gardient(model, inputs, outputs) #  기울기 계산 
    # 기울기 -> 최적화 객체 반영 
    opt.apply_gradients(zip(grad, [model.W, model.B]))
    
    if (step+1) % 20 == 0 :
        print("step = {}, loss = {:.6f}".format((step+1), 
                                            loss(model, inputs, outputs)))
    
# model 최적화 
print("최종 손실값  : {:.6f}".format(loss(model, inputs, outputs)))
print("w : {}, b : {}".format(model.W.numpy(), model.B.numpy()))
    
# model test 
y_pred = model.call(2.5) # x = 2.5
print("y pred =", y_pred.numpy()) # y pred = [5.0104895]

'''
최초 손실값  : 87.562386
w : [-1.5415783], b : [-1.8163118]


최종 손실값  : 0.001586
w : [2.0462575], b : [-0.10515408]
'''























