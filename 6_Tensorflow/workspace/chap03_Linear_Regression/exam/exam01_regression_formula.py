'''
문1) 다음과 같이 다중선형회귀방정식으로 모델의 예측치와 평균제곱오차를 계산하시오.
    조건1> X변수 : 4개, Y변수 : 1개  
    조건1> X변수 공급 데이터 : x_data = [[1.2,2.2,3.5,4.2]] - (1,4)
    조건2> Y변수[정답] : Variable()이용 표준정규분포 난수 상수 1개  
    조건3> a변수[기울기] : Variable()이용 표준정규분포 난수 상수 4개
    조건4> b변수[절편] : Variable()이용 표준정규분포 난수 상수 1개     
    조건5> model 예측치 : pred_Y = (X * a) + b 
        -> 행렬곱 함수 적용  
    조건6> model 손실함수 출력 
        -> 손실함수는 python 함수로 정의 : 함수명 -> loss_fn(err)

<< 출력 예시 >>    
a[기울기] = 
 [[-2.063591 ]
 [ 0.10648511]
 [ 0.49105361]
 [ 0.00555888]]
b[절편] = [[ 1.4659301]]
Y[정답] = [[-0.39188424]]
pred_Y[예측치] = [[ 0.96592301]]
loss function = = 1.84364 
'''

import tensorflow as tf  

# 1. 변수 정의 
X = [[1.2,2.2,3.5,4.2]] # 공급 data [1, 4]
Y = tf.Variable(tf.random.normal([1])) # 출력 
a = tf.Variable(tf.random.normal([4, 1])) # 기울기(4,1) 
b = tf.Variable(tf.random.normal([1])) # 절편

# 2. model 예측치/오차/손실함수 정의 
y_pred = tf.matmul(X, a) + b

err = Y - y_pred

def loss_fn(error) :
    loss = tf.reduce_mean(tf.square(error))
    return loss

# 3. 결과 출력 
print("기울기 :")
print(a.numpy())
   
print("절편 :", b.numpy())

print("정답 : ", Y.numpy())

print("y_pred :", y_pred.numpy())

print("loss function =", loss_fn(err).numpy())

'''
기울기 :
[[-0.19896524] - x1
 [-0.634054  ] - x2
 [-0.80100024] - x3
 [ 1.281508  ]]- x4
절편 : [-0.34021875]
정답 :  [0.5928572]
y_pred : [[0.60493636]]
loss function = 0.00014590657
'''











