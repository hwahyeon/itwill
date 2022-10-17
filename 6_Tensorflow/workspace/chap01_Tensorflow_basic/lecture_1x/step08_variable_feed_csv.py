# -*- coding: utf-8 -*-
"""
step08_variable_feed_csv.py
    - csv(pandas object) -> tensorflow variable
"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함

import pandas as pd # csv file read
from sklearn.model_selection import train_test_split # data split

iris = pd.read_csv("C:/ITWILL/6_Tensorflow/data/iris.csv")
iris.info()


# 1. 공급 data 생성 : DataFrame
cols = list(iris.columns)
x_data = iris[cols[:4]]
y_data = iris[cols[-1]]

x_data.shape # (150, 4)
y_data.shape # (150,)


# 2. X, Y 변수 정의 : tensorflow // 공급 데이터를 저장할 변수를 만듦
X = tf.placeholder(dtype=tf.float32, shape = [None,4]) # [?, 4]
Y = tf.placeholder(dtype=tf.string, shape = [None]) # [?]


# 3. train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size = 0.3)

x_train.shape # (105, 4)
x_test.shape # (45, 4) // (150, 4) 중 30%를 추린거니까
y_train.shape # (105,)
y_test.shape # (45,)


# 4. session object : data 공급 -> 변수
with tf.Session() as sess:
    # 훈련용 data 공급
    train_feed_data = {X : x_train, Y : y_train} # 공급 데이터
    X_val, Y_val = sess.run([X, Y], feed_dict = train_feed_data)
    print(X_val)
    print(Y_val)
    
    # 평가용 data 공급
    test_feed_data = {X : x_test, Y : y_test}
    X_val2, Y_val2 = sess.run([X, Y], feed_dict = test_feed_data)
    print(X_val2)
    print(Y_val2)
    print(Y_val2.shape) #(45,)
    print(type(Y_val2)) #<class 'numpy.ndarray'>
    
    # numpy -> pandas 변경
    X_df = pd.DataFrame(X_val2, columns = ['a', 'b', 'c', 'd'])    
    print(X_df.info())
    print(X_df.mean(axis = 0))

    Y_ser = pd.Series(Y_val2)
    print(Y_ser.value_counts())



















