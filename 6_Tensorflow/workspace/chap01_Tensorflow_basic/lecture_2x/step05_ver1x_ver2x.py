# -*- coding: utf-8 -*-
"""
step05_ver1x_ver2x.py

ver1.x -> ver2.x

'''
step08_variable_feed_csv.py -> ver2.0
    1. 즉시 실행 모드
    2. 세션 대신 함수
    3. @tf.funtion 함수 장식자
'''

"""
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

iris = pd.read_csv("C:/ITWILL/6_Tensorflow/data/iris.csv")
iris.info()


# 1. 공급 data 생성 : DataFrame
cols = list(iris.columns)
x_data = iris[cols[:4]]
y_data = iris[cols[-1]]

x_data.shape  # (150, 4)
y_data.shape  # (150,)

x_data = np.array(x_data)

# 2. X,Y 변수 정의
X = tf.constant(x_data, dtype=tf.float32)
Y = tf.constant(y_data, dtype=tf.string)


# 3. 평균, 빈도수 함수 정의
def iris_func(x, y):
    
    # 평균
    x_df = pd.DataFrame(x, columns= ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
    x_mean = x_df.mean(axis=0)
    
    # 빈도수
    y_ser = pd.Series(y)
    y_val_count = y.value_counts()
    
    return x_mean, y_val_count


iris_mean, iris_count = iris_func(x_data, y_data)
print(iris_mean)
print(iris_count)







