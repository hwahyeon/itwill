# -*- coding: utf-8 -*-
"""
kMeans 알고리즘
    - 비계층적(확인적) 군집분석
    - 군집수(k) 알고 있는 경우 이용
"""

import pandas as pd #DataFrame
import numpy as np # array
from sklearn.cluster import KMeans # model
import matplotlib.pyplot as plt # 시각화

# 1. dataset

# text file -> numpy
def dataMat(file):
    dataset = [] # data mat 저장
    
    f = open(file, mode = 'r') # file object
    lines = f.readlines()
    for line in lines: #1.658985	4.285136
        cols = line.split('\t') #'1.658985'	'4.285136'
        
        rows = [] # x, y 저장
        for col in cols: # '1.658985'
            rows.append(float(col)) #[1.658985,	4.285136]
            
        dataset.append(rows) # [[rows], [rows], ... [rows]]
    
    return np.array(dataset) # 중첩리스트를 array로(2차원)으로 만듦

dataset = dataMat('C:/ITWILL/4_Python-II/data/testSet.txt')
dataset.shape #(80, 2)
dataset[:5]

# numpy -> pandas
dataset_df = pd.DataFrame(dataset, columns = ['x', 'y'])

dataset_df.info()


# 2. kMeans model : k=4
model = KMeans(n_clusters = 4, algorithm = 'auto')
model

model = model.fit(dataset_df)
pred = model.predict(dataset_df)
pred # 0 ~ 3



# 각 cluster의 center
centers = model.cluster_centers_
'''
array([[-3.38237045, -2.9473363 ,  3.        ],
       [ 2.6265299 ,  3.10868015,  2.        ],
       [-2.46154315,  2.78737555,  1.        ],
       [ 2.80293085, -2.7315146 ,  0.        ]])
'''


# 3. 시각화
dataset_df['cluster'] = pred
dataset_df.head()

plt.scatter(x = dataset_df['x'], y = dataset_df['y'],
            c=dataset_df['cluster'], marker = 'o')
# 중심점
plt.scatter(x = centers[:,0], y = centers[:,1],
            c='red', marker = 'D') # Diamond
plt.show()

grp = dataset_df.groupby('cluster')
grp.mean()
'''
                x         y
cluster                    
0       -2.461543  2.787376
1        2.802931 -2.731515
2       -3.382370 -2.947336
3        2.626530  3.108680
'''





























































