# -*- coding: utf-8 -*-
"""
문2) 아래와 같은 단계로 kMeans 알고리즘을 적용하여 확인적 군집분석을 수행하시오.

 <조건> 변수 설명 : tot_price : 총구매액, buy_count : 구매횟수, 
                   visit_count : 매장방문횟수, avg_price : 평균구매액

  단계1 : 3개 군집으로 군집화
 
  단계2: 원형데이터에 군집 예측치 추가
  
  단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)
  
  단계4 : 산점도에 군집의 중심점 시각화
"""

import pandas as pd
from sklearn.cluster import KMeans # kMeans model
import matplotlib.pyplot as plt

sales = pd.read_csv("../data/product_sales.csv")
print(sales.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
tot_price      150 non-null float64 -> 총구매금액 
visit_count    150 non-null float64 -> 매장방문수 
buy_count      150 non-null float64 -> 구매횟수 
avg_price      150 non-null float64 -> 평균구매금액 
'''

model = KMeans(n_clusters=3, random_state=0, algorithm='auto') # k=3, auto:default
model.fit(sales)

# kMeans model 에측치 
pred = model.predict(sales)
print(pred)


# 예측치 추가 
sales['predict'] = pred # column 추가 = numpy vector 추가 가능 
print(sales)

# 상관계수
print(sales.corr()) # tot_price vs avg_price

# tot_price vs avg_price 산점도  
plt.scatter(sales['tot_price'], sales['avg_price'], c=sales.iloc[:,4])


# 군집 중앙값  
centers = model.cluster_centers_
print(centers)
'''
[[6.83902439 5.67804878]
 [5.00784314 1.49215686]
 [5.87413793 4.39310345]]
'''

# 중앙값 시각화 
plt.scatter(centers[:,0], centers[:, 3], marker='D', c='r')
plt.show()







