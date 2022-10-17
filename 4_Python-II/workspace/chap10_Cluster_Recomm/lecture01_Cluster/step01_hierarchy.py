# -*- coding: utf-8 -*-
"""
계층적 군집분석
    - 상향식(Bottom-up)으로 계층적 군집 형성
    - 유클리드안 거리계산식 이용
    - 숫자형 변수만 사용 가능
"""

import pandas as pd #Dataframe 생성
from sklearn.datasets import load_iris # dataset
from scipy.cluster.hierarchy import linkage, dendrogram # tool
import matplotlib.pyplot as plt # 산점도 시각화

# 1. dataset load
iris = load_iris()
iris.feature_names

X = iris.data # 연속형 변수
y = iris.target # 집단 변수 

iris_df = pd.DataFrame(X, columns = iris.feature_names)
sp = pd.Series(y)

# y변수 추가
iris_df['species'] = sp

iris_df.info()
'''
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   species            150 non-null    int32  
 '''

# 2. 계층적 군집분석
clusters = linkage(iris_df, method='complete', metric='euclidean' )
# 단일기준결합방식(single)이 default값이다.
'''
method = 'single' : 단순연결 가장 가까운 것끼리 연결
method = 'complete' : 완전연결
method = 'average' : 평균연결
'''

clusters.shape # (149, 4)

# 3. 덴드로그램 시각화
plt.figure(figsize = (20, 5))
dendrogram(clusters, leaf_rotation = 90, leaf_font_size = 20,) # 뒤에 ,붙일 것
plt.show()


# 4. 클러스터 자르기/평가 : 덴드로그램의 결과를 보고 판단함.
from scipy.cluster.hierarchy import fcluster # cluster 자르기

# 1) 클러스터 자르기
cluster = fcluster(clusters, t=3, criterion = 'distance')
# 전체 데이터를 3개의 군집으로 쪼개서 보겠다.
cluster # 1 ~ 3

# 2) DF 칼럼 추가
iris_df['cluster'] = cluster
iris_df.info()
'''
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   species            150 non-null    int32  
 5   cluster            150 non-null    int32  
'''

iris_df.head()
iris_df.tail()
# 군집의 결과도 알 수 있다.

# 3) 산점도 시각화
plt.scatter(x=iris_df['sepal length (cm)'],
            y=iris_df['petal length (cm)'],
            c=iris_df['cluster'],
            marker = 'o')
plt.show()

# 4) 관측치 vs 예측치
# pd.crosstab(행 변수, 컬럼 변수)
tab = pd.crosstab(index=iris_df['species'],
                  columns=iris_df['cluster'])
tab
'''
cluster   1   2   3
species            
0        50   0   0
1         0   0  50
2         0  34  16
'''

# 5) 군집별 특성분석
# DF.groupby('집단변수')
cluster_grp = iris_df.groupby('cluster')
cluster_grp.size()
cluster_grp.mean()
'''
         sepal length (cm)  sepal width (cm)  ...  petal width (cm)   species
cluster                                       ...                            
1                 5.006000          3.428000  ...          0.246000  0.000000
2                 6.888235          3.100000  ...          2.123529  2.000000
3                 5.939394          2.754545  ...          1.445455  1.242424
'''





















