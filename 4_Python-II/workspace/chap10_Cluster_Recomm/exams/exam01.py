'''
 문) 중학교 1학년 신체검사(bodycheck.csv) 데이터 셋을 이용하여 다음과 같이 군집모델을 생성하시오.
  <조건1> 악력, 신장, 체중, 안경유무 칼럼 대상 [번호 칼럼 제외]
  <조건2> 계층적 군집분석의 완전연결방식 적용 
  <조건3> 덴드로그램 시각화 
  <조건4> 텐드로그램을 보고 3개 군집으로 서브셋 생성  
'''

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram # 계층적 군집 model
import matplotlib.pyplot as plt

# data loading - 중학교 1학년 신체검사 결과 데이터 셋 
body = pd.read_csv("../data/bodycheck.csv", encoding='ms949')
print(body.info())

# <조건1> subset 생성 - 악력, 신장, 체중, 안경유무 칼럼  이용 
body_df = body[['악력', '신장', '체중', '안경유무']]
body_df


# <조건2> 계층적 군집 분석  완전연결방식  
clusters = linkage(body_df, method='complete', metric='euclidean' )
clusters.shape


# <조건3> 덴드로그램 시각화 : 군집수는 사용 결정 
plt.figure(figsize = (20, 10))
dendrogram(clusters,)
plt.show()


# <조건4> 텐드로그램을 보고 3개 군집으로 서브셋 생성
'''
cluster1 - 9 3 7 0 14
cluster2 - 10 2 4 5 13
cluster3 - 1 8 12 6 11
'''

# cluster 구성 
cluster1 = body_df.iloc[[9, 3, 7, 0, 14]]
cluster2 = body_df.iloc[[10, 2, 4, 5, 13]]
cluster3 = body_df.iloc[[1, 8, 12, 6, 11]]

# 칼럼추가 
body_df['cluster'] = 0 # 칼럼추가
body_df.loc[[9, 3, 7, 0, 14],'cluster'] = 1
body_df.loc[[10, 2, 4, 5, 13],'cluster'] = 2
body_df.loc[[1, 8, 12, 6, 11],'cluster'] = 3

body_df

# 각 집단별 특성분석
grp = body_df.groupby('cluster') # 그룹 생성
grp.size()
grp.mean()
'''
           악력     신장    체중  안경유무
cluster                         
1        25.6  149.8  36.6   1.0 -> 소, 안경(1)
2        33.8  161.2  48.8   1.4 -> 중, 안경(1, 2)
3        40.6  158.8  56.8   2.0 -> 대, 안경(2)
'''









