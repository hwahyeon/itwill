# -*- coding: utf-8 -*-
"""
 - DataFrame의 요약통계량
 - 상관관계
"""

import pandas as pd

product = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\product.csv") #절대경로
product.info()
product.head()
product.tail()

# 요약통계량
summ = product.describe() #product가 pandas 객체니까 이렇게 사용 가능.
# describe는 R에서의 summary와 비슷한 역할을 해줌.

print(summ)

# 행/열 통계
product.sum(axis=0) #행축 : 열단위 합계
'''
a    773
b    827
c    817
dtype: int64
'''
product.sum(axis=1) #열축 : 행단위 합계

# 산포도 : 분산, 표준편차
product.var() #분산 // axis=0가 디폴트값
product.std() #표준편차 // axis=0가 디폴트값

# 빈도수 : 집단변수
a_cnt = product['a'].value_counts() #product.a // a가 가지고 있는 전체 빈도수를 카운트해서 반환함.
print(a_cnt)

b_cnt = product['b'].value_counts()
print(b_cnt)

# 유일값 보기
print(product['c'].unique())
# [3 2 4 5 1]

# 상관관계
product.corr() # 상관계수 정방행렬
'''
          a         b         c
a  1.000000  0.499209  0.467145
b  0.499209  1.000000  0.766853
c  0.467145  0.766853  1.000000
'''

# iris dataset 적용
iris = pd.read_csv('iris.csv') #상대경로
iris = pd.read_csv('C:\\ITWILL\\4_Python-II\\data\\iris.csv') #절대경로
iris.info()

# subset 생성
iris_df = iris.iloc[:,:4] 
iris_df.shape # (150, 4)

# 변수 4개 요약통계량
iris_df.describe()

# 상관관계 행렬
iris_df.corr()

# 집단변수
species = iris.Species
species.value_counts()
'''
versicolor    50
virginica     50
setosa        50
Name: Species, dtype: int64
'''

species.unique()
'''
array(['setosa', 'versicolor', 'virginica'], dtype=object)
'''

list(species.unique())
#['setosa', 'versicolor', 'virginica']
























