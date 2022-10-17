# -*- coding: utf-8 -*-
"""
피벗테이블(pivot table)
 - 사용자가 행, 열 그리고 교차셀에 변수를 지정하여 테이블 생성
"""

import pandas as pd

pivot_data = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\pivot_data.csv")
pivot_data.info()
'''
교차셀 : 매출액(price) -> 숫자변수
행 : 년도(year), 분기(quarter) -> 집단변수
열 : 매출규모(size) -> 집단변수
셀에 적용할 통계 :sum
'''

ptable = pd.pivot_table(pivot_data, values = 'price',
                        index = ['year', 'quarter'],
                        columns = 'size',
                        aggfunc = 'sum')

ptable
'''
size          LARGE  SMALL
year quarter              
2016 1Q        2000   1000
     2Q        2500   1200
2017 3Q        2200   1300
     4Q        2800   2300
'''

ptable.shape #(4, 2)

ptable.plot(kind='barh', title='2016 vs 2017')
ptable.plot(kind='barh', title='2016 vs 2017', stacked = True)


######### movie_rating.csv 이용 피벗테이블 생성하기 #########
# 행 : 평가자(critic)
# 열 : 영화제목(title)
# 셀 : 평점(rating)
# 적용함수 : sum


movie = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\movie_rating.csv")
movie.head()

movie_table = pd.pivot_table(movie,
                             index = 'critic',
                             columns = 'title',
                             values = 'rating',
                             aggfunc = 'sum')
# sum대신 size를 넣었다면, 영화를 얼마나 봤는지 영화의 빈도를 나타낼 수 있다.

movie_table
'''
title    Just My  Lady  Snakes  Superman  The Night  You Me
critic                                                     
Claudia      3.0   NaN     3.5       4.0        4.5     2.5
Gene         1.5   3.0     3.5       5.0        3.0     3.5
Jack         NaN   3.0     4.0       5.0        3.0     3.5
Lisa         3.0   2.5     3.5       3.5        3.0     2.5
Mick         2.0   3.0     4.0       3.0        3.0     2.0
Toby         NaN   NaN     4.5       4.0        NaN     1.0
'''

# 평가자 기준 평점의 평균
movie_table.mean(axis = 1) # 행 단위 평균
# 영화 기준 평점의 평균
movie_table.mean(axis = 0) # 열 단위 평균
'''
title
Just My      2.375000
Lady         2.875000
Snakes       3.833333
Superman     4.083333
The Night    3.300000
You Me       2.500000
Superman이 가장 평균평점이 높다고 할 수 있다.
'''













