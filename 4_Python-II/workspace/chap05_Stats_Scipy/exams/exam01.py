# -*- coding: utf-8 -*-
"""
문01) 이항검정 : 토요일(Sat)에 오는 여자 손님 중 비흡연자가 흡연자 보다 많다고 할 수 있는가?

 # 귀무가설 : 비흡연자와 흡연자의 비율은 차이가 없다.(P=0.5)
"""

import pandas as pd

tips = pd.read_csv("../data/tips.csv")
print(tips.info())
print(tips.head())

day = tips['day']
print(day.value_counts())
'''
Sat     87  -> 토요일 빈도수 
Sun     76
Thur    62
Fri     19
'''

gender = tips['sex']
print(gender.value_counts())
'''
Male      157
Female     87 -> 여자 빈도수
'''

# 행사 요일이 토요일 이면서 성별이 여성인 경우 subset 생성

# 1) 행사 요일 = 토요일
tips_day = tips[tips['day'] == 'Sat']
tips_day.shape # (87, 7)

# 2) 성별 = 여성
tips_day_gender = tips_day[tips_day['sex'] == 'Female']
tips_day_gender.shape # (28, 7)

# 3) 전체 28건 중에서 흡연유무 빈도수
tips_day_gender['smoker'].value_counts()
'''
Yes    15 -> 흡연자 
No     13 -> 비흡연자 = 성공횟수 
시행회수 = 28
'''

N = 28 # 시행횟수
x = 13 # 성공횟수
P = 0.5 # 성공확률

pvalue = stats.binom_test(x=x, n=N, p=P, alternative='two-sided')

pvalue # 0.8505540192127226

# [해설] 비흡연자와 흡연자의 비율은 차이가 없다고 볼 수 있다.

import numpy as np

np.random.uniform(size=100)





