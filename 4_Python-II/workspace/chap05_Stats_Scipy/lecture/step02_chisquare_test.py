# -*- coding: utf-8 -*-
"""
카이제곱검정(chisquare test)
 - 일원 카이제곱, 이원 카이제곱
"""

from scipy import stats
import numpy as np

# 1. 일원 카이제곱 검정
# 귀무가설 : 관측치와 기대치는 차이가 없다. (게임에 적합하다.)
# 대립가설 : 관측치와 기대치는 차이가 있다. (게임에 적합하지 않다.)
real_data = [4, 6, 17, 16, 8, 9] # 관측치
exp_data = [10,10,10,10,10,10] # 기대치
chis = stats.chisquare(real_data, exp_data)
chis
#(statistic=14.200000000000001 = 기대비율, pvalue=0.014387678176921308)
# 카이제곱 χ2 = Σ (관측값 - 기댓값)^2 / 기댓값

print('statistic = %.3f, pvalue = %.3f'%(chis))
# statistic = 14.200, pvalue = 0.014

# 공식을 적용해서 기대비율이 제대로 나오는지 확인하기

# list -> numpy
real_arr = np.array(real_data)
exp_arr = np.array(exp_data)

chis2 = sum((real_arr - exp_arr)**2 / exp_arr) #sum은 빌트인 함수이다.
chis2 #14.200000000000001

# 2. 이원 카이제곱 검정
import pandas as pd

'''
 교육수준 vs 흡연유무 독립성 검정
 귀무가설 : 교육수준과 흡연유무 간의 관련성이 없다.
'''

smoke = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\smoke.csv")
smoke.info()
'''
RangeIndex: 355 entries, 0 to 354
Data columns (total 2 columns):
'''

# DF -> vector
education = smoke.education
smoking = smoke.smoking

# cross table
table = pd.crosstab(education, smoking)
table
'''
smoking     1   2   3
education            
1          51  92  68
2          22  21   9
3          43  28  21
'''


chis = stats.chisquare(education, smoking)
chis
#statistic=347.66666666666663, pvalue=0.5848667941187113

# pvalue=0.5848667941187113 >= 0.05 : 기각할 수 없음. 귀무가설 채택
# -> 교육수준과 흡연유무 간의 관련성은 없다.

'''
성별 vs 흡연유무 독립성 검정
'''
tips = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\tips.csv")
tips.info()

gender = tips.sex
smoker = tips.smoker

table = pd.crosstab(gender, smoker)
table
'''
smoker  No  Yes
sex            
Female  54   33
Male    97   60
'''

chis = stats.chisquare(gender, smoker) # str타입이면 -> 에러 발생
# 이 문자들을 숫자로 범위변수로 만들어주는 작업이 필요하다.

# dummy 생성 : 0 or 1 -> 1(Male) or 2(Female)
# dummy 생성 : 0 or 1 -> 1(Yes) or 2(No) 흡연유무
# R에는 factor형이 있지만, python에는 없기 때문에 이런 작업이 필요하다.

gender_dummy = [1 if g == 'Male' else 2 for g in gender]
# g가 male이면 1, 아니면 2로 하겠다는 뜻

smoker_dummy = [1 if s == 'Yes' else 2 for s in smoker]

'''
# 참고 dummy 생성 : 0 or 1 -> 1(No) or 2(Yes)
smoker_dummy = [1 if s =='No' else 2  for s in smoker]
'''


chis = stats.chisquare(gender_dummy, smoker_dummy)
chis
# Power_divergenceResult(statistic=81.5, pvalue=1.0)



















