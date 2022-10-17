# -*- coding: utf-8 -*-
"""
집단 간 평균차이 검정(t-test)
 1. 한 집단 평균차이 검정
 2. 두 집단 평균차이 검정
 3. 대응 두 집단 평균차이 검정
"""

from scipy import stats
import numpy as np
import pandas as pd

# 1. 한 집단 평균차이 검정

# 대한민국 남자 평균 키(모평균) : 175.5cm
# 모집단 -> 표준 추출(300명)

sample_data = np.random.uniform(172, 180, size=300)
sample_data

# 기술통계
sample_data.mean() #176.08272655273174

one_group_test = stats.ttest_1samp(sample_data, 175.5)
#기존 집단의 평균과 한 집단의 평균을 검정하려는 것
one_group_test
#Ttest_1sampResult(statistic=4.19592309475279, pvalue=3.586642905084937e-05)
print('statistic = %.5f, pvalue = %.5f'%(one_group_test))
#statistic = 4.19592, pvalue = 0.00004 < 0.05
# -> 샘플링과 모집단 사이의 평균 차이가 있다.

# 2. 두 집단 평균차이 검정
female_score = np.random.uniform(50, 100, size = 30)
male_score = np.random.uniform(45, 95, size = 30)
two_sample = stats.ttest_ind(female_score, male_score)

two_sample
#statistic=0.9843379620930749, pvalue=0.3290378740999149 >= 0.05
#-1.96 ~ +1.96이 사이에 있으니까 채택한다. 이 사이에 없으면 기각한다.
#즉, statistic은 신뢰구간에 있다. 즉, 평균 차이가 없다.
print('statistic = %.5f, pvalue = %.5f'%(two_sample))
#statistic = 0.98434, pvalue = 0.32904

# 기술 통계
female_score.mean() #73.89207619235846
male_score.mean() #70.04987930565552


# csv file load
two_sample = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\two_sample.csv")
two_sample.info()

sample_data = two_sample[['method', 'score']] #통계검정에 필요한 2개 검정의 subset생성
sample_data.head() #method : 집단변수의 역할,  score : 평균
sample_data['method'].value_counts() #pandas의 멤버, 각 집단의 개수
'''
2    120
1    120
'''

# 교육방법에 따른 subset
method1 = sample_data[sample_data['method']==1] #method가 1인 것을 빼서 서브셋만듦
method2 = sample_data[sample_data['method']==2]

score1 = method1.score
score2 = method2.score

# 결측치 제거하기(NA -> 평균 대체)
score1 = score1.fillna(score1.mean())
score2 = score2.fillna(score2.mean())

two_sample = stats.ttest_ind(score1, score2)
two_sample
print('statistic = %.5f, pvalue = %.5f'%(two_sample))
#statistic = -0.94686, pvalue = 0.34467

score1.mean() #5.496590909090908
score2.mean() #5.591304347826086

# 3. 대응 두 집단 평균차이 검정 : 복용전 65 -> 복용후 : 60 변환
before = np.random.randint(65, size = 30) * 0.5
after = np.random.randint(60, size = 30) * 0.5

before.mean() #15.633333333333333
after.mean() #15.566666666666666

pired_test = stats.ttest_rel(before, after)
pired_test #statistic=-0.04511616811862168, pvalue=0.9643239550195635
print('statistic = %.5f, pvalue = %.5f'%(pired_test))
#statistic = 0.03209, pvalue = 0.97462 > 0.05 귀무가설을 기각할 수 없다.
#복용전과 복용후 몸무게 차이는 별 차이가 없는 것으로 볼 수 있다.

















