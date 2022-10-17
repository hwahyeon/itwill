# -*- coding: utf-8 -*-
"""
scipy 패키지의 확률분포 검정
 1. 정규분포 검정 : 연속변수의 확률분포
    - 연속확률분포 : 정규분포, 균등분포, 카이제곱, T/Z/F 분포
 2. 이항분포 검정 : 2가지(성공/실패) 분포의 확률분포
    - 이산확률분포 : 베르누이 분포, 이항 분포, 포아송 분포
"""

from scipy import stats # 확률분포 검정
import numpy as np
import matplotlib.pyplot as plt # 히스토그램

# 1. 정규분포 검정 : 평균 중심 좌우 대칭성 검정

# 1) 정규분포객체 생성
mu = 0; std = 1 #표준정규분포
std_norm = stats.norm(mu, std) #정규분포객체 생성
std_norm #object info

# 2) 정규분포확률변수
N = 1000 # 시행횟수
norm_data = std_norm.rvs(N) # 시뮬레이션 : 1000난수 생성
len(norm_data) #1000
norm_data

# 3) 히스토그램 : 좌우 대칭성
plt.hist(norm_data)
plt.show()

# 4) 정규성 검정
# 귀무가설(H0) : 정규분포와 차이가 없다.
test_stats, pvalue = stats.shapiro(norm_data)
'''
(검정통계량 : 0.9979461431503296, pvalue : 0.26081162691116333)
'''
print('검정통계량 : %.5f'%(test_stats))
print('pvalue : %.5f'%(pvalue))
'''
검정통계량 : 0.99795 -> -1.96 ~ +1.96 : 채택역
pvalue : 0.26081 >= 알파(0.05) : 채택역
'''

# 2. 이항분포 검정 : 2가지(성공/실패) 분포의 확률분포 + 가설검정
'''
 - 베르누이 분포 : 이항변수(성공 or 실패)에서 성공(1)이 나올 확률분포(모수 : 성공확률)
 - 이항분포 : 베르누이 분포에 시행횟수(N)을 적용한 확률분포(모수 : P, N)

ex) P = 게임에 이길 확률(40%), N = 시행횟수(100) -> 성공횟수(?)
'''

N = 100 # 시행횟수
P = 0.4 # 성공확률

# 1) 베르누이 분포 -> 이항분포 확률변수
x = stats.bernoulli(P).rvs(N) # 성공확률 40% -> 100번 시뮬레이션
x # 0(실패) or 1(성공)

# 2) 성공횟수
x_succ = np.count_nonzero(x)
print('성공횟수 =', x_succ) # 성공횟수 = 36

# 3) 이항분포 검정 : 이항분포에 대한 가설검정

# 귀무가설 : 게임에 이길 확률은 40%와 다르지 않다.
pvalue = stats.binom_test(x=x_succ, n=N, p=P, alternative = 'two-sided')
'''
x : 성공횟수
n : 시행횟수
p : 성공확률
alternative = 'two-sided' : 양측검정
'''

pvalue # 0.5005662441253662

if pvalue >= 0.05:
    print('게임에 이길 확률은 40%와 다르지 않다. 귀무가설 채택')
else:
    print('게임에 이길 확률은 40%와 다르다고 볼 수 있다.')


'''
100의 응시자 중에서 남자 합격자는 45일 때
남녀 합격률에 차이가 있다고 할 수 있는가?
'''


# 귀무가설 : 남여 합격률에 차이가 없다.(p=0.5)
N = 100 # 시행횟수
x = 45 # 성공횟수
P = 0.5 # 성공확률

pvalue = stats.binom_test(x=x, n=N, p=P, alternative='two-sided')

print('유의확률 =', pvalue)
# 유의확률 = 0.36820161732669654













































