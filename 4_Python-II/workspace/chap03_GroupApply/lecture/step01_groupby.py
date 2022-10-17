# -*- coding: utf-8 -*-
"""
DataFrame 객체 대상 그룹화
    - 형식) DF.groupby('집단변수').수학/통계함수()
"""

import pandas as pd

tips = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\tips.csv")
tips.info()
tips.head()
tips.tail()

# 팁 비율 : 파생변수(사칙연산)
tips['tip_pct'] = tips['tip'] / tips['total_bill']
tips.info()

# 변수 복제
tips['gender'] = tips['sex']

# 변수 제거
del tips['sex']

tips.head()

# 1. 집단변수 1개 -> 전체 컬럼 그룹화
gender_grp = tips.groupby('gender')
gender_grp # object info

# 그룹객체.함수()
gender_grp.size() # 각 그룹의 빈도수
'''
gender
Female     87
Male      157
dtype: int64
'''

# 그룹 통계량 : 숫자변수만 대상
gender_grp.sum()
'''
        total_bill     tip  size    tip_pct
gender                                     
Female     1570.95  246.51   214  14.484694
Male       3256.82  485.07   413  24.751136
'''
gender_grp.mean()
'''
        total_bill       tip      size   tip_pct
gender                                          
Female   18.056897  2.833448  2.459770  0.166491
Male     20.744076  3.089618  2.630573  0.157651
'''

# 객체 -> 호출 가능한 멤버 확인
dir(gender_grp) #호출할 수 있는 멤버를 볼 때 dir을 사용한다.
#멤버란,gender_grp.sum()에서 sum 위치에 오는 것들을 말함.

# 그룹별 요약 통계량
gender_grp.describe() # 수치 제공
gender_grp.boxplot() # 그래프 제공

# 2. 집단변수 1개 -> 특정 칼럼 그룹화
smoker_grp = tips['tip'].groupby(tips['smoker'])
smoker_grp.size() # 그룹 빈도수
'''
smoker
No     151
Yes     93
'''

smoker_grp.mean()
'''
smoker
No     2.991854
Yes    3.008710
'''

# 3. 집단변수 2개 -> 전체 칼럼 그룹화
# 형식) DF.groupby(['칼럼1','칼럼2'])
# 1차 : 칼럼1, 2차: 첫번째 칼럼값이 똑같을 때 칼럼2 사용
gender_smoker_grp = tips.groupby([tips['gender'],tips['smoker']])

# 그룹 빈도수
gender_smoker_grp.size()

'''
gender  smoker
Female  No        54
        Yes       33
Male    No        97
        Yes       60
'''

# 특정 변수 통계량
gender_smoker_grp.describe()
gender_smoker_grp['tip'].describe()
# 현재 그룹핑 중에서 특정변수(tip)의 것만 요약통계내줌.
'''
               count      mean       std   min  25%   50%     75%   max
gender smoker                                                          
Female No       54.0  2.773519  1.128425  1.00  2.0  2.68  3.4375   5.2
       Yes      33.0  2.931515  1.219916  1.00  2.0  2.88  3.5000   6.5
Male   No       97.0  3.113402  1.489559  1.25  2.0  2.74  3.7100   9.0
       Yes      60.0  3.051167  1.500120  1.00  2.0  3.00  3.8200  10.0
[해설] 여성은 흡연자, 남성은 비흡연자가  tip 지불에 후하다.
'''

# 4. 집단변수 2개 -> 특종 칼럼 그룹화
gender_smoker_tip_grp = tips['tip'].groupby([tips['gender'],
                                         tips['smoker']])
gender_smoker_tip_grp.size()
'''
gender  smoker
Female  No        54
        Yes       33
Male    No        97
        Yes       60
'''
gender_smoker_tip_grp.size().shape
#(4,) : 1차원 - vector

gender_smoker_tip_grp.sum()
'''
gender  smoker
Female  No        149.77
        Yes        96.74
Male    No        302.00
        Yes       183.07
'''

gender_smoker_tip_grp.sum().shape
# (4,)

# 1d -> 2d
grp_2d = gender_smoker_tip_grp.sum().unstack()
grp_2d # 성별 vs 흡연유무 -> 교차분할표(합계)
'''
smoker      No     Yes
gender                
Female  149.77   96.74
Male    302.00  183.07
'''
gender_smoker_tip_grp.sum().unstack().shape
grp_2d.shape
# (2, 2)

##### advance ######
grp_2d = gender_smoker_tip_grp.sum().unstack()
# sum 대신 빈도수로 교차분할표를 표시하려면 size를 넣어주면 된다.
grp_2d # 성별 vs 흡연유무 -> 교차분할표(합계로 집계함)
'''
smoker      No     Yes
gender                
Female  149.77   96.74
Male    302.00  183.07
'''

# 성별 vs 흡연유무 -> 교차분할표(빈도수)
grp_2d_size = gender_smoker_tip_grp.size().unstack()
#unstack()은 교차분할표를 만들어주는 것.
grp_2d_size
'''
smoker  No  Yes
gender         
Female  54   33
Male    97   60
'''

grp_2d.shape 
# (2, 2)

# 2d -> 1d
grp_1d = grp_2d.stack()
grp_1d
'''
gender  smoker
Female  No        149.77
        Yes        96.74
Male    No        302.00
        Yes       183.07
'''

# iris dataset 그룹화
# 1) dataset load
iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\iris.csv")

iris.info()

# 2) group : group -> apply 4개 변수 그룹화
iris.groupby('Species').sum()

# 3) group -> 1개 변수 그룹화
iris['Sepal.Length'].groupby(iris['Species']).sum()
# groupby안의 형식을 주의할 것!
'''
Species
setosa        250.3
versicolor    296.8
virginica     329.4
Name: Sepal.Length, dtype: float64
'''



















