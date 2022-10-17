'''
문02) winequality-both.csv 데이터셋을 이용하여 다음과 같이 처리하시오.
   <조건1> quality, type 칼럼으로 교차분할표 작성 
   <조건2> 교차분할표를 대상으로 white 와인 내림차순 정렬       
   <조건3> red 와인과 white 와인의 quality에 대한 두 집단 평균 검정
           -> 각 집단 평균 통계량 출력
   <조건4> alcohol 칼럼과 다른 칼럼 간의 상관계수 출력  
'''

import pandas as pd
from scipy import stats

wine = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\winequality-both.csv")
wine.info()
wine.head()


#<조건1> quality, type 칼럼으로 교차분할표 작성 
tab = pd.crosstab(wine.quality, wine.type)
#tab = pd.crosstab(wine['quality'], wine['type']) # teacher
tab
'''
type     red  white
quality            
3         10     20
4         53    163
5        681   1457
6        638   2198
7        199    880
8         18    175
9          0      5
'''

#<조건2> 교차분할표를 대상으로 white 와인 내림차순 정렬       
tab_sort = tab.sort_values('white', ascending = False)
tab_sort

'''
type     red  white
quality            
6        638   2198
5        681   1457
7        199    880
8         18    175
4         53    163
3         10     20
9          0      5
'''

#<조건3> red 와인과 white 와인의 quality에 대한 두 집단 평균 검정
#          -> 각 집단 평균 통계량 출력
red_wine = wine.loc[wine['type']=='red', 'quality']
white_wine = wine.loc[wine['type']=='white', 'quality']

len(red_wine) #1599
len(white_wine) #4898

two_sample = stats.ttest_ind(red_wine, white_wine)
two_sample
print('statistic = %.5f, pvalue = %.5f'%(two_sample))
#statistic = -9.68565, pvalue = 0.00000 -> 양측 검정 : 귀무가설 기각

# 대립가설 채택(귀무가설이 기각되었으니) -> 단측검정(red wine < white wine)
red_wine.mean() #5.6360225140712945
white_wine.mean() #5.87790935075541


#<조건4> alcohol 칼럼과 다른 칼럼 간의 상관계수 출력  
cor = wine.corr()
cor

cor['alcohol'] #cor.alcohol
'''
fixed acidity          -0.095452
volatile acidity       -0.037640
citric acid            -0.010493
residual sugar         -0.359415
chlorides              -0.256916
free sulfur dioxide    -0.179838
total sulfur dioxide   -0.265740
density                -0.686745
pH                      0.121248
sulphates              -0.003029
alcohol                 1.000000
quality                 0.444319
Name: alcohol, dtype: float64

alcohol과 가장 상관관계가 높은 것은 quality라고 할 수 있다.
'''






