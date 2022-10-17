'''
문1) score_iq.csv 데이터셋을 이용하여 단순선형회귀모델을 생성하시오.
   <조건1> y변수 : score, x변수 : academy      
   <조건2> 회귀모델 생성과 결과확인(회귀계수, 설명력, pvalue, 표준오차) 
   <조건3> 회귀선 적용 시각화 
   
문2) iris.csv 데이터셋을 이용하여 다중선형회귀모델을 생성하시오.
   <조건1> 칼럼명에 포함된 '.' 을 '_'로 수정
   iris = pd.read_csv('../data/iris.csv')
   iris.columns = iris.columns.str.replace('.', '_')
   <조건2> y변수 : 1번째 칼럼, x변수 : 2~4번째 칼럼    
   <조건3> 회귀계수 확인 
   <조건4> 회귀모델 세부 결과 확인  : summary()함수 이용 
'''

from scipy import stats
import pandas as pd
import statsmodels.formula.api as sm

#문1_
score_iq = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\score_iq.csv")
score_iq

#<조건1> y변수 : score, x변수 : academy    
x = score_iq.academy
y = score_iq['score']

#<조건2> 회귀모델 생성과 결과확인(회귀계수, 설명력, pvalue, 표준오차)
model = stats.linregress(x, y)
model
'''
LinregressResult
(slope=4.847829398324446,
intercept=68.23926884996192,
rvalue=0.8962646792534938, 
pvalue=4.036716755167992e-54, 
stderr=0.1971936807753301)
'''

#<조건3> 회귀선 적용 시각화 

y_pred = x * model.slope + model.intercept

from pylab import plot, legend, show

plot(x, y, 'bo', label = 'x, y scatter') # 회귀선
plot(x, y_pred, 'r.-', label = 'y pred')
legend(loc = 'best')
show()

#문2
iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\iris.csv")
iris.info()

#<조건1> 칼럼명에 포함된 '.' 을 '_'로 수정
iris.columns = iris.columns.str.replace('.', '_')

#<조건2> y변수 : 1번째 칼럼, x변수 : 2~4번째 칼럼 
cor = iris.corr()
cor['Sepal_Length']
formula = "Sepal_Length ~ Sepal_Width  + Petal_Length + Petal_Width + Species"


from statsmodels.formula.api import ols

model = ols(formula, data = iris).fit()
model #object info

#<조건3> 회귀계수 확인 
model.params
'''
Intercept                2.171266
Species[T.versicolor]   -0.723562
Species[T.virginica]    -1.023498
Sepal_Width              0.495889
Petal_Length             0.829244
Petal_Width             -0.315155
dtype: float64
'''

#<조건4> 회귀모델 세부 결과 확인  : summary()함수 이용 
model.summary()
# Adj. R-squared:                  0.863
# F-statistic:                     188.3 (-1.96 ~ +1.96)
# Prob (F-statistic):           2.67e-61 < 0.05

# 예측치 vs 관측치 비교
y_pred = model.fittedvalues
y_true = iris['Sepal_Length']

import matplotlib.pyplot as plt
plt.plot(y_pred, 'r.-', label = 'fitted values')
plt.plot(y_true, 'b-', label='real values')
plt.legend(loc = 'best')
plt.show()



### 위에 내가 한 것은 Species 포함한 것 아래는 포함하지 않은 teacher것

formula = "Sepal_Length ~ Sepal_Width + Petal_Length + Petal_Width"
model = ols(formula, data = iris).fit()

# 회귀계수
model.params
'''
Intercept       1.855997
Sepal_Width     0.650837
Petal_Length    0.709132
Petal_Width    -0.556483
'''

# 회귀모델 결과
model.summary()
# Adj. R-squared:                  0.856
# F-statistic:                     295.5(-1.96 ~ +1.96)
# Prob (F-statistic):           8.59e-62 < 0.05
'''
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        1.8560      0.251      7.401      0.000       1.360       2.352
Sepal_Width      0.6508      0.067      9.765      0.000       0.519       0.783
Petal_Length     0.7091      0.057     12.502      0.000       0.597       0.821
Petal_Width     -0.5565      0.128     -4.363      0.000      -0.809      -0.304
'''

# 예측치 vs 관측치 비교
y_pred = model.fittedvalues
y_true = iris['Sepal_Length']

import matplotlib.pyplot as plt
plt.plot(y_pred, 'r.-', label='fitted values')
plt.plot(y_true, 'b-', label='real values')
plt.legend(loc = 'best')
plt.show()












