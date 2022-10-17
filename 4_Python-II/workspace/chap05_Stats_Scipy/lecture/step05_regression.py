# -*- coding: utf-8 -*-
"""
scipy 패키지의 state 모듈의 함수
 - 통계적인 방식의 회귀분석
1. 단순선형회귀모델
2. 다중선형회귀모델
"""

from scipy import stats # 회귀모델 생성
import pandas as pd # csv file read

# 1. 단순선형회귀모델
'''
x -> y
'''
score_iq = pd.read_csv("C:/ITWILL/4_Python-II/data/score_iq.csv")
score_iq.info()

# 변수 선택
x = score_iq.iq
y = score_iq['score']

# 회귀모델 생성
model = stats.linregress(x, y)
model
'''
LinregressResult(slope=0.6514309527270075, : 기울기
intercept=-2.8564471221974657, : 절편
rvalue=0.8822203446134699, : 설명력
pvalue=2.8476895206683644e-50, : x의 유의성 검정 (x변수가 y에 어떤 영향을 미치는지)
stderr=0.028577934409305443) : 표본 오차
'''

print('x 기울기 =', model.slope)
print('y 절편 =', model.intercept)
'''
x 기울기 = 0.6514309527270075
y 절편 = -2.8564471221974657
'''

score_iq.head(1)
'''
     sid  score   iq  academy  game  tv
0  10001     90  140        2     1   0
'''

# y = X * a + b
X = 140
y_pred = X * model.slope + model.intercept
y_pred #88.34388625958358

Y = 90
err = Y - y_pred
err #1.6561137404164157

##################
## product.csv
##################
product = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\product.csv")
product.info()

product.corr()
'''
          a         b         c
a  1.000000  0.499209  0.467145
b  0.499209  1.000000  0.766853 
c  0.467145  0.766853  1.000000
'''

# x : 제품 적절성 -> y : 제품 만족도

model2 = stats.linregress(product['b'], product['c'])
model2
'''
LinregressResult(slope=0.7392761785971821,
intercept=0.7788583344701907,
rvalue=0.766852699640837,
pvalue=2.235344857549548e-52,
stderr=0.03822605528717565)
'''

product.head(1)
'''
   a  b(x)  c(y)
0  3  4     3
'''
X = 4
y_pred = X * model2.slope + model2.intercept
y_pred # y의 예측치
Y = 3
err = Y - y_pred
err #-0.7359630488589191

err = (Y - y_pred) ** 2
err #0.5416416092857157

#-0.7359630488589191 -> 0.5416416092857157

X = product['b']
y_pred = X * model2.slope + model2.intercept # 예측치
Y = product['c'] # 관측치

len(y_pred) #264
y_pred[:10]
Y[:10]

# 2. 회귀모델 시각화
from pylab import plot, legend, show

plot(product['b'], product['c'], 'b.') # (x, y) -산점도
plot(product['b'], y_pred, 'r.-') # 회귀선
legend(['x,y scatter', 'regress model line'])
show()

# 3. 다중선형회귀모델 : y ~ x1 + x2, ...
from statsmodels.formula.api import ols #function

wine = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\winequality.csv")
wine.info()

#컬럼에 . 이나 공백이 있으면 사용할 때 힘들다.
# 예를 들면 volatile acidity는 한 컬럼이지만 volatile가 x1, acidity가 x2로 인식된다.
wine.columns = wine.columns.str.replace(' ', '_')
wine.info()

# quality vs other
cor = wine.corr()
cor['quality']
formula = "quality ~ alcohol + chlorides + volatile_acidity"

model = ols(formula, data = wine).fit()
model #object info

model.summary() #분석 결과 확인
# Adj. R-squared:                  0.259 -> 설명력
# F-statistic:                     758.6
# Prob (F-statistic): 모델 유의성 검정 기준 0.00 < 0.05 : 통계적으로 유의하다.
# x의 유의성 검정

# 기울기, 절편 // params
model.params
'''
Intercept           2.909941
alcohol             0.319578
chlorides           0.159258
volatile_acidity   -1.334944
dtype: float64
'''

#model. 후 Ctrl + spacebar하면 쓸 수 있는 목차가 뜬다.

# y의 예측치 vs 관측치 // fittedvalues
y_pred = model.fittedvalues #예측치
y_true = wine['quality'] #관측치

err = (y_true - y_pred) **2
err

y_true[:10]
y_pred[:10]

y_true.mean() #5.818377712790519
y_pred.mean() #5.81837771279059

# 차트 확인
import matplotlib.pyplot as plt

plt.plot(y_true[:10], 'b', label='real values')
plt.plot(y_pred[:10], 'r', label='y prediction')
plt.yticks(range(0, 10)) # 0에서 10 사이의 눈금
plt.legend(loc = 'best')
plt.show()












