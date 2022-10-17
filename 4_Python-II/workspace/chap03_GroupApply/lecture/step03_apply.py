# -*- coding: utf-8 -*-
"""
1. group 객체에 외부 함수 적용
  - obj.apply(func1)
  - obj.agg([fun1, fun2, ...])

2. data 정규화
"""

import pandas as pd

# 1. group 객체에 외부 함수 적용
'''
apply vs agg
 - 공통점 : 그룹 객체에 외부 함수 적용
 - 차이점 : 적용할 함수의 개수
'''

# apply()
test = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\test.csv")
test.info()

grp = test['data2'].groupby(test['key'])
grp.size()
'''
key
a    3
b    3
'''
grp.sum()
'''
key
a    300
b    500
'''
grp.max()
grp.min()

# 사용자 정의함수
def diff(grp):
    result = grp.max() - grp.min()
    return result
    
# 내장함수 적용
grp.apply(sum)
grp.apply(max)
grp.apply(min)

# 사용자 함수 적용
grp.apply(diff)  
'''
key
a      0
b    100
'''
# agg([fun1, fun2, ...])
agg_func = grp.agg(['sum', 'max', 'min', 'diff']) # 한번에 4개의 함수를 적용할 수 있다.
agg_func


# 2. data 정규화 : 다양한 특징을 갖는 변수(x)를 대상으로 일정한 범위로 조정하는 것
# - x(30) -> y
    
import numpy as np # max, min, log

# 1) 사용자 함수 : 0 ~ 1
def normal(x):
    n = (x - np.min(x)) / (np.max(x) - np.min(x))
    return n

x = [10, 20000, -100, 0]
normal(x)
#[0.00547264, 1.        , 0.        , 0.00497512]
# 가장 작은 값은 0으로 최대값은 1로 맞춰지고 나머지는 0과 1 사이의 임의의 값으로 맞춰짐

# 2) 자연 log
np.log(x) # 밑수 e : 음수 이거나 0인 경우 -> 음수 : 결측치, 0 :무한대
e = np.exp(1)
e #2.718281828459045
#[2.30258509, 9.90348755,        nan,       -inf]

# 로그 -> 지수값(8 = 2^3)
np.log(10) # 밑수 e : 2.302585092994046 즉, e^2.3025 = 10이라는 뜻
e**2.3025 #9.99914910 약 10이 나오는 것을 볼 수 있다.

# 지수 -> 로그값
np.exp(2.302585092994046) #10.000000000000002

'''
로그함수과 지수함수는 역함수 관계
 - 로그 : 지수값 반환
 - 지수 : 로그값 반환
'''

iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\iris.csv")
# 전체 컬럼명 가져오기
cols = list(iris.columns) #컬럼들을 가져와 리스트형으로 변환
cols #['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']

iris_x = iris[cols[:4]]
iris_x.shape #(150, 4)
iris_x.head()

# x변수 정규화
iris_x_nor = iris_x.apply(normal) #정규화를 할 수 있는 함수를 불러옴.
iris_x_nor.head()


iris_x.agg(['var', 'mean', 'max', 'min'])



















