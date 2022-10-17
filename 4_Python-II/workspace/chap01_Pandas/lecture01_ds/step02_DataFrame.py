# -*- coding: utf-8 -*-
"""
DataFrame 자료구조 특징
    - 2차원 행렬구조(table 유사함)
    - 열(컬럼) 단위 데이터 처리 용이
    - Series(1차)의 모임 -> DataFrame(2차) 
"""

import pandas as pd
from pandas import Series, DataFrame

# 1. DataFrame 생성

# 1) 기본 자료구조(list, dict) 이용
name = ['hong', 'lee', 'kang', 'yoo'] #list
age = [35, 45, 55, 25] #list
pay = [250, 350, 450, 200]
addr = ['서울시', '부산시', '대전시', '인천시']

#dict 키는 랜덤
data = {'name' : name, 'age':age, 'pay':pay, 'addr':addr}

frame = pd.DataFrame(data=data)
print(frame)
'''
   name  age  pay addr
0  hong   35  250  서울시
1   lee   45  350  부산시
2  kang   55  450  대전시
3   yoo   25  200  인천시
'''

frame = pd.DataFrame(data=data,
                     columns = ['name', 'age', 'addr', 'pay'])
print(frame)
'''
   name  age addr  pay
0  hong   35  서울시  250
1   lee   45  부산시  350
2  kang   55  대전시  450
3   yoo   25  인천시  200
'''

# 칼럼 추출
age = frame['age'] #frame.age
print(age.mean()) # 40
print(type(age)) # <class 'pandas.core.series.Series'>

# 새 칼럼 추가
gender = Series(['남','남','남','여']) # 1차
frame['gender'] = gender
print(frame)

# 2) numpy 이용 : 선형대수 관련 함수
import numpy as np
frame2 = DataFrame(np.arange(12).reshape(3,4),
                   columns = ['a','b','c','d'])

print(frame2)
'''
   a  b   c   d
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
'''

# 행/열 통계 구하기
ax0 = frame2.mean(axis = 0) # 행축 : 열단위
ax1 = frame2.mean(axis = 1) # 열축 : 행단위

print('열 단위 평균 :', ax0)
print('행 단위 평균 :', ax1)


# 2. DataFrame 칼럼 참조
frame2.index # 행 이름
# RangeIndex(start=0, stop=3, step=1)
frame2.values # 값들
'''
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
'''

#emp.csv
emp = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\emp.csv",
             encoding='utf-8')
print(emp.info()) # str(emp)

emp.head() #head(emp)

# 1) 단일 컬럼 선택
print(emp['No'])
print(emp.No) # 칼럼명에 점 포함된 경우(x)
print(emp.No[1:]) #특정 원소 선택
no = emp.No
no.plot()
pay = emp['Pay']
pay.plot() #선 그래프가 기본차트로 만들어진다.


# 2) 복수 컬럼 선택 : 중첩 list
print(emp[['No', 'Pay']]) #2개 이상 보려면 중첩리스트를 사용해야한다.
#print(emp[['No' : 'Name']]) # :쓰면 Error, R과의 차이이다.

emp[['No', 'Pay']].plot() #범례도 기본으로 만들어진다.

# 3. subset 만들기 : old DF -> new DF

# 1) 특정 칼럼 제외 : 칼럼 적은 경우
emp.info()
subset1 = emp[['Name', 'Pay']]
subset1

# 2) 특정 행 제외
subset2 = emp.drop(0) #한 두개 정도의 행을 제외하고 나머지로 서브셋을 만들 때 주로 사용함.
subset2

# 3) 조건식으로 행 선택
subset3 = emp[emp.Pay > 350]
subset3

# 4) columns 이용 : 칼럼 많은 경우
iris = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\iris.csv")
print(iris.info()) # file info
print(iris.head()) # 앞부분 5개 관측치

iris['Sepal.Width']
iris.Sepal.Width #컬럼에 .이 포함되어 있으면 위의 방법을 활용해야 Error가 나지 않는다.

cols = list(iris.columns) #칼럼명 추출
print(cols)
type(cols) #list

iris_x = cols[:4]
iris_y = cols[-1]
print(iris_x) #['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
print(iris_y) # Species

# x, y 변수 선택
x = iris[iris_x]
y = iris[iris_y]
x.shape # (150, 4) : 2차원
y.shape # (150,) : 1차원

x.head()
y.head()
y.tail()

# 4. DataFrame 행렬 참조 : DF[row, col]
# 1) DF.loc[row, col] : label index 사용
# 2) DF.iloc[row, col] : integer index 사용

emp
'''
    No Name  Pay
0  101  홍길동  150
1  102  이순신  450
2  103  강감찬  500
3  104  유관순  350
4  105  김유신  400
열 이름 : No Name  Pay
행 이름 : 0 1 2 3 4
'''

emp.loc[1:3] #순서하게 인덱스 이름인 1번부터 3번까지 가져오겠다/ 안의 것을 숫자로 보지말고 인덱스 이름으로 볼 것.
'''
    No Name  Pay
1  102  이순신  450
2  103  강감찬  500
3  104  유관순  350
'''

# loc[행label index, 열labelindex]
emp.loc[1:3, :] #label index
emp.loc[1:3] # 위와 동일한 결과 # 3행
emp.loc[1:3, 'No':'Name']
'''
    No Name  Pay
1  102  이순신  450
2  103  강감찬  500
3  104  유관순  350
'''

emp.loc[1:3, ['No','Pay']]
'''
    No  Pay
1  102  450
2  103  500
3  104  350
'''
# 그렇다면 열label index 자리에 숫자를 넣으면?
emp.loc[1:3, [0,2]] #error가 난다.


# iloc[행숫자 index, 열숫자 index]
emp.iloc[1:3] # 숫자 index : 2행
'''
    No Name  Pay
1  102  이순신  450
2  103  강감찬  500
'''

emp.iloc[1:3, 0:2]
emp.iloc[1:3, :2]#컬럼이 연속인 경우 
'''
    No Name
1  102  이순신
2  103  강감찬
'''
emp.iloc[1:3, [0,2]] #컬럼이 불연속인 경우 

# 그렇다면 열숫자 index 자리에 문자를 넣으면?
emp.iloc[1:3, ['No','Pay']] #error가 난다.


#################################
##### DF 행렬 참조 example
#################################
iris.shape # (150, 5)

from numpy.random import choice
help(choice)
# choice(a, size=None, replace=True, p=None)

row_idx = choice(a=len(iris), size=int(len(iris)*0.7), replace = False)
#len(iris)*0.7 // 150개중 105개만 취해 훈련데이터셋으로 취하겠다는 뜻. True 중복
row_idx
len(row_idx) # 105

train_set = iris.iloc[row_idx]
train_set.head()
train_set.shape # (105, 5)


train_set2 = iris.loc[row_idx]
train_set2.head()
# 행의 index도 숫자이기 때문에 결괏값은 같다.
train_set2.shape # (105, 5)

# test dataset : list + for
test_idx = [i for i in range(len(iris)) if not i in row_idx] # i = 0~149

len(test_idx) # 45

test_set = iris.iloc[test_idx]
test_set.shape # (45, 5)

# x, y 변수 분리
cols = list(iris.columns)
x = cols[:4]
y = cols[-1]

iris_x = iris.loc[test_idx, x]
iris_y = iris.loc[test_idx, y]
iris_x.shape # (45, 4)
iris_y.shape # (45,)



















