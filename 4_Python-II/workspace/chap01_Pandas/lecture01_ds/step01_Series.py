# -*- coding: utf-8 -*-
"""
step01_Series.py

Series 객체 특징
    - 1차원의 배열구조
    - 수학/통계 함수 제공
    - 범위 수정, 블럭 연산
    - indexing/slicing 기능
    - 시계열(time series) 데이터 생성
"""

import pandas as pd # 별칭
from pandas import Series # 패키지 import class 

# 1. Series 객체 생성

# 1) list 이용
lst = [4000, 3000, 3500, 2000]
ser = pd.Series(lst) #list -> Series
print('lst =', lst)
print('ser =\n', ser)


#object.member
print(ser.index) # 색인
# RangeIndex(start=0, stop=4, step=1)
print(ser.values) # 값
# [4000 3000 3500 2000]

print(ser[0]) # 4000

ser1_2 = Series([4000, 3000, 3500, 2000],
                index=['a','b','c','d'])
print(ser1_2.index)
# Index(['a', 'b', 'c', 'd'], dtype='object')
print(ser1_2.values)
# [4000 3000 3500 2000]


# 2) dict 이용
person = {'name':'홍길동', 'age':35, 'addr':'서울시'}
Ser2 = Series(person)
print('ser =\n', Ser2)

'''
ser =
 name    홍길동
age      35
addr    서울시
dtype: object
'''
print(Ser2.index)
print(Ser2.values)


ser1_2 = Series([4000, 3000, 3500, 2000],
                index=['a', 'b', 'c', 'd'])
print(ser1_2.index)
print(ser1_2)

#index 사용 : object[]
print(Ser2['age']) #35

#boolean 조건식
print(ser[ser >= 3000])

# 2. indexing : list 동일
ser3 = Series([4, 4.5, 6, 8, 10.5]) #생성자
print(ser3)
print(ser3[0]) # 4.0
print(ser3[:3]) # 0 ~ 2
print(ser3[3:]) # 3 ~ n
print(ser3[-1]) # Error : list와의 차이

# 3. Series 결합과 NA 처리
p1 = Series([400, None, 350, 200],
            index = ['a', 'b', 'c', 'd'])
p2 = Series([400, 150, 350, 200],
            index = ['a', 'c', 'd', 'e'])

# Series 결합
p3 = p1 + p2
print(p3)

'''
a    800.0
b      NaN -> 결측치
c    500.0
d    550.0
e      NaN -> 결측치
'''

# 4. 결측치 처리 방법(평균, 0, 제거)
print(type(p3)) #<class 'pandas.core.series.Series'>

# 1) 평균으로 대체
p4 = p3.fillna(p3.mean())
print(p4)

# 2) 0으로 대체
p5 = p3.fillna(0)
print(p5)

# 3) 결측치 제거
p6 = p3[pd.notnull(p3)] #subset 
print(p6)

'''
a    800.0
c    500.0
d    550.0
'''

# 4) 분산으로 대체
p7 = p3.fillna(p3.var())
print(p7)

# 5. 범위 수정, 블럭 연산
print(p2)
'''
a    400
c    150
d    350
e    200
'''
# 1) 범위 수정
p2[1:3] = 300
print(p2)
'''
a    400
c    300
d    300
e    200
'''

#list에서는 범위 수정은 사용 불가
#lst = [1, 2, 3, 4]
#lst[1:3] = 3

# 2) 블럭 연산
print(p2 + p2) # 2배
print(p2 - p2) # 0

# 3) broadcast 연산(1차원 vs 0차원)
v1 = Series([1,2,3,4])
scala = 0.5

b = v1 * scala # vector(1) * scale(0)
print(b)
'''
0    0.5
1    1.0
2    1.5
3    2.0
dtype: float64
'''

#for문을 써도 되나 코드가 길어지는 단점이 있다.
for i in v1:
    b = i * scala
    print(b)

# 4) 수학/통계 함수
print('sum =', v1.sum())
print('mean =', v1.mean())
print('var =', v1.var())
print('std =', v1.std())
print('max =', v1.max())

# 호출 가능한 멤버 확인(수학/통계 함수)
print(dir(v1))

print(v1.shape) # (4,)
print(v1.size) # 4




















