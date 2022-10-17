# -*- coding: utf-8 -*-
"""
1. csv file read
2. csv file write
3. random data sampling
"""

import pandas as pd

# 1. csv file read
iris = pd.read_csv('C:\\ITWILL\\4_Python-II\\data\\iris.csv')
iris.info()

# 칼럼명 : 특수문자 or 공백 -> _ 문자 변경
iris.columns = iris.columns.str.replace('.','_') # .를 _로 교체하겠다는 뜻.
iris.info()
iris.Sepal_Length #이런 식으로 불러올 때 .이나 공백을 포함한 칼럼명은 오류가 난다.

# 컬럼명이 없는 경우
st = pd.read_csv('C:\\ITWILL\\4_Python-II\\data\\student.csv', header=None)
st #     0     1    2   3 인덱스처럼 숫자로 컬럼명이 만들어짐.
col_names = ['학번', '이름', '키', '몸무게']
st.columns = col_names
st

# 행 이름 변경
#st.index = 수정 값

# 비만도 지수(BMI)
# BMI = 몸무게 / 키**2 (단, 몸무게 : kg, 키 : m)
BMI = [st.loc[i,'몸무게'] / (st.loc[i, '키']*0.01)**2
       for i in range(len(st))]
print(BMI)
type(BMI) #list

st['bmi'] = BMI
st['bmi'] = st['bmi'].round(2) #셋째짜리에서 둘째자리로 소숫점 반올림
st

# 2. csv file write
st.to_csv('C:\\ITWILL\\4_Python-II\\data\\student_df.csv', header=None, encoding='utf-8')
st_df = pd.read_csv('C:\\ITWILL\\4_Python-II\\data\\student_df.csv', encoding='utf-8')
st_df.info()
st_df


# 3. random sampling
wdbc = pd.read_csv('C:\\ITWILL\\4_Python-II\\data\\wdbc_data.csv')
wdbc.info()

wdbc_train = wdbc.sample(400) # 400개만 샘플링으로 받겠단 뜻
wdbc_train.shape #(400, 32)

wdbc_train.head()




















