# -*- coding: utf-8 -*-
"""
시계열 분석(time series analysis)
 1. 시계열 자료 생성
 2. 날짜형식 수정(다국어)
 3. 시계열 시각화
 4. 이동평균 기능 : 5, 10, 20일 평균 -> 추선세 평활(스뮤딩)
"""

from datetime import datetime # 날짜형식 수정
import pandas as pd # csv file read
import matplotlib.pyplot as plt # 시계열 시각화
import numpy as np # 수치 자료 생성


# 1. 시계열 자료 생성
time_data = pd.date_range("2017-03-01", "2020-03-30")
time_data # length=1126, freq='D' (D : default)

# 월 단위 시계열 자료
time_data2 = pd.date_range("2017-03-01", "2020-03-30", freq = 'M')
time_data2 # freq='M'
len(time_data2) # 36

# 월 단위 매출현황
x = pd.Series(np.random.uniform(10, 100, size=36))

df = pd.DataFrame({'data' : time_data2, 'price' : x})
df

plt.plot(df['data'], df['price'], 'g--') # (x축, y축)
plt.show()

# 2. 날짜형식 수정(다국어)
cospi = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\cospi.csv")
cospi.info()

cospi.head()
'''
cospi
        Date     Open     High      Low    Close  Volume
0  26-Feb-16  1180000  1187000  1172000  1172000  176906
1  25-Feb-16  1172000  1187000  1172000  1179000  128321
'''

data = cospi['Date']
len(data) #247

# list + for : 26-Feb-16 -> 2016-02-26  #윈도우 운영체제에 맞는 포맷으로 변경해주는 것
kdate = [datetime.strptime(d, '%d-%b-%y') for d in data]
kdate 

# 날짜 칼럼 수정
cospi['Date'] = kdate
cospi.head()
'''
        Date     Open     High      Low    Close  Volume
0 2016-02-26  1180000  1187000  1172000  1172000  176906
1 2016-02-25  1172000  1187000  1172000  1179000  128321
2 2016-02-24  1178000  1179000  1161000  1172000  140407
3 2016-02-23  1179000  1189000  1173000  1181000  147578
4 2016-02-22  1190000  1192000  1166000  1175000  174075
'''

cospi.tail()

# 3.시계열 시각화
cospi.index #RangeIndex(start=0, stop=247, step=1)

# 칼럼 -> index 적용
new_cospi = cospi.set_index('Date')
new_cospi.index

new_cospi['2016']
len(new_cospi['2015']) # 210
new_cospi['2015']
new_cospi['2015-05':'2015-03'] #큰 수를 앞에 둠.

# subset
new_cospi_HL = new_cospi[['High', 'Low']]
new_cospi_HL.index # Date
new_cospi_HL.columns #['High', 'Low']

# 2015년 기준
new_cospi_HL['2015'].plot(title = '2015 year : High vs Low')
plt.show()

# 2016년 기준
new_cospi_HL['2016'].plot(title = '2016 year : High vs Low')
plt.show()

# 2016년 2월 기준
new_cospi_HL['2016-02'].plot(title = '2016 year : High vs Low')
plt.show()


# 4. 이동평균 기능 : 5, 10, 20일 평균 -> 추선세 평활(스뮤딩)

# 1) 5일 단위 이동평균 : 5일 단위 평균 -> 마지막 5일째 이동 (rolling mean)
# 주말에는 열지 않는 주식을 일주일 단위로 분석할 수 있다.

roll_mean5 = pd.Series.rolling(new_cospi_HL,
                               window=5, center=False).mean()
# default(center에 아무 입력을 안할 때) 가운데부터 5일씩 평균을 구한다.
# 처음부터 5일씩 평균을 구하려면 False를 준다.

roll_mean5

# 2) 10일 단위 이동평균 : 10일 단위 평균 -> 마지막 10일째 이동
roll_mean10 = pd.Series.rolling(new_cospi_HL,
                               window=10, center=False).mean()
roll_mean10


# 3) 20일 단위 이동평균 : 20일 단위 평균 -> 마지막 20일째 이동
roll_mean20 = pd.Series.rolling(new_cospi_HL,
                               window=20, center=False).mean()
roll_mean20[:25]


# 4) 5일 단위 이동평균 : 5일 단위 평균 -> 마지막 5일째 이동 (High만)
roll_mean5h = pd.Series.rolling(new_cospi_HL.High,
                               window=5, center=False).mean()
roll_mean5h

# 5) 10일 단위 이동평균 : 10일 단위 평균 -> 마지막 10일째 이동 (High만)
roll_mean10h = pd.Series.rolling(new_cospi_HL.High,
                               window=10, center=False).mean()
roll_mean10h

# 6) 20일 단위 이동평균 : 20일 단위 평균 -> 마지막 20일째 이동 (High만)
roll_mean20h = pd.Series.rolling(new_cospi_HL.High,
                               window=20, center=False).mean()
roll_mean20h[:25]



# rolling mean 시각화
new_cospi_HL.High.plot(color = 'b', label = 'High column') # 원본
roll_mean5h.plot(color = 'red', label = 'rolling mean 5day')
roll_mean10h.plot(color = 'green', label = 'rolling mean 10day')
roll_mean20h.plot(color = 'orange', label = 'rolling mean 20day')
plt.legend(loc = 'best') #범례
plt.show()











