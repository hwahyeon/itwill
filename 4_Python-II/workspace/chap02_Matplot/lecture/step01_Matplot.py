# -*- coding: utf-8 -*-
'''
matplotlib API 사용 차트 그림
    형식) plt.plot(data); plt.show()

1. 기본 차트 그리기 : plt.plot()
2. 산점도 그리기 : plt.scatter()
3. subplot 이용 차트 그리기 : 격자 형식 차트
'''

import matplotlib.pyplot as plt #별칭 부여

# 차트에서 한글 지원 
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 음수 부호 지원 
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import pandas as pd

import numpy as np # 숫자 데이터 생성

# 1. 기본차트 그리기
help(plt.plot)
'''
plot(x, y)        # plot x and y using default line style and color
plot(x, y, 'bo')  # plot x and y using blue circle markers
plot(y)           # plot y using x as index array 0..N-1
plot(y, 'r+')     # ditto, but with red plusses
'''

data = np.arange(10) # 0~9
data

plt.plot(data) #plot(y)
plt.show()

plt.plot(data, 'r+')
plt.show()

# 평균=0, 표준편차=1 : 표준정규분포
data2 = np.random.randn(10)
plt.plot(data, data2) #(x, y)
plt.show()

plt.plot(data, data2, 'ro')
plt.show()

# 2. 산점도 그리기
# 단색
plt.scatter(data, data2, c='r', marker='o')
plt.show()

# 여러가지 색상 산점도
cdata = np.random.randint(1,4, 10) # 1~3
cdata # 임의의 변수

plt.scatter(data, data2, c=cdata, marker='o') #수에 따라 색상이 달라짐.
plt.show()

# 3. subplot 이용 차트 그리기
fig = plt.figure(figsize = (5, 3)) #차트 크기
x1 = fig.add_subplot(2,2,1)
x2 = fig.add_subplot(2,2,2)
x3 = fig.add_subplot(2,2,3)
x4 = fig.add_subplot(2,2,4)
#-> 2x2 공간으로 나누고 왼쪽 위부터 1, 오른쪽 위 2 순으로 차트를 배치함.

# data 생성
data3 = np.random.randint(1, 100, 100) #1에서 99사이의 100개의 데이터를 생성하겠다는 뜻.
data4 = np.random.randint(10, 110, 100)
cdata = np.random.randint(1, 4, 100) #색상 지정하는 데 쓰임.

# 첫번째 격자 : 히스토그램
x1.hist(data3)

# 두번째 격자 : 산점도
x2.scatter(data3, data4, c=cdata)

# 세번째 격자 : 선 그래프
x3.plot(data3)

# 네번째 격자 : 점선 그래프
x4.plot(data4, 'g--')
plt.show()

# 4. 차트 크기 지정, 두 개 이상 차트 그리기
fig = plt.figure(figsize=(12,5))

chart = fig.add_subplot() # (1,1,1)

# 계단형 차트
chart.plot(data, color='r',
           label='step', drawstyle='steps-post')

# 선 스타일 차트
chart.plot(data2, color='b', label='line')

# 차트 제목
plt.title('계단형 vs 선 스타일')
plt.xlabel('데이터')
plt.ylabel('난수 정수')
plt.legend(loc='best') #가장 비어있는 자리에 범례를 넣어줌.
plt.show()

# 설정에서 그래프 inline를 automatic으로 바꾸면 큰 창으로 볼 수 있다.

dir(chart)
'''
'bar',
'barh',
'boxplot',
'hist',
'pie',
'scatter'
'''

# 5. 막대차트
fig2 = plt.figure()

chart2 = fig2.add_subplot() #(1, 1, 1)

data = [127, 90, 202, 150, 250]
idx = range(len(data)) #0~4 축눈금
chart2.bar(idx, data, color='darkblue')

# x축 눈금 레이블
x_label = ['서울','대전','부산','광주','인천']
plt.xticks(idx, x_label)
plt.xlabel('판매 지역')
plt.ylabel('지역별 매출현황')
plt.title('2020년 1분기 전국 지역별 판매현황')
plt.show()






















