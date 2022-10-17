# -*- coding: utf-8 -*-
'''
# 문) 2019년11월 ~ 2020년2월 까지(4개월) daum 뉴스기사를 다음과 같이 크롤링하고, 단어구름으로 시각화 하시오.
# <조건1> 날짜별 5개 페이지 크롤링
# <조건2> 불용어 전처리 
# <조건3> 단어빈도수 분석 후 top 20 단어 단어구름 시각화 
'''

import urllib.request as req  # url 가져오기 
from bs4 import BeautifulSoup
import pandas as pd # 시계열 date
import re

# 수집 년도 생성 : 시계열 date 이용
date = pd.date_range("2019-11-01", "2020-02-29")
len(date) #1827 (5 * 365) = 1825

sdate = [re.sub('-', '',str(d))[:8] for d in date]
sdate[:10]

#<조건1> 날짜별 5개 페이지 크롤링
# 클로러 함수(페이지, 검색날자) 
def newsCrawler(date,  pages=5) : # 1 day news 
    
    one_day_date = []
    for page in range(1, pages+1) : # 1 ~ 5
        url = f"https://news.daum.net/newsbox?regDate={date}&page={page}"
        
        try : 
            # 1. url 요청
            res = req.urlopen(url)
            src = res.read() # source
               
            # 2. html 파싱
            src = src.decode('utf-8')
            html = BeautifulSoup(src, 'html.parser')
                
            # 3. tag[속성='값'] -> 'a[class="link_txt"]'
            links = html.select('a[class="link_txt"]')   
            
            one_page_data = [] # 빈 list
            
            print('date :', date)
            for link in links :
                link_str = str(link.string) # 내용 추출                
                one_page_data.append(link_str.strip()) # 1page news
            
            # 1day news 
            one_day_date.extend(one_page_data[:40])
        except Exception as e :
            print('오류 발생 : ', e)
    return one_day_date # list

# Crawler 함수 호출
year_news_date = [newsCrawler(date)[0] for date in sdate]
len(year_news_date) # 121

year_news_date[0]

# <조건3> 단어빈도수 분석 후 top 20 단어 단어구름 시각화 
from konlpy.tag import Kkma # class
from wordcloud import WordCloud #class
from re import match
kkma = Kkma()

# 2. 명사 추출 : Kkma
nouns_word = [] # 명사 저장

for sent in year_news_date: 
    for noun in kkma.nouns(sent): # 문장 -> 명사
        nouns_word.append(noun)

nouns_word
len(nouns_word) # 1380
nouns_word[0]

# 전처리 + 단어 카운트

nouns_count = {} # 단어 카운트

for noun in nouns_word:
    if len(noun) > 1 and not (match('^[0-9]', noun)) :
        # key[noun] = value[출현빈도수]
        nouns_count[noun] = nouns_count.get(noun, 0) + 1

nouns_count
len(nouns_count) #757

# 4. WordCloud

# top20 word
from collections import Counter # class

word_count = Counter(nouns_count) #dict

top20_word = word_count.most_common(20)
top20_word

# word cloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
          width=800, height=600,
          max_words=100, max_font_size=200,
          background_color='white')

wc_result = wc.generate_from_frequencies(dict(top20_word))

import matplotlib.pyplot as plt
plt.figure(figsize = (12,8))
plt.imshow(wc_result)
plt.axis('off') # x, y 축 눈금 감추기
plt.show()

























