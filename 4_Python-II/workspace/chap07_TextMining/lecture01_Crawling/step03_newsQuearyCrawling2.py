# -*- coding: utf-8 -*-
"""
 방법2) url query 이용 : 년도별 뉴스 자료 수집
        ex) 2015.01.01 ~ 2020.01.01
            page : 1 ~ 5
"""

import urllib.request as req # url 요청
from bs4 import BeautifulSoup # html 피싱
import pandas as pd # 시계열 date

# 1. 수집 년도 생성 : 시계열 date 이용
date = pd.date_range("2015-01-01", "2016-01-01")
len(date) #1827 (5 * 365) = 1825
date[0] #Timestamp('2015-01-01 00:00:00', freq='D')
date[-1] #Timestamp('2020-01-01 00:00:00', freq='D')

import re #sub('pattern', '', string)

#'2015-01-01 00:00:00' -> 20150101
sdate = [re.sub('-', '',str(d))[:8] for d in date]
sdate[:10]
sdate[-10:]


# 2. Crawler 함수
def newsCrwaler(date, pages=5):
    one_day_date = []
    for page in range(1, pages+1): # 1 ~ 5
        url = f"http://media.daum.net/newsbox?regDate={date}&page={page}"
        
        try:
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
            cnt=0
            for link in links:
                link_str = str(link.string) # 내용 추출 후 자료를 문자타입으로 변경
                cnt += 1
                print('cnt :', cnt)
                print('news :', link_str)
                one_page_data.append(link_str.strip()) # 문자 끝의 불용어(\n, 공백 등) 처리
                
            one_day_date.extend(one_page_data[:40])
        except Exception as e:      
            print('오류 발생 :',e)

     return one_page_data[:40]


# 3. Crawler 함수 호출
[newsCrwaler(date) for date in sdate] #이렇게 하면 중첩리스트 형태가 된다.
# [ [1day(1~5page)], [2day(1~5page)], ... , [29day(1~5page)]]   

year5_news_date = [newsCrwaler(date)[0] for date in sdate]
len(year5_news_date)
# [1page(40), 2page(40), ......]



'''
 방법2) url query 이용 : 년도별 뉴스 자료 수집 
   ex) 20150101 ~ 20200130
'''

import urllib.request as req  # url 가져오기 
from bs4 import BeautifulSoup
import pandas as pd

# 1. 수집 년도 생성 : 시계열 자료 이용 
date = pd.date_range("2019-11-01", "2020-03-30") # 5개월
print(date)
len(date) # 151

# '-' 제거 : url date parameter
import re
#'20200330 00:00:00' -> '20200330'
sdate = [re.sub('-', '', str(d))[:8] for d  in  date] # 
sdate # '20191101' ~ '20200330'

# base_url : https://news.daum.net/newsbox?regDate=20200505&page=2

# Crawler 함수(페이지, 검색날짜) 
def crawler_func(date, pages=5): # 페이지 고정 

    one_day_news = [] # 1 day news
    # page 번호(page 수 만큼 반복)
    for page in range(1, int(pages)+1) : # pg= 1 ~ 5
        # 1) 최종 url 구성
        url = f"https://news.daum.net/newsbox?regDate={date}&page={page}"

        # 2) url 요청 -> html source          
        res = req.urlopen(url)
        data = res.read()
        
        # 3) html 파싱
        src = data.decode('utf-8') # charset='euc-kr'
        html = BeautifulSoup(src, 'html.parser')
        
        # 4) tag 기반 crawling : 태그[속성='값'] > 태그[속성='값'] 
        links = html.select('a[class=link_txt]') # element 수집
        
        one_page_news = [] # 빈 list

        for a in links :
            content = str(a.string).strip() 
            one_page_news.append(content) # 1 page news 저장  
        
        print('date : ', date)
        one_day_news.extend(one_page_news[:40]) # 1 day news 저장 
        print(one_day_news)
    return one_day_news
    

# 클로러 함수 호출 
crawling_news = [crawler_func(date) for date in sdate]
len(crawling_news) # 151
5 * 30 # 150
crawling_news # [ [1day], [2day], ....[151day]]

crawling_news[0] # [1day]
len(crawling_news[0]) # 200 = 5 * 40












