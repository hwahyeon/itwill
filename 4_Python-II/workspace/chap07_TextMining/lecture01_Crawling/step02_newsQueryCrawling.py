# -*- coding: utf-8 -*-
"""
news Crawling : url query 이용
  url : http://media.daum.net -> [배열이력]
  url : http://media.daum.net/newsbox -> base url [특정 날짜]
  url : https://news.daum.net/newsbox?regDate=20200513 -> [특정 페이지]
  url : https://news.daum.net/newsbox?regDate=20200513&page=2
"""

import urllib.request as req  # url 요청
from bs4 import BeautifulSoup # html 파싱

# 1. url query 만들기
'''
date : 2020.2.1.~2020.2.29.
page : 1 ~ 10 pages
'''

base_url = "http://media.daum.net/newsbox?regDate="
date = list(range(20200201, 20200230))
date
len(date) #29

url_list = [base_url+str(d) for d in date] #문자+문자로 해야하기 때문에 d의 타입을 변경
url_list

# base url + date + page

page = list(range(1, 11)) # 1 ~ 10
page

pages = ['&page='+str(p) for p in page] #&page=1 ~ &page=10
pages

final_url = []

for url in url_list : # base url + date
    for page in pages : # page1 ~ page10
        final_url.append(url + page)
    
len(final_url) # 290 = 29 * 10
final_url[0]
# 'http://media.daum.net/newsbox?regDate=20200201&page=1'
final_url[-1]
# 'http://media.daum.net/newsbox?regDate=20200229&page=10'

### Crawlier 함수 정의 ###
def Crawlier(url) :
    # 1. url 요청
    res = req.urlopen(url)
    src = res.read() # source
    
    # 2. html 파싱
    src = src.decode('utf-8')
    html = BeautifulSoup(src, 'html.parser')
    
    # 3. tag[속성='값'] -> 'a[class="link_txt"]'
    links = html.select('a[class="link_txt"]') 
    one_page_data = [] # 빈 list
    
    cnt=0
    for link in links:
        link_str = str(link.string) # 내용 추출 후 자료를 문자타입으로 변경
        cnt += 1
        print('cnt :', cnt)
        print('news :', link_str)
        one_page_data.append(link_str.strip()) # 문자 끝의 불용어(\n, 공백 등) 처리
        
    return one_page_data


##############################

one_page_data = Crwalier(final_url[0])
len(one_page_data) #134
type(one_page_data) #list

one_page_data[0]
one_page_data[:10]
one_page_data[-1]

# 특정한 날짜의 뉴스를 크롤링하는 것이지만, 그렇지 않은 최신의 뉴스도 가져오게 되기에 이를 추려야함.
one_page_data[:40]


# 2월(1개월) 전체 news 수집
month_news = []
page_cnt = 0

# 3. Crawlier 함수 호출
for url in final_url:
    page_cnt += 1
    print('page :', page_cnt)
    
    try : # 예외처리 - url이 없는 경우
        one_page_news = Crwalier(url) # 1page news
        print('one page news')
        print(one_page_news)
        
        #month_news.append(one_page_news) #[[1page],[2page]...]
        month_news.extend(one_page_news) #[1page ~ 290page]
    except:
        print('해당 url 없음:', url)


len(final_url) #290
len(month_news) #11600          37955???
29*10*40

# 4. binary file save
import pickle # list -> file save -> load(list)

# file save
file = open('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/news_data.pck',
            mode = 'wb')
pickle.dump(month_news, file)
file.close()

# file load
file = open('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/news_data.pck',
            mode = 'rb')
month_news2 = pickle.load(file)
month_news2 # news 확인
file.close()







