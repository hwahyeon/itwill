'''
 문) 아래 url을 이용하여 어린이날(20200505)에 제공된 뉴스 기사를 
   1~5페이지 크롤링하는 크롤러 함수를 정의하고 크롤링 결과를 확인하시오.
   
   base_url = "https://news.daum.net/newsbox?regDate="   
   
   <조건1> 크롤러 함수의 파라미터(page번호, 날짜)
   <조건2> 크롤링 대상  : <a> 태그의 'class=link_txt' 속성을 갖는 내용 
   <조건3> 크롤링 결과 확인  : news 갯수와  문장 출력(한글 깨짐 확인)  
'''

import urllib.request as req  # url 가져오기 
from bs4 import BeautifulSoup

base_url = "https://news.daum.net/newsbox?regDate="

# 클로러 함수(페이지, 검색날자) 
def crawler_func(pages, date):
    
    res_url = base_url+str(date)
    crawling_news = []
    for p in range(1, pages+1):
        res_url2 = res_url+'&page='+str(p)
        crawling_news.append(res_url2)
    
    print(crawling_news)

    one_page_data = []
    
    for url in crawling_news:
        # 1. url 요청
        res = req.urlopen(url)
        src = res.read() # source
        
        # 2. html 파싱
        src = src.decode('utf-8')
        html = BeautifulSoup(src, 'html.parser')
        
        # 3. tag[속성='값'] -> 'a[class="link_txt"]'
        links = html.select('a[class="link_txt"]') 
    
        for link in links:
            link_str = str(link.string)
            one_page_data.append(link_str.strip())
    
    return one_page_data


len(crawler_func(5, 20200505))
len(crawler_func(7, 20200506))
    


# 클로러 함수 호출 
crawler_func(5, '20200505')

# 크롤링 결과 확인
print('크롤링 news 수 =', len(crawling_news)) 
print(crawling_news)
    




# teacher
import urllib.request as req  # url 가져오기 
from bs4 import BeautifulSoup

base_url = "https://news.daum.net/newsbox?regDate="

# 클로러 함수(페이지, 검색날자) 
def crawler_func(pages, date):
    crawling_news = [] # news 저장 
    url = base_url + date
    # "https://news.daum.net/newsbox?regDate=20200505"
    
    for page in range(1, pages+1) : # page = 1~5
        final_url = url + '&page='+str(page)
        # "https://news.daum.net/newsbox?regDate=20200505&page=1"
        print(final_url)
        
        # 1. url 요청
        res = req.urlopen(final_url)
        src = res.read() # source
           
        # 2. html 파싱
        src = src.decode('utf-8')
        html = BeautifulSoup(src, 'html.parser')
            
        # 3. tag[속성='값'] -> 'a[class="link_txt"]'
        links = html.select('a[class="link_txt"]')   
        
        one_page_data = [] # 빈 list
        
        for link in links :
            link_str = str(link.string) # 내용 추출
            one_page_data.append(link_str.strip()) # 문장 끝 불용어 처리
            
        crawling_news.extend(one_page_data[:40]) # 40개 선택 
        
    return crawling_news # 1~5 page news


# 클로러 함수 호출 
crawling_news = crawler_func(5, '20200505')

# 크롤링 결과 확인
print('크롤링 news 수 =', len(crawling_news)) # 크롤링 news 수 = 200
print(crawling_news)
type(crawling_news)