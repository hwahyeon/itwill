'''
원격 서버의 웹문서 수집
'''

from bs4 import BeautifulSoup # source -> html문서로 변환시켜줌 (파싱)
import urllib.request as res # 별칭 : 원격 서버 파일 요청

url = "http://www.naver.com/index.html"

# 1. 원격 서버 url 요청
req = res.urlopen(url) # 요청 -> 응답
print(req) #object info
data = req.read() # source
print(data) # source 형태로 출력한 것.

# 2. source(문자열) -> html 형식 : html 파싱
src = data.decode('utf-8') # 디코딩 -> 소스 생성
html = BeautifulSoup(src, 'html.parser') # 생성자 # source -> html
print(html)

link = html.find('li' > 'board_list_type_textarea board_list_type_ext board_list_eid_ext_recipe')
print(link)




# 3. Tag 내용 가져오기
link = html.find('li class') # <a href = 'url'>내용</a>
print(link)


links = html.select('board_list_type_textarea board_list_type_ext board_list_eid_ext_recipe > dd')
print(links)
