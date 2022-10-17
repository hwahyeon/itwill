'''
1. news Crwaling
  url : http://media.daum.net
2. pickle save
  binary file save
'''

import urllib.request as req  # url 요청
from bs4 import BeautifulSoup # html 파싱

url = 'http://media.daum.net'

# 1. url 요청
res = req.urlopen(url)
src = res.read() # source
print(src)

# 2. html 파싱
src = src.decode('utf-8')
html = BeautifulSoup(src, 'html.parser')
print(html)

# 3. tag[속성='값'] -> 'a[class="link_txt"]'
links = html.select('a[class="link_txt"]')
print(len(links))
print(links)

crawling_data = [] # 빈 list

for link in links:
    link_str = str(link.string) # 내용 추출 후 자료를 문자타입으로 변경
    print(link_str)
    crawling_data.append(link_str.strip()) # 문자 끝의 불용어(\n, 공백 등) 처리

print(crawling_data) #list
print(len(crawling_data)) #62

# 4. pickle file save
import pickle

# save
file = open("./data/new_crawling.pickle", mode='wb')
pickle.dump(crawling_data, file)
print('pickle file saved')










