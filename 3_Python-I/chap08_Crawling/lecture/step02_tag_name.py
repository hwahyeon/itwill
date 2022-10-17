'''
tag명으로 찾기
형식)
html.find('tag') : 최초로 발견된 tag 수집
html.find_all('tag') : 해당 tag 전체 수집
'''

from bs4 import BeautifulSoup #html 파싱


# 1. local file 불러오기
import os
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap08_Crawling')
file = open("./data/html01.html", mode = 'r', encoding='utf-8') #파일을 읽어볼때 utf-8로 읽어옴
src = file.read()
print(src)

# 2. src -> html 파싱
html = BeautifulSoup(src, 'html.parser')
print(html)

# 3. tag 찾기 -> 내용 추출

# 1) tag 계층적 구조
h1 = html.html.body.h1 # <h1>
print(h1) # element : <h1> 시멘틱 태그 ?</h1>
print(h1.string) # 내용 : <h1> 시멘틱 태그 ?</h1>

# 2) find('tag')
h2 = html.find('h2') # <h2> 주요 시멘틱 태그 </h2>
print(h2) #<h2> 주요 시멘틱 태그 </h2>
print(h2.string) #주요 시멘틱 태그  string을 이용해서 내용만 꺼내올 수 있다.

# 3) find_all('tag') 해당 태그 전체를 가져옴.
# python에선 원소가 두 개 이상이면 list나 tuple로 반환함 여기서는 list로 반환함.

html.find_all()
lis = html.find_all('li')
print(lis) # list
# [<li> header : 문서의 머리말(사이트 소개, 제목, 로그 )</li>, <li> nav : 네이게이션(메뉴) </li>, <li> section : 웹 문서를 장(chapter)으로 볼 때 절을 구분하는 태그</li>, <li> aside : 문서의 보조 내용(광고, 즐겨찾기, 링크) </li>, <li> footer : 문서의 꼬리말(작성자, 저작권, 개인정보보호) </li>]
print(len(lis)) # 5 li가 5개라는 뜻

for li in lis:
    print(li.string) #내용만 출력하기

li_cont = [li.string for li in lis]
print(li_cont) # list




