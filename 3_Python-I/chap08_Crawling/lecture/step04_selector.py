'''
선택자(selector)
    - 웹 문서 디자인(css)에서 사용
    - 선택자 : id(#) : 중복 불가, class(.) : 중복 가능
    - html.select('선택자') : 여러 개 element 수집
    - html.select_one('선택자') : 한 개의 element 수집

'''

from bs4 import BeautifulSoup

import os
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap08_Crawling')

file = open("./data/html03.html", encoding='utf-8')
src = file.read()
print(src)

html = BeautifulSoup(src, 'html.parser')

# 태그 & 선택자 -> element 수집

# 1) id 선택자 : #으로 표시
table = html.select_one("#tab") # id = 'tab'를 #tab처럼 표현한다.
print(table) # <table> ~ </table>

# <table> <tr> <th> or <td>

ths = html.select("#tab > tr > th") # list
print(ths)
print(len(ths))

for th in ths:
    print(th.string) # tag 내용


# 2) class 선택 : .으로 표시
trs = html.select("#tab > .odd") # 5<tr> -> 2<tr>
print(trs)

for tr in trs:
    #print(tr)
    tds = tr.find_all('td') # list 형식으로 반환
    for td in tds:
        print(td.string)

# 3) tag[속성 = '값'] 찾기
trs = html.select("tr[class='odd']") #계층적으로 접근하지 않고 해당하는 element의 이름, 속성 등을 통해 한번에 찾는 방법.
print(trs)

for tr in trs:
    tds = tr.find_all('td')
    for td in tds:
        print(td.string)




