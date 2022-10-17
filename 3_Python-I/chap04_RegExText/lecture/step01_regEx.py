'''
정규 표현식

[주요 메타문자]
. : 임의의 한 문자
.x : 임의의 한 문자 뒤에 x가 오는 문자열(ex : abc, mbc -> .bc)
^x : x로 시작하는 문자열(접두어 추출)
x$ : x로 끝나는 문자열(접미어 추출)
x. : x 다음에 임의의 한 문자가 오는 문자열(ex : t1, t2, ta -> t.)
x* : x가 0번 이상 반복
x+ : x가 1개 이상 반복
x? : x가 0 또는 1개 존재
x{m, n} : x가 m~n 사이 연속
x{m, } : x가 m 이상 연속
x{,n} : x가 n 이하 연속
[x] : x문자 한 개 일치
'''

st1 = '1234 abc홍길동 ABC_555_6 이사도시'
st2 = 'test1abcABC 123mbc 45test'
st3 = 'test^홍길동 abc 대한*민국 123$tbc'

#라이브러리 규모가 크면 패키지(폴더)로 제공하고 그밖엔 모듈(.py파일)로 제공한다.

import re # 방법1) 정규표현식 모듈
from re import findall, match, sub # 방법2) 함수들을 불러옴. 형식) from 모듈 import 모듈에 소속되어 있는 functions
'''
re.findall() # 방법1)
findall() # 방법2)
'''

# 방법1)로 import했으면 re.findall() 로 써야하고,
# 방법2)로 import했으면 findall() 로 사용한다.
# 만약 모듈 안에 함수가 엄청 많다면 방법1)은 비효율적이다.

# 1. findall
# 형식) findall(pattern='메타문자 패턴', string='문자열')

# 1) 숫자 찾기
print(re.findall('1234', st1)) #['1234'] - list반환
print(findall('[0-9]{3}', st1)) #['123', '555'] // 숫자가 연달아 3개 오는 것을 찾는다는 의미.
print(findall('[0-9]{3,}', st1)) #['1234', '555'] // 3개 이상
print(findall('\\d{3,}',st1)) #['1234', '555']

# 2) 문자열 찾기
print(findall('[가-힣]{3,}', st1)) #['홍길동', '이사도시']
print(findall('[a-z]{3,}', st1)) #['abc']
print(findall('[a-z|A-Z]{3}', st1)) #['abc', 'ABC']

str_list = st1.split(sep=' ')
print(str_list) #['1234', 'abc홍길동', 'ABC_555_6', '이사도시']

names = []
for s in str_list:
    tmp = findall('[가-힣]{3,}',s)
    print(tmp) # [], ['홍길동'], [], ['이사도시'] 일치된 것이 없으면 []빈리스트를 반환함.

    if tmp : #[] -> False, ['홍길동'] -> True
        # names.append(tmp) 이렇게 하면 [['홍길동']] 중첩리스트 형식으로 append됨.
        names.append(tmp[0])

print(names) # ['홍길동', '이사도시']

# 3) 접두어/접미어 문자열 찾기 (문장의 시작과 끝을 기준으로 함)
st2 = 'test1abcABC 123mbc 45test'
print(findall('^test',st2)) #['test'] ^로 시작하는 것을 가져오는 것
print(findall('st$',st2)) #['st'] 마지막 문자가 st로 끝나는 것을 가져오는 것.

# 종료 문자 찾기
print(findall('.bc',st2)) #['abc', 'mbc']

# 시작 문자 찾기
print(findall('t.',st2)) #['te', 't1', 'te']
print(findall('t..',st2)) #['tes', 't1a', 'tes']

# 4) 단어 찾기(\\w) : 한글, 영문자, 숫자 (특수문자는 제외)
st3 = 'test^홍길동 abc 대한*민국 123$tbc'

words = findall('\\w{3,}',st3) #3자 이상인 단어들을 찾겠다는 의미. list로 반환
print(words) #['test', '홍길동', 'abc', '123', 'tbc']

# 5. 특정 문자열 제외
print(findall('[^t]+',st3)) # +는 1개 이상 반복할 때
# ['es', '^홍길동 abc 대한*민국 123$', 'bc']
print(findall('[^t]',st3)) #+를 제외하면 문자가 한 개씩 반환됨.
#['e', 's', '^', '홍', '길', '동', ' ', 'a', 'b', 'c', ' ', '대', '한', '*', '민', '국', ' ', '1', '2', '3', '$', 'b', 'c']

# 특수문자 제외
print(findall('[^^*$]+',st3)) #첫번째^는 명령어(접두어의 ^) 두번째^는 제거할 문자, 둘이 같은 것이 아님.
# [^^*$] = [\^*$]
# ['test', '홍길동 abc 대한', '민국 123', 'tbc']

# 2. match
# match(pattern = '패턴', string = '문자열')
# - 패턴 일치 여부 반환(패턴과 일치하면 object 반환, 불일치하면 NULL반환 (findall에서 불일치하면 빈리스트를 반환))
# 결과를 반환받으려면 findall, 맞는지 아닌지 확인하려면 match

jumin = "123456-1234567"
jumin
jumin2 = "1234125312"
result = match("[0-9]{6}-[1-4]\\d{6}", jumin2)
print(result)

if result: # object
    print("정상 주민번호")
else: # None(NULL)
    print("비정상 주민번호")

# 3.sub('pattern', 'rep', 'string')
st3 = 'test^홍길동 abc 대한*민국 123$tbc'

result = sub('[\^*$]', '', st3)
# \의미 : 메타문자가 아닌 일반 특수문자
print(result) # test홍길동 abc 대한민국 123tbc


