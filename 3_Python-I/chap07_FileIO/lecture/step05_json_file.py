'''
JSON 파일
    - 네트워크에서 표준으로 사용되는 파일 형식
    - 파일 형식 : {key:value, key:value}, {key:value, key:value} dataframe으로 쉽게 변환이 됨
    각각의 {}은 행이 된다.
    - 활용 예 : 서로 다른 플랫폼(java, python)에서 파일 공유 목적

    - json 모듈
    1. encoding : file save : python object(list, dict) -> json file
    2. decoding : file read : json file -> python object
'''

import json

# 1. encoding : object -> 문자열 # dumps 사용
'''
python object -> json 문자열로 변경하는 과정 -> file save
형식) json.dumps(object)
'''
user = {'id' : 1234, 'name' : '홍길동'} # dict형식
print(type(user)) # <class 'dict'>

json_str = json.dumps(user, ensure_ascii=False) #ensure_ascii 아스키코드로 변경할 것인지 묻는 것
# False는 unicode -> ascii로 가는 인코딩을 안하겠다는 뜻.
print(json_str) # 딕셔너리처럼 출력되지만,
print(type(json_str)) #<class 'str'>
#이러한 것을 json문자열이라고 부름.

# 2. decoding : 문자열 -> object # loads 사용
'''
json 문자열 -> python object
형식) json.loads(json 문자열)
'''
py_obj = json.loads(json_str) #파이썬의 오브젝트를 반환한다는 뜻.
print(py_obj) #{'id': 1234, 'name': '홍길동'}
print(type(py_obj)) #<class 'dict'>

# 3. json file read/write

# 1) json file read : decoding
import os
print(os.getcwd()) #C:\ITWILL\3_Python-I\workspace

os.chdir('C:\ITWILL\\3_Python-I\workspace\chap07_FileIO')
file = open('./data/usagov_bitly.txt', mode = 'r', encoding='utf-8')
data = file.readlines() #전체 내용 -> 줄단위 읽기
file.close()
print(data)

#decoing : json 문자열 ->
rows = [json.loads(row) for row in data] #row {'key':value, ....}
print(len(rows)) #3560

for row in rows[:10]: #10개만 일단 받아보기
    print(row) #
    print(type(row)) #<class 'dict'>  -> python object로 변경된 것을 볼 수 있다.

# json object -> DataFrame
import pandas as pd
rows_df = pd.DataFrame(rows) # 행렬 자료 구조
pd.DataFrame(rows)
print(rows_df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3560 entries, 0 to 3559
Data columns (total 18 columns):
'''

print(rows_df.head())
print(rows_df.tail())


# 2) json file write : json encoding
file = open("./data/json_text.txt", mode = "w", encoding='utf-8')
print(type(rows)) # python object : <class 'list'>

for row in rows[:100]: # row = {key:value, ...} : dict
    json_str = json.dumps(row) # dict -> json 문자열
    file.write(json_str) # file save
    file.write('\n') # 줄바꿈

print('file 저장 완료 ~~~')

# 3) json file read : json decoding
file = open("./data/json_text.txt", mode = 'r', encoding ='utf-8')
data = file.readlines()
print(len(data))

# json decoding
rows = [json.loads(row) for row in data] # json 문자열 -> python object

rows_df = pd.DataFrame(rows) #가독성 좋게 행렬식으로 만들어주는 것임.
print(rows_df.info())

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 18 columns):
'''


