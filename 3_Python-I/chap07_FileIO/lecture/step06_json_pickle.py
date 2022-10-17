'''
json 파일 2가지 형식

1. 중괄호 : {key:value, ...}, {key:value, ...}
    -> json.loads(row)
2. 대괄호 : [{key:value, ...}, {key:value, ...}]
    -> json.load(row)
# loads와 load의 차이
'''
import os
import json
import pandas as pd

# 1번 형식 : 중괄호 : {key:value, ...}, {key:value, ...}

# 2번 형식 : 대괄호 : [{key:value, ...}, {key:value, ...}]
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap07_FileIO')

file = open("./data/labels.json", mode = 'r', encoding='utf-8')
#print(file.read()) #[{row}, {row}, ...] 주석처리해야함 왜냐하면 read를 실행하면서 다 자료를 소진하기 때문에
# 주석처리를 안하고 실행하면 다음 코드가 실행이 안된다.

rows = json.load(file) # json 문자열 -> object
print(len(rows)) # 30
print(rows)
print(type(rows)) #<class 'list'>

for row in rows[:5]: #5개만 보자는 뜻
    print(row)
    print(type(row))

file.close()

#list -> DataFrame
rows_df = pd.DataFrame(rows)
print(rows_df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 5 columns):
'''
print(rows_df.head())

# 2. pickle
'''
python object(list, dict) -> file(binary) -> python object(list, dict)
'''
import pickle
'''
save : pickle.dump(data, file)
load : pickle.load(file)
'''

# 1) pickle save
file = open("./data/rows_data.pickle", mode='wb') #wb : write binary 기계어 형식으로 저장하겠다는 뜻
pickle.dump(rows, file) #list -> binary
print('pickle file saved')

# file reload

# 2) pickle load
file = open("./data/rows_data.pickle", mode='rb')
rows_data = pickle.load(file)
print(rows_data)
print(type(rows_df)) #저장할 때는 list였는데 다시 복원했을 때도 list인지 확인하는 과정



