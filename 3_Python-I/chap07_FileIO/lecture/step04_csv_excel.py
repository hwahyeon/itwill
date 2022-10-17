'''
csv, excel file read/write
    - 칼럼 단위로 작성된 파일 유형

cmd에서 외부 라이브러리 설치
 pip install 패키지명

'''

import pandas as pd # as 별칭
import os
print(os.getcwd()) # C:\ITWILL\3_Python-I\workspace\chap07_FileIO\exams
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap07_FileIO')

# 1. csv file read
spam_data = pd.read_csv("../data/spam_data.csv",header=None,encoding='ms949')
print(spam_data.info()) # str()
'''
<class 'pandas.core.frame.DataFrame'> R에서의 Dataframe과 같다. spark에서도 나옴. (행렬구조도 되어 있는 자료구조를 말함)
RangeIndex: 5 entries, 0 to 4 : 관측치의 개수가 5개
Data columns (total 2 columns)
칼럼 이름 명(header)가 None이면, python에서는 0부터 자동으로 컬럼명이 들어감. R에서는 v1, v2...식으로 들어감 
'''
print(spam_data)

# 2. x,y 변수 선택
target = spam_data[0] # DF[칼럼명] 첫번째 칼럼
texts = spam_data[1] # DF[칼럼명] 두번째 칼럼
print(target) # index data
print(texts)

# 3. target -> dummy
target = [1 if x == 'spam' else 0 for x in target]
print(target) # [0, 1, 0, 1, 0]

# 4. text 전처리
def clean_text(texts) :
    from re import sub  # gsub() 유사함
    # 1. 소문자 변경
    texts_re = texts.lower() # 문장 1개 소문자 변경
    # 2. 숫자 제거
    texts_re2 = sub('[0-9]', '', texts_re)
    # 3. 문장부호 제거
    punc_str = '[.,;:?!]'
    texts_re3 = sub(punc_str, '', texts_re2)
    # 4. 특수문자 제거
    spec_str = '[@#$%^&*()]'
    texts_re4 = sub(spec_str, '', texts_re3)
    # 5. 영문자 제거
    texts_re5 = sub('[a-z]', '', texts_re4)
    # 6. 공백 제거
    texts_re6 = ' '.join(texts_re5.split())

    return texts_re6

clean_texts = [clean_text(text) for text in texts]
print('텍스트 전처리 후')
print(clean_texts)

######################
## bmi.csv
######################
bmi = pd.read_csv("../data/bmi.csv", encoding='utf-8')
print(bmi.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20000 entries, 0 to 19999
Data columns (total 3 columns):
'''
print(bmi.head()) # 앞부분 5개
print(bmi.tail()) # 뒷부분 5개
# 큰 데이터이니 앞 뒤의 내용을 조금이나마 살펴보기 위함이다.

height = bmi['height'] # DF['칼럼명']
weight = bmi['weight']
label = bmi.label # DF.칼러명

print(len(height)) # 20000
print(len(label)) # 20000

print('키 평균 :', height.mean())
print('몸무게 평균 :', weight.mean())
'''
키 평균 : 164.9379
몸무게 평균 : 62.40995
'''

# max()/min()
max_h = max(height)
max_w = max(weight)
print('max h =', max_h) # max h = 190
print('max w =', max_w) # max w = 85

print("정규화")
height_nor = height / max_h #height는 벡터이고 max_h는 스칼라/ pandas에서는 이 둘의 연산이 가능하다.
weight_nor = weight / max_w

print(height_nor.mean()) # 0.8680942105263159
print(weight_nor.mean()) # 0.734234705882353

# 범주형 변수 : label
lab_cnt = label.value_counts() # 빈도수 / R의 table과 비슷한 기능.
print(lab_cnt)
'''
normal    7677
fat       7425
thin      4898
'''

# 2. excel file read
'''
pip install xlrd
'''
excel = pd.ExcelFile("../data/sam_kospi.xlsx")
print(excel) # object info

kospi = excel.parse('sam_kospi')
print(kospi) # 내용
print(kospi.info()) # 정보
'''
RangeIndex: 247 entries, 0 to 246
Data columns (total 6 columns):
엑셀도 행렬구조인 Dataframe 자료구조로 만들어짐.
'''

# 3. csv file save
kospi['Diff'] = kospi.High - kospi.Low  # 파생변수
print(kospi.info()) # Data columns (total 7 columns):

# csv file 저장 : 행 번호 제외
kospi.to_csv("../data/kospi_df.csv", index=None, encoding='utf-8')

kospi_df = pd.read_csv("../data/kospi_df.csv", encoding='utf-8')
print(kospi_df.head())








