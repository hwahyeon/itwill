'''
step02 관련문제
문2) wdbc_data.csv 파일을 읽어와서 단계별로 x, y 변수를 생성하시오.
     단계 1 : 파일 가져오기, 정보 확인 
     단계 2 : y변수 : diagnosis 
         x변수 : id 칼럼 제외  나머지 30개 칼럼
'''
import pandas as pd

# 단계 1 : 파일 가져오기, 정보 확인 
wdbc = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\wdbc_data.csv")
print(wdbc.info())
print(wdbc.head())

# 단계 2 : y변수, x변수 선택
cols = list(wdbc.columns)
wdbc_x = cols[2:]
wdbc_y = cols[1]   
print(wdbc_x)
print(wdbc_y)

###teacher
# 단계 1 : 파일 가져오기, 정보 확인
wdbc = pd.read_csv("C:/ITWILL/4_Python-II/data/wdbc_data.csv")
print(wdbc.info())

# 단계 2 : y변수, x변수 선택
wdbc.head()

# 칼럼명 가져오기
cols = list(wdbc.columns)
cols
len(cols) # 32

wdbc_x = cols[2:]
len(wdbc_x) # 30
wdbc_y = cols[1]
wdbc_y # 'diagnosis'

X = wdbc[wdbc_x]
y = wdbc[wdbc_y]
X.shape # (569, 30)
y.shape # (569,)