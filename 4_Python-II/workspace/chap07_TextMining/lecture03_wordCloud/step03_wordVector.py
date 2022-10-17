# -*- coding: utf-8 -*-
"""
news crwaling data -> word vector
    문장 -> 단어 벡터 -> 희소행렬(Sparse matrix)
  ex) '직업은 데이터 분석가 입니다.' -> '직업 데이터 분석가'
  즉, 문장에서 단어들만 추출해서 행렬화 하는 것.
"""

from konlpy.tag import Kkma # class
from wordcloud import WordCloud #class
import pickle

# object 생성
kkma = Kkma()

# 1. pickle file 읽기 : news_data.pck
file = open('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/news_data.pck', mode='rb')
# 바이너리 코드로 읽어오는 것이기 때문에 incoding은 필요 없다.
news_data = pickle.load(file)
file.close()

news_data
type(news_data) # list
len(news_data) #11600
news_data[:5]

# 2. docs -> sentence
ex_sent = [kkma.sentences(sent)[0] for sent in news_data] #list
ex_sent
len(ex_sent) # 11600
ex_sent[0]
# '의협 " 감염병 위기 경보 상향 제안.. 환자 혐오 멈춰야"'

# 3. sentence -> word vector
from re import match

sentence_nouns = [] # 단어 벡터 저장
for sent in ex_sent :
    word_vec = ""
    for noun in kkma.nouns(sent): # 문장 -> 명사 추출
        if len(noun) > 1 and not(match('^[0-9]', noun)):
            word_vec += noun + " " # 나 " " 홍길동 " " # 명사 + 명사
    
    print(word_vec)
    sentence_nouns.append(word_vec)

len(sentence_nouns) #11600
sentence_nouns[0]
#'의협 감염병 위기 경보 상향 제안 환자 혐오 ' -> 문장 번호 1
ex_sent[0] # 원본
#'의협 " 감염병 위기 경보 상향 제안.. 환자 혐오 멈춰야"'
sentence_nouns[-1]
#'인건비 우선 협의 제안 포괄적 신속 타결 손상 ' -> 문장 번호 : 11600

# 4. file save
import pickle

file = open('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/sentence_nouns.pck', mode='wb')
#wb 바이너리 형식으로 저장겠다.
pickle.dump(sentence_nouns, file) # 변수 저장
print('file save')

file.close

# file load
file = open('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/sentence_nouns.pck', mode='rb')
word_vector = pickle.load(file)
word_vector[0]




































