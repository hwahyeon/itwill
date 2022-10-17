# -*- coding: utf-8 -*-
"""
1. pickle file 읽기 : news crawling data
2. 명사 추출 : Kkma(class)
3. 전처리 : 단어 길이, 숫자 제외
4. WordCloud
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

# docs -> sentence 이 부분은 생략 가능. news_data를 직접 사용해도 된다.
ex_sent = [kkma.sentences(sent)[0] for sent in news_data] #list
ex_sent
len(ex_sent) # 11600


# 2. 명사 추출 : Kkma
nouns_word = [] # 명사 저장

for sent in ex_sent: #'형태소 분석을 시작합니다.'
    for noun in kkma.nouns(sent): # 문장 -> 명사
        nouns_word.append(noun)

nouns_word
len(nouns_word) # 120939

# 3. 전처리 + 단어 카운트 : 단어 길이 제한(1음절), 숫자 제외
from re import match

nouns_count = {} # 단어 카운트

for noun in nouns_word:
    if len(noun) > 1 and not (match('^[0-9]', noun)) :
        # key[noun] = value[출현빈도수]
        nouns_count[noun] = nouns_count.get(noun, 0) + 1

nouns_count
len(nouns_count) #12194

nouns_count['확진자'] = nouns_count['진자']
del nouns_count['진자']

# 4. WordCloud

# 1) top5 word
from collections import Counter # class

word_count = Counter(nouns_count) #dict

top50_word = word_count.most_common(50)
top50_word
# [('분석', 3), ('데이터', 2), ('형태소', 1), ('직업', 1), ('전문가', 1)]

# 2) word cloud
help(WordCloud)
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
          width=800, height=600,
          max_words=100, max_font_size=200,
          background_color='white')

wc_result = wc.generate_from_frequencies(dict(top50_word))

import matplotlib.pyplot as plt
plt.figure(figsize = (12,8))
plt.imshow(wc_result)
plt.axis('off') # x, y 축 눈금 감추기
plt.show()































