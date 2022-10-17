# -*- coding: utf-8 -*-
"""
1. text file 읽기
2. 문장에서 명사 추출 : Kkma(class)
3. 전처리 : 단어 길이, 숫자 제외
4. WordCloud
"""

from konlpy.tag import Kkma # class
from wordcloud import WordCloud #class
help(WordCloud)

# object 생성
kkma = Kkma()

# 1. text file 읽기 : text_data.txt
file = open('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/text_data.txt',
            mode='r', encoding='utf-8')
docs = file.read()
docs

# docs -> sentence
ex_sent = kkma.sentences(docs) #list
ex_sent
'''
['형태소 분석을 시작합니다.',
 '나는 데이터 분석을 좋아합니다.',
 '직업은 데이터 분석 전문가 입니다.',
 'Text mining 기법은 2000대 초반에 개발된 기술이다.']
'''
len(ex_sent) # 4

# docs -> nouns
ex_nouns = kkma.nouns(docs) # 유일한 명사 추출
ex_nouns
len(ex_nouns) # 13

# 2. 명사 추출 : Kkma
nouns_word = [] # 명사 저장

for sent in ex_sent: #'형태소 분석을 시작합니다.'
    for noun in kkma.nouns(sent): # 문장 -> 명사
        nouns_word.append(noun)

nouns_word
len(nouns_word) # 15

# 3. 전처리 + 단어 카운트 : 단어 길이 제한(1음절), 숫자 제외
from re import match

nouns_count = {} # 단어 카운트

for noun in nouns_word:
    if len(noun) > 1 and not (match('^[0-9]', noun)) :
        # key[noun] = value[출현빈도수]
        nouns_count[noun] = nouns_count.get(noun, 0) + 1

nouns_count
len(nouns_count) #9

# 4. WordCloud

# 1) top5 word
from collections import Counter # class

word_count = Counter(nouns_count) #dict

top5_word = word_count.most_common(5)
top5_word
# [('분석', 3), ('데이터', 2), ('형태소', 1), ('직업', 1), ('전문가', 1)]

# 2) word cloud
help(WordCloud)
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf',
          width=500, height=400,
          max_words=100, max_font_size=150,
          background_color='white')
# 한글font를 설정 안해주면 한글이 깨진다.

wc_result = wc.generate_from_frequencies(dict(top5_word))

import matplotlib.pyplot as plt

plt.imshow(wc_result)
plt.axis('off') # x, y 축 눈금 감추기
plt.show()




















