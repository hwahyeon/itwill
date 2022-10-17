# -*- coding: utf-8 -*-
"""
Konlpy : 한글 형태소 분석을 제공하는 패키지
pip install konlpy
"""

import konlpy
from konlpy.tag import Kkma # class

kkma = Kkma() # 생성자 -> object 생성

# 문단 -> 문장
para = "나는 홍길동 입니다. 나이는 23세 입니다. 대한민국 만세 입니다."
ex_sent = kkma.sentences(para)
ex_sent #list
# ['나는 홍길동 입니다.', '나이는 23세 입니다.', '대한민국 만세 입니다.']
len(ex_sent) #3


# 문장 -> 단어(명사)
ex_nouns = kkma.nouns(para)
ex_nouns
#['나', '홍길동', '나이', '23', '23세', '세', '대한', '대한민국', '민국', '만세']

# 문단 -> 품사(형태소)
ex_pos = kkma.pos(para)
ex_pos #(word, 품사 기호)
type(ex_pos) #list [(word, '품사'), (word, '품사')]

# NNG 일반 명사 NNP 고유 명사 NP 대명사
nouns = [] # 명사 저장

for word, wclass in ex_pos: #(word, 품사)
    if wclass == 'NNG' or wclass == 'NNP' or wclass == 'NP':
        nouns.append(word)


nouns # ['나', '홍길동', '나이', '대한민국', '만세']




















