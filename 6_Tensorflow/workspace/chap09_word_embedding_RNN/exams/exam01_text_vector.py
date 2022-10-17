# -*- coding: utf-8 -*-
"""
문) 토큰 생성기를 이용해서 문장의 전체 단어 길이를 확인하고,
    희소행렬과 max length를 적용하여 문서단어 행렬을 만드시오. 
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # padding

sentences = [
    "Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow.",
    "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."
]

# 전체 문장 길이
print(len(sentences)) # 3


# 1. num_words 토큰 생성기 : 중요단어 50개 선정 
tokenizer = Tokenizer( num_words = 51) # 중요단어 50개 선정   


# 2. 전체 단어 개수 확인 
tokenizer.fit_on_texts(sentences) # 토큰 생성기에 문장 넣기
word_index = tokenizer.word_index 
word_index # 단어 인덱스 -> {'단어' : 고유번호}
len(word_index) # 33


# 3. sparse matrix : count, freq, tfidf
count = tokenizer.texts_to_matrix(sentences, mode='count')
count.shape # (3, 51)
count # num_words=n에 의해서 n차원 matrix 생성 

freq = tokenizer.texts_to_matrix(sentences, mode='freq')
freq # (index 1 : green -> 모든 문장 출현) 

tfidf = tokenizer.texts_to_matrix(sentences, mode='tfidf')
tfidf 


# 4. max length 지정 문장 길이 맞춤 : 두번째 단어 길이 이용 
seq_index = tokenizer.texts_to_sequences(sentences) 
seq_index
'''
[[2, 1, 10, 11, 12, 3, 4, 5, 13, 4, 14, 2, 1, 15, 16, 6, 17, 18, 19],
 [7, 20, 21, 6, 1, 8, 3, 9, 5],
 [22, 23, 24, 7, 25, 1, 8, 26, 27, 28, 29, 30, 9, 31, 32, 33]]
'''

lens = [len(seq) for seq in seq_index] # [19, 9, 16]
maxlen = 16

# 패딩(padding) : 가장 긴 단어 수 이용 문장 길이 맞춤
x_data = pad_sequences(seq_index, maxlen=maxlen)
x_data.shape # (3, 16)
x_data

