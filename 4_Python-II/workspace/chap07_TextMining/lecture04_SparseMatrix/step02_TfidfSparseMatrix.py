# -*- coding: utf-8 -*-
"""
1. csv file read
2. texts, target -> 전처리
3. max features
4. Sparse matrix
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. csv file read
spam_data = pd.read_csv('C:/ITWILL/4_Python-II/workspace/chap07_TextMining/data/temp_spam_data.csv',
                        encoding='utf-8', header = None)
spam_data
'''
      0                        1
0   ham    우리나라    대한민국, 우리나라 만세
1  spam      비아그라 500GRAM 정력 최고!
2   ham               나는 대한민국 사람
3  spam  보험료 15000원에 평생 보장 마감 임박
4   ham                   나는 홍길동
'''

target = spam_data[0]
texts = spam_data[1]

target # y변수로 사용할 변수
texts # sparse matrix -> x변수

# 2. texts, target -> 전처리

# 1) target 전처리 -> dummy 변수
target = [1 if t == 'spam' else 0 for t in target]
target # [0, 1, 0, 1, 0]

# 2) texts 전처리

import string # text 전처리 

def text_prepro(texts): # 문단(input) -> 문장(string) 단위로 쪼갬 -> 음절 단위로 쪼갬 > 문장
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return texts

texts_re = text_prepro(texts)
texts_re
'''
['우리나라 대한민국 우리나라 만세',
 '비아그라 gram 정력 최고',
 '나는 대한민국 사람',
 '보험료 원에 평생 보장 마감 임박',
 '나는 홍길동']
'''

# 3. max features
'''
사용할 x변수의 개수(열의 차수)
'''
tfidf = TfidfVectorizer()
tfidf_fit = tfidf.fit(texts_re) #문장 -> 단어 생성

#아래처럼 한 문장으로 할 수도 있다.
tfidf_fit = TfidfVectorizer().fit(texts_re)
vocs = tfidf_fit.vocabulary_
vocs
max_features = len(vocs) # 16
'''
만약 max_features = 10라고 하면, 전체 16개 단어 중에서 10개의 단어만 이용하겠다는 뜻
sparse matrix = [5 x 10]
'''
max_features = 10



# 4. Sparse matrix
sparse_mat = TfidfVectorizer().fit_transform(texts_re)
sparse_mat #<5x16 sparse matrix of type '<class 'numpy.float64'>'

# max_features 적용 예
sparse_mat = TfidfVectorizer(max_features = max_features).fit_transform(texts_re)
sparse_mat #<5x10 sparse matrix of type '<class 'numpy.float64'>'


# scipy -> numpy
sparse_mat_arr = sparse_mat.toarray()
sparse_mat_arr.shape #(5, 16)
sparse_mat_arr
'''
array([[0.        , 0.        , 0.33939315, 0.        , 0.42066906,
        0.        , 0.        , 0.        , 0.        , 0.84133812,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.5       , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.5       , 0.        , 0.        ,
        0.        , 0.        , 0.5       , 0.5       , 0.        ,
        0.        ],
       [0.        , 0.53177225, 0.53177225, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.659118  , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.40824829, 0.        ,
        0.40824829, 0.40824829, 0.        , 0.        , 0.        ,
        0.40824829, 0.40824829, 0.        , 0.        , 0.40824829,
        0.        ],
       [0.        , 0.62791376, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.77828292]])
'''


































