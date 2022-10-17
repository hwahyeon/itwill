# -*- coding: utf-8 -*-
"""
TFiDF 단어 생성기 : TfidfVectorizer  
  1. 단어 생성기[word tokenizer] : 문장(sentences) -> 단어(word) 생성
  2. 단어 사전[word dictionary] : (word, 고유수치)
  3. 희소행렬[sparse matrix] : 단어 출현 비율에 의해서 가중치 적용[type-TF, TFiDF]
    1] TF : 가중치 설정 - 단어 출현 빈도수
    2] TFiDF : 가중치 설정 - 단어 출현 빈도수 x 문서 출현빈도수의 역수            
    - tf-idf(d,t) = tf(d,t) x idf(t) [d(document), t(term)]
    - tf(d,t) : term frequency - 특정 단어 빈도수 
    - idf(t) : inverse document frequency - 특정 단어가 들어 있는 문서 출현빈도수의 역수
       -> TFiDF = tf(d, t) x log( n/df(t) ) : 문서 출현빈도수의 역수(n/df(t))
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# 문장
sentences = [
    "Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow.",
    "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."
]


# 1. 단어 생성기[word tokenizer]
tfidf = TfidfVectorizer() # object
tfidf

# 2. 문장 -> 단어 생성
tfidf_fit = tfidf.fit(sentences)
tfidf_fit

# 단어 사전 : 고유숫자(영문 오름차순 결정)
vocs = tfidf_fit.vocabulary_ #단어들
vocs # dict {'word1' : 거유숫자, 'word2' : 고유숫자} #이를 단어 사전이라고 표현함.
len(vocs) # 31

# 3. 희소행렬(sparse matrix)
# 희소행렬을 근거로 밀도행렬을 만들어 사이즈를 줄인 후, 그것으로 모델을 만든다.
sparse_mat = tfidf.fit_transform(sentences) #Document X Term 행렬구조
type(sparse_mat) #scipy.sparse.csr.csr_matrix

sparse_mat #3x31 sparse matrix of type
sparse_mat.shape #(3, 31)
print(sparse_mat)
'''
  (row:문서번호, col:단어일련번호)    weight(tfidf방식에 의해서 나온 가중치)
  (0, 3)        0.2205828828763741 'fellow': 3
  (0, 16)       0.2205828828763741
  (0, 25)       0.2205828828763741
  (0, 17)       0.2205828828763741
  (0, 10)       0.2205828828763741
  (0, 1)        0.2205828828763741
            :   
  (0, 14)       0.4411657657527482 'mr': 14 : 출연빈도수가 많으면 가중치가 높다.
'''

# scipy -> numpy
sparse_mat_arr = sparse_mat.toarray()
type(sparse_mat_arr) #numpy.ndarray
sparse_mat_arr.shape #(3, 31)
sparse_mat_arr

'''
array([[0.        , 0.22058288, 0.22058288, 0.22058288, 0.        ,
        0.26055961, 0.        , 0.        , 0.        , 0.16775897,
        0.22058288, 0.22058288, 0.        , 0.        , 0.44116577,
        0.22058288, 0.22058288, 0.22058288, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.16775897, 0.44116577,
        0.22058288, 0.        , 0.        , 0.        , 0.        ,
        0.22058288],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.26903992, 0.45552418, 0.        , 0.34643788, 0.34643788,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.34643788,
        0.34643788, 0.34643788, 0.        , 0.34643788, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.27054288, 0.        , 0.        , 0.        , 0.27054288,
        0.15978698, 0.        , 0.27054288, 0.20575483, 0.        ,
        0.        , 0.        , 0.27054288, 0.27054288, 0.        ,
        0.        , 0.        , 0.        , 0.27054288, 0.20575483,
        0.20575483, 0.20575483, 0.27054288, 0.        , 0.        ,
        0.        , 0.27054288, 0.27054288, 0.27054288, 0.27054288,
        0.        ]])
'''















