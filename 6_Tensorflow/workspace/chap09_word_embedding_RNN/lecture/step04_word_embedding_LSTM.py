# -*- coding: utf-8 -*-
"""
step04_word_embedding_LSTM.py

1. encoding 유형 : 딥러닝 모델에서 사용되는 data
  - one hot encoding : texts -> sparse matrix(encoding) -> DNN -> label 분류 
    -> DNN : 단어출현빈도수 입력 -> label 분류   
  - word_embedding : texts -> sequence(10진) -> embedding -> RNN -> label 분류 
    -> RDD : 자연어 입력 -> label 분류 
  
2. Embedding(input_dim, output_dim, input_length)   
  - input_dim : 임베딩층에 입력되는 전체 단어 수 
  - output_dim : 임베딩층에서 출력되는 vector 수 
  - input_length : 1문장을 구성하는 단어 수 
"""

import pandas as pd 
import numpy as np 
import string
from tensorflow.keras.preprocessing.text import Tokenizer # token
from sklearn.model_selection import train_test_split 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense, Embedding, Flatten, LSTM # layer 생성 
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -- step03 참조 ---

# 1. file load
temp_spam = pd.read_csv("C:/ITWILL/6_Tensorflow/data/temp_spam_data2.csv",
                        header = None, encoding = "utf-8")
temp_spam.info()
'''
 0   0       5574 non-null   object : label(ham or spam)
 1   1       5574 non-null   object : texts(영문장)
 '''
 
# 변수 선택 
label = temp_spam[0]
texts = temp_spam[1]
len(label) 

# 2. data 전처리 
# target dummy('spam'=1, 'ham'=0)
target = [1 if x=='spam' else 0 for x in label]
print('target :', target)
target = np.array(target)

# texts 전처리
def text_prepro(texts):
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return texts

# 함수 호출 : texts 전처리 완료 
texts = text_prepro(texts)
texts[0]


# -- step01 참고 -- 
tokenizer = Tokenizer() # 토큰 생성기 object

# 1. 토큰 생성 
tokenizer.fit_on_texts(texts) # 텍스트 적용 
token = tokenizer.word_index # 토큰 반환 
print(token) 
print("전체 단어 수 :", len(token)) # 전체 단어 수 : 8629

voc_size = len(token) + 1
voc_size # 8630


# 2. 정수 인덱스 : 토큰(단어) -> 정수인덱스(단어 순서 인덱스)
seq_index = tokenizer.texts_to_sequences(texts)
print(seq_index) # [docs, terms]
len(seq_index) # 전체 문장수 : 5574
len(seq_index[0]) # 첫번째 문장의 단어 길이 : 20
len(seq_index[-1]) # 마자막 문장의 단어 길이 : 6

lens = [len(seq)  for seq in seq_index]
maxlen = max(lens)
maxlen #  171

# 3. 패딩(padding) : 정수 인덱스 길이 맞춤
x_data = pad_sequences(seq_index, maxlen = maxlen)
print(x_data)

x_data[0] # 151개를 171로 맞춘 것 (0 padding)


# 4. dataset split
x_train, x_val, y_train, y_val = train_test_split(x_data, target)
x_train.shape # (4180, 171)
x_val.shape # (1394, 171)


# 5. embedding 층 : 인코딩 
embedding_dim = 32 # 64, 128, 256 ... 전체 단어 길이 따라서 변경됨 

model = Sequential() # model 객체 생성 


# embedding layer 추가 
model.add(Embedding(input_dim=voc_size, output_dim=embedding_dim, 
                    input_length=maxlen))

# 2d -> 1d
#model.add(Flatten())

# LSTM(RNN)
model.add(LSTM(32)) # RNN layer : Flatten 기능 포함 

# DNN hidden layer 
model.add(Dense(32, activation="relu"))

# DNN output layer
model.add(Dense(1, activation="sigmoid"))

model.summary()


# 6. model 생성 & 평가  
model.compile(optimizer = 'rmsprop',
              loss = "binary_crossentropy", 
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size = 512,
          epochs=5,
          validation_data=(x_val, y_val))


loss, score = model.evaluate(x_val, y_val)
print("loss =", loss)
print("accuracy=", score)
'''
Embedding(encoding) + DNN model
loss = 0.16122994626440654
accuracy= 0.94763273

Embedding(encoding) + RNN model + DNN model 
loss = 0.1460218087929051
accuracy= 0.95767576
'''













