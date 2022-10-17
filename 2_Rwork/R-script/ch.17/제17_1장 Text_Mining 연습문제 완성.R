#################################
## <제17장 연습문제>
#################################
data()
# 문1) acq 데이터 셋을 대상으로 다음과 같이 TDM 객체를 생성하시오.
# <조건1> 전체 단어의 갯수는 몇 개인가 ? 1107
# <조건2> 최대 단어 길이는 몇 개인가 ? 16

data(acq) # corpus 객체 
str(acq)
head(str(acq[1]))

# 작업절차 : acq -> DATA전처리(2단계 ~ 8단계) -> DTM -> TDM -> ?

# 1. DATA 전처리(2단계 ~ 8단계)
acq_corpus = tm_map(acq, tolower)  # 2) 소문자 변경
acq_corpus = tm_map(acq_corpus, PlainTextDocument) # [추가] 평서문 변경

acq_corpus = tm_map(acq_corpus, removeNumbers) # 3) 숫자 제거 
acq_corpus = tm_map(acq_corpus, removePunctuation) # 4) 문장부호(콤마 등) 제거 
acq_corpus = tm_map(acq_corpus, removeWords, stopwords("SMART")) # 5) stopwords(the, of, and 등) 제거  
acq_corpus = tm_map(acq_corpus, stripWhitespace) # 6) 여러 공백 제거(stopword 자리 공백 제거)   
acq_corpus = tm_map(acq_corpus, stemDocument) # 7) 유사 단어 어근 처리 
acq_corpus = tm_map(acq_corpus, stripWhitespace) # 8) 여러 공백 제거(어근 처리 공백 제거)   


# 2. DTM 생성(9단계) 
dtm <- DocumentTermMatrix(acq_corpus)
dtm
# 50 documents, 1107 terms
# Maximal term length: 16

# 3. TDM 생성(전치행렬) -> 평서문 
tdm <- as.matrix(t(dtm))
str(tdm) # [1:1107, 1:50]

tdm[1, ] # 첫번째 단어보기 
table(tdm[1, ]) # abil
table(tdm[2, ]) # accept
#  0  1  2 -> 가중치 
# 48  1  1 -> 빈도수 


# 문2) crude 데이터 셋을 대상으로 다음과 같이 TDM 객체를 생성하시오.
# <조건1> 단어 길이 : 1 ~ 8
# <조건2> 가중치 적용 : 출현빈도수의 비율 
# <조건3> 위 조건의 결과를 대상으로 단어수는 몇개인가 ? 651  

data(crude)

# 1. DATA전처리(2단계 ~ 8단계)
acq_corpus = tm_map(acq, PlainTextDocument)  # 2) 평서문 변경
crude_corpus = tm_map(crude_corpus, removeNumbers) # 3) 숫자 제거 
crude_corpus = tm_map(crude_corpus, removePunctuation) # 4) 문장부호(콤마 등) 제거 
crude_corpus = tm_map(crude_corpus, removeWords, stopwords("SMART")) # 5) stopwords(the, of, and 등) 제거  
crude_corpus = tm_map(crude_corpus, stripWhitespace) # 6) 여러 공백 제거(stopword 자리 공백 제거)   
crude_corpus = tm_map(crude_corpus, stemDocument) # 7) 유사 단어 어근 처리 
crude_corpus = tm_map(crude_corpus, stripWhitespace) # 8) 여러 공백 제거(어근 처리 공백 제거)   


# 2. DTM 생성 
crude_dtm <- DocumentTermMatrix(crude_corpus, 
                                control = list(wordLengths= c(1,8),  
                                        weighting = weightTfIdf))
crude_dtm
# 20 documents, 651 terms

# 3. TDM 생성 
crude_tdm <- TermDocumentMatrix(crude_corpus,
                                control = list(wordLengths= c(1,8),  
                                 weighting = weightTfIdf))
crude_tdm
# 651 terms, 20 documents







