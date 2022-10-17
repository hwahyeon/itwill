#################################
## <제3장 연습문제>
#################################
data()
# 문1) acq 데이터 셋을 대상으로 다음과 같이 TDM 객체를 생성하시오.
# <조건1> 전체 단어의 갯수는 몇 개인가 ? 
# <조건2> 최대 단어 길이는 몇 개인가 ? 
data(acq) # corpus 객체 
str(acq)
head(str(acq[1]))

# 작업절차 : acq -> DATA전처리(2단계 ~ 8단계) -> DTM -> TDM -> ?

# 1. DATA 전처리(3단계 ~ 8단계)
acq_corpus = tm_map(acq, tolower)  # 2) 소문자 변경
acq_corpus = tm_map(acq_corpus, PlainTextDocument) # [추가] 평서문 변경

acq_corpus = tm_map(acq_corpus, removeNumbers) # 3) 숫자 제거 
acq_corpus = tm_map(acq_corpus, removePunctuation) # 4) 문장부호(콤마 등) 제거 
acq_corpus = tm_map(acq_corpus, removeWords, stopwords("SMART")) # 5) stopwords(the, of, and 등) 제거  불용어 제거
acq_corpus = tm_map(acq_corpus, stripWhitespace) # 6) 여러 공백 제거(stopword 자리 공백 제거)   
acq_corpus = tm_map(acq_corpus, stemDocument) # 7) 유사 단어 어근 처리 
acq_corpus = tm_map(acq_corpus, stripWhitespace) # 8) 여러 공백 제거(어근 처리 공백 제거)   

# <<DocumentTermMatrix (documents: 50, terms: 1107)>>
# Non-/sparse entries: 2455/52895
# Sparsity           : 96%
# Maximal term length: 16
# Weighting          : term frequency (tf)

# 2. DTM 생성(9단계)
acq_dtm = DocumentTermMatrix(acq_corpus) # 9) 문서와 단어 집계표 작성 (희소행렬)
acq_dtm

# 3. TDM 생성(전치행렬) 
acq_mt <- as.matrix(t(acq_dtm)) # 행렬 변경 
str(acq_mt) #[1:1107, 1:50]

acq_mt[1,] #첫번째 단어보기
table(acq_mt[1,]) #abil
table(acq_mt[2,]) #accept

# 0  1 
# 49  1 

# 0  1  2 
# 48  1  1 


# 문2) crude 데이터 셋을 대상으로 다음과 같이 TDM 객체를 생성하시오.
# <조건1> 단어 길이 : 1 ~ 8
# <조건2> 가중치 적용 : 출현빈도수의 비율 
# <조건3> 위 조건의 결과를 대상으로 단어수는 몇개인가 ?  

data(crude)

# 1. DATA전처리(3단계 ~ 8단계)
crude_corpus = tm_map(crude, tolower)  # 2) 소문자 변경
crude_corpus = tm_map(crude_corpus, PlainTextDocument) # [추가] 평서문 변경

crude_corpus = tm_map(crude_corpus, removeNumbers) # 3) 숫자 제거 
crude_corpus = tm_map(crude_corpus, removePunctuation) # 4) 문장부호(콤마 등) 제거 
crude_corpus = tm_map(crude_corpus, removeWords, stopwords("SMART")) # 5) stopwords(the, of, and 등) 제거  불용어 제거
crude_corpus = tm_map(crude_corpus, stripWhitespace) # 6) 여러 공백 제거(stopword 자리 공백 제거)   
crude_corpus = tm_map(crude_corpus, stemDocument) # 7) 유사 단어 어근 처리 
crude_corpus = tm_map(crude_corpus, stripWhitespace) # 8) 여러 공백 제거(어근 처리 공백 제거)   


# 2. DTM 생성
crude_dtm = DocumentTermMatrix(crude_corpus) # 9) 문서와 단어 집계표 작성 (희소행렬)
crude_dtm

# 3. TDM 생성
crude_mt <- as.matrix(t(crude_dtm)) # 행렬 변경 
str(crude_mt) #[1:705, 1:20]

#첫번째 단어 비율(TFiDF)
table(crude_mt[1,])


