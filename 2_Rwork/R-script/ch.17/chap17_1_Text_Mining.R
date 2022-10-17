# chap17_1_Text_Mining

# 텍스트 마이닝(Text Mining)
# - 정형화 되지 않은 텍스트를 대상으로 유용한 정보를 찾는 기술
# - 토픽분석, 연관어 분석, 감성분석 등
# - 텍스트 마이닝에서 가장 중요한 것은 형태소 분석
# - 형태소 분석 : 문장을 분해 가능한 최소한의 단위로 분리하는 작업 

# 1. sms_spam.csv 가져오기 - stringsAsFactors = FALSE : factor형으로 읽지 않음 
setwd("C:/ITWILL/2_Rwork/Part-IV")
sms_data <- read.csv('sms_spam.csv', stringsAsFactors = FALSE)
str(sms_data)
#'data.frame':	5558 obs. of  2 variables: 변수가 2개 (type, text) = (정답, 문장)
# $ type: chr  "ham" "ham" "ham" "spam" ...
# $ text: chr

# ham인지 spam인지 처음에 나와있고 뒤에 문장이 나와있다. 즉, type, text구조로 되어있다.
# text는 string구조이기에 전처리(단어단위로 쪼개거나 등)를 하고 모델에 넣어줘야한다.
# 행렬에서 스키마행은 (ham or spam) / 여러 단어들..... 식으로 되어있다.
# 그 단어가 포함되어있으면 1 아니면 0 즉, 희소행렬식으로 나타낸다.

# 2. 분석을 위한 데이터 처리 : sms 문장을 단어 단위로 생성 

#install.packages('tm')
install.packages('slam') # tm 패키지 의존 관계 
library(slam) 
library(tm)  

#install.packages('SnowballC') # 어근 처리 함수
library(SnowballC) 
# 동명사, 분사(-ing, -ed등으로 끝나는 것들) -> 명사
sms_corpus = Corpus(VectorSource(sms_data$text)) # 1) 말뭉치 생성(vector -> corpus 변환) 
sms_corpus
inspect(sms_corpus[1]) #첫번째 문장에 대한 내부 객체 내용을 보겠다. #[1] Hope you are having a good week. Just checking in
inspect(sms_corpus[4]) #[1] complimentary 4 STAR Ibiza Holiday or 짙10,000 cash needs your URGENT collection.
                       #09066364349 NOW from Landline not to lose out! Box434SK38WP150PPM18+

sms_corpus = tm_map(sms_corpus, tolower)  # 2) 소문자 변경
inspect(sms_corpus[1]) #[1] hope you are having a good week. just checking in

sms_corpus = tm_map(sms_corpus, removeNumbers) # 3) 숫자 제거 
inspect(sms_corpus[1])

sms_corpus = tm_map(sms_corpus, removePunctuation) # 4) 문장부호(콤마 등) 제거 

sms_corpus = tm_map(sms_corpus, removeWords, stopwords("SMART")) # 5) stopwords(the, of, and 등) 제거  불용어 제거
stopwords("SMART") #571개의 불용어를 가지고 있는 집합.

sms_corpus = tm_map(sms_corpus, stripWhitespace) # 6) 여러 공백 제거(stopword 자리 공백 제거)   

sms_corpus = tm_map(sms_corpus, stemDocument) # 7) 유사 단어 어근 처리 

sms_corpus = tm_map(sms_corpus, stripWhitespace) # 8) 여러 공백 제거(어근 처리 공백 제거)   
inspect(sms_corpus[1]) #[1] hope good week check
inspect(sms_corpus[4]) #[1] complimentari star ibiza holiday 짙 cash urgent collect landlin lose boxskwpppm

sms_dtm = DocumentTermMatrix(sms_corpus) # 9) 문서와 단어 집계표 작성 (희소행렬)
sms_dtm

# <<DocumentTermMatrix (documents: 5558 행, terms: 6764 열)>> 중복되지 않은 유니크한 단어 6764개
# -> 5558 x 6764 희소행렬
# 5558*6764 #전체 셀 수
# Non-/sparse entries: 33887/37560425 -> 1이상의 빈도수를 가지고있는 것이 33887개라는 뜻. 1이상 출현빈도/전체셀
33887/37560425 #0.0009021996
# Sparsity           : 100%    -> 희소비율 : 위의 수치로 계산한 것. (0.99999...)
# Maximal term length: 40 -> 알파벳 40개로 구성된 것이 가장 긴 단어라는 뜻.
# Weighting          : term frequency (tf) -> 단어 출현 빈도수 가중치 (빈도수가 높은 단어에 가중치를 주는 것)

# 단어 구름 시각화 
library(wordcloud) # 단어 시각화(빈도수에 따라 크기 달라짐)

# 색상 12가지 적용 
pal <- brewer.pal(12,"Paired") # RColorBrewer 패키지 제공(wordcloud 설치 시 함께 설치됨)

# DTM -> TDM -> 평서문 변환
sms_mt <- as.matrix(t(sms_dtm)) # 행렬 변경 

# 행 단위(단어수) 합계 -> 내림차순 정렬   
rsum <- sort(rowSums(sms_mt), decreasing=TRUE) 

# vector에서 칼럼명(단어명) 추출 
myNames <- names(rsum) # rsum 변수에서 칼럼명(단어이름) 추출  
myNames # 단어명 

# 단어와 빈도수를 이용하여 df 생성 
df <- data.frame(word=myNames, freq=rsum) 
head(df) # word freq

# wordCloud(단어(word), 빈도수(v), 기타 속성)
# random.order=T : 위치 랜덤 여부(F 권장) , rot.per=.1 : 회전수, colors : 색상, family : 글꼴 적용 
wordcloud(df$word, df$freq, min.freq=2, random.order=F, scale=c(4,0.7),
          rot.per=.1, colors=pal, family="malgun") 


########################################
### 단어수 조정 - 단어길이, 가중치 적용 
########################################
sms_dtm = DocumentTermMatrix(sms_corpus) # 9)
sms_dtm 
# <<DocumentTermMatrix (documents: 5558, terms: 6764)>>
# Non-/sparse entries: 33887/37560425
# Sparsity           : 100%
# Maximal term length: 40
# Weighting          : term frequency (tf)


# 1. 단어길이 : 1 ~ 8
sms_dtm1 = DocumentTermMatrix(sms_corpus,
                              control = list(wordLengths= c(1,8)))
sms_dtm1
# <<DocumentTermMatrix (documents: 5558, terms: 6122)>> 단어가 제거된 이유는 단어 길이를 제한했기 때문이다.
# Non-/sparse entries: 34550/33991526
# Sparsity           : 100%
# Maximal term length: 8
# Weighting          : term frequency (tf)

# DTM -> TDM 변경 
t(sms_dtm1) # (terms: 6156, documents: 5558)
sms_tdm1 <- as.matrix(t(sms_dtm1))
dim(sms_tdm1) #6122 5558

# 2. 가중치 : 단어출현빈도로 가중치(비율) 적용 
# - 출현빈도수 -> 비율 가중치 조정  
sms_dtm2 = DocumentTermMatrix(sms_corpus,
                              control = list(wordLengths= c(1,8), weighting = weightTfIdf))

#가중치 적용 방법
# 1. TF : 단어의 출현 빈도수
# 2. TfIdf = TF * (1/DF) // 많이 선호하는 방법
TF <- 2  #한 문장 2회 출현
DF <- 10
TfIdf = TF * (1/DF)
TfIdf #0.2 -> 0.1

# DTM -> TDM 변경
sms_tdm2 <- as.matrix(t(sms_dtm2))
str(sms_tdm2)
dim(sms_tdm1) #6122 5558
sms_tdm1[1,]

sms_dtm2
# <<DocumentTermMatrix (documents: 5558, terms: 6122)>>
# Non-/sparse entries: 34550/33991526
# Sparsity           : 100%
# Maximal term length: 8
# Weighting          : term frequency - inverse document frequency (normalized) (tf-idf)

# TfIdf는 실제로 중요하지 않은 단어임에도 빈도수가 많을 때 사용함.

################################
### DTM 대상 단어 검색 
################################

# 단어 길이 검색 
class(sms_dtm1)
str(sms_dtm1)
terms <- sms_dtm1$dimnames$Terms #객체 안에 있는 단어들만 추출이 가능
terms

# 길이가 5~6자 이상 단어 검색 
library(stringr)
result <- terms[str_length(terms) >= 5 & str_length(terms) >= 6]
result
length(result) #2527

# 단어 출현빈도수 검색
word_freq <- findFreqTerms(sms_dtm1, lowfreq = 40) #tm
word_freq #40번 이상 출현한 단어들.


###############################
## chap17_Text_Mining(2)
###############################
# 기계학습을 위한 데이터 셋 생성 

# 1. sms_spam.csv 가져오기 - stringsAsFactors = FALSE : factor형으로 읽지 않음 
sms_data <- read.csv('sms_spam.csv', stringsAsFactors = FALSE)
str(sms_data)


# 2. 분석을 위한 데이터 처리 : sms 문장을 단어 단위로 생성해야 한다. 
library(tm)
library(SnowballC) # stemDocument()함수 제공 
sms_corpus = Corpus(VectorSource(sms_data$text)) # 1) 말뭉치 생성(vector -> corpus 변환) 
sms_corpus = tm_map(sms_corpus, tolower)  # 2) 소문자 변경
sms_corpus = tm_map(sms_corpus, removeNumbers) # 3) 숫자 제거 
sms_corpus = tm_map(sms_corpus, removePunctuation) # 4) 문장부호(콤마 등) 제거 
sms_corpus = tm_map(sms_corpus, removeWords, stopwords("SMART")) # 5) stopwords(the, of, and 등) 제거  
sms_corpus = tm_map(sms_corpus, stripWhitespace) # 6) 여러 공백 제거(stopword 자리 공백 제거)   
sms_corpus = tm_map(sms_corpus, stemDocument) # 7) 유사 단어 어근 처리 
sms_corpus = tm_map(sms_corpus, stripWhitespace) # 8) 여러 공백 제거(어근 처리 공백 제거)   

################################
### DTM 생성 -> X변수 생성 
################################

# 1. DTM 생성 -> 단어길이 : 2 ~ 8, 출현빈도수로 가중치 
sms_dtm = DocumentTermMatrix(sms_corpus,
                             control = list(wordLengths= c(2,8))) 
sms_dtm
# 2. DTM -> matrix 변경
sms_dtm_mat <- as.matrix(sms_dtm)
sms_dtm_mat
dim(sms_dtm_mat) # 5558 6122

# 3. 가중치 -> Factor('YES', 'NO')
convert_Func <- function(x){
  # 가중치가 1이상 -> 1, 0 -> 0
  x <- ifelse(x > 0, 1, 0) # 가중치 비율인 경우(TfIDf할 때) 
  f <- factor(x, levels = c(0,1), labels = c('NO','YES'))
  return(f) 
}

# 4. sms_dtm 사용자 함수 적용 
sms_dtm_mat_text <- apply(sms_dtm_mat, 2, convert_Func)
dim(sms_dtm_mat_text) #5558 6122

sms_dtm_mat_text[,1] # 첫번째 단어  
table(sms_dtm_mat_text[,1])
#   NO  YES 
# 5503   55 

table(sms_dtm_mat_text[1,])
# NO  YES 
# 6118    4 

# 5. DF = y + x(sms_dtm_mat_text)
sms_dtm_df = data.frame(sms_data$type, sms_dtm_mat_text)
dim(sms_dtm_df) #5558 6123

# 6. csv file save
write.csv(sms_dtm_df,
          "c:/ITWILL/2_Rwork/part-IV/sms_dtm_df.csv",
          quote = F, row.names = F)









