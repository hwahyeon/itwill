#################################
## <제9장-2 연습문제>
################################# 


# 01. 트럼프 연설문(trump.txt)과 오바마 연설문(obama.txt)을 대상으로 빈도수가 2회 이상 단어를 대상으로 단어구름 시각화하시오.

##################
## 오바바 연설문 
##################
## 1. 패키지 설치와 준비 

library(tm) # 전처리 용도 
library(wordcloud) # 단어 구름 시각화

## 2. obama.txt 가져오기
obama <- file(file.choose(), encoding="UTF-8")
obama.txt <- readLines(obama) # 줄 단위 TEXT FILE 읽기 

head(obama.txt) # 앞부분 6줄 보기 - 줄 단위 문장 확인 
str(obama.txt) # chr  [1:496]

obama.txt[76]


## 3. 데이터 전처리   
# (1) 말뭉치(코퍼스:Corpus) 생성 : 텍스트를 처리할 수 있는 자료의 집합 
myCorpus <- Corpus(VectorSource(obama.txt))  # 벡터 소스 생성 -> 코퍼스 생성 
myCorpus

inspect(myCorpus[1]) # corpus 내용 보기 
inspect(myCorpus[2])

# (2) 데이터 전처리 : 말뭉치 대상 전처리 
#tm_map(x, FUN)
myCorpusPrepro <- tm_map(myCorpus, removePunctuation) # 문장부호 제거
myCorpusPrepro <- tm_map(myCorpusPrepro, removeNumbers) # 수치 제거
myCorpusPrepro <- tm_map(myCorpusPrepro, tolower) # 소문자 변경
myCorpusPrepro <-tm_map(myCorpusPrepro, removeWords, stopwords('english')) # 불용어제거
stopwords('english')

inspect(myCorpusPrepro[1])

# (3) 전처리 결과 확인 
myCorpusPrepro # Content:  documents: 76
inspect(myCorpusPrepro[1:5]) # 데이터 전처리 결과 확인(숫자, 영문 확인)


## 4. 단어 선별(단어 길이 2개 이상)
# (1) 단어길이 2개 이상(영문 1개 1byte) 단어 선별 -> matrix 변경
myCorpusPrepro_term <- TermDocumentMatrix(myCorpusPrepro, 
                        control=list(wordLengths=c(2,Inf))) # 2~무한대 
myCorpusPrepro_term


# (2) Corpus -> 평서문 변환 : matrix -> data.frame 변경
myTerm_df <- as.data.frame(as.matrix(myCorpusPrepro_term)) 


## 5. 단어 빈도수 구하기
# (1) 단어 빈도수 내림차순 정렬
wordResult <- sort(rowSums(myTerm_df), decreasing=TRUE) 
wordResult[1:10]

# (2) 내림정렬 
wordResult <- sort(rowSums(myTerm_df), decreasing=TRUE) 
wordResult[1:10]


## 6. 단어구름에 디자인 적용(빈도수, 색상, 랜덤, 회전 등)
# (1) 단어 이름 생성 -> 빈도수의 이름
myName <- names(wordResult) # 단어이름 추출  

# (2) 단어이름과 빈도수로 data.frame 생성
word.df <- data.frame(word=myName, freq=wordResult) 
head(word.df)
str(word.df) # word, freq 변수

# (3) 단어 색상과 글꼴 지정
pal <- brewer.pal(12,"Paired") # 12가지 색상 pal <- brewer.pal(9,"Set1") # Set1~ Set3

# (4) 단어 구름 시각화 - 별도의 창에 색상, 빈도수, 글꼴, 회전 등의 속성을 적용하여 
wordcloud(word.df$word, word.df$freq, 
          scale=c(5,1), min.freq=2, random.order=F, 
          rot.per=.1, colors=pal)#, family="malgun")


# 02. 공공데이터 사이트에서 관심분야 데이터 셋을 다운로드 받아서 빈도수가 5회 이상 단어를 이용하여 
#      단어 구름으로 시각화 하시오.
# 공공데이터 사이트 : www.data.go.kr


