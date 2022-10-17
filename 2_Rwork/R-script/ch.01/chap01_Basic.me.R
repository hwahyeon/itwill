# chap01_Basic

#수업내용
#1. 패키지와 세션
#2. 패키지 사용법
#3. 변수와 자료형
#4. 기본함수와 작업공간

#save with encoding : UTF08로 저장해놓는다. -> 한글이 깨지지 않게 초기에 설정하는 것.

#1. 패키지와 세션
# Ctrl + spacebar 자동완성
dim(available.packages())
# [1] 15297(패키지 수)    17(패키지 정보) 2차원의 정보로 가용한 패키지의 목록을 보여줌.

getOption("max.print") #1000 모든 내용을 다 보겠다는 의미.

# session
sessionInfo() #세션 정보 제공/ 세션 실행과 종료 이전의 작업 환경 내용을 세션이라고 함
#R환경, OS환경, 다국어(locale)정보,
#바로 쓸 수 있는 패키지(attached base packages) 7개

#주요 단축키
#script 실행 : Ctrl + Enter
#save : Ctrl + s
#자동완성 : Ctrl + space bar

# a <- 10
# b <- 20
# c <- a + b
# print(c)
#한번에 주석하기 : Ctrl + Shift + C(토글) 해제할 때는 한 번 더 하면 됨.

#2. 패키지 사용법 : package = function + dataset

#1) 패키지 설치
install.packages('stringr')
install.packages(stringr) #에러 발생 패키지 설치 시 따옴표 필수
#패키지(1) + 의존성 패키지(3) -의존성 패키지는 패키지와 연관성이 있는 패키지를 말함.


# WARNING: Rtools is required to build R packages but is not currently installed. Please download and install the appropriate version of Rtools before proceeding:
#   
#   https://cran.rstudio.com/bin/windows/Rtools/
#   Installing package into ‘C:/Users/user/Documents/R/win-library/3.6’
# (as ‘lib’ is unspecified)
# URL 'https://cran.rstudio.com/bin/windows/contrib/3.6/stringr_1.4.0.zip'을 시도합니다
# Content type 'application/zip' length 216367 bytes (211 KB)
# downloaded 211 KB
# 
# package ‘stringr’ successfully unpacked and MD5 sums checked
# 
# The downloaded binary packages are in
# C:\Users\user\AppData\Local\Temp\RtmpScjS7h\downloaded_packages

# 예전 버젼의 패키지를 설치하는 방법.
# (업그레이드한 R버젼에선 전에 사용하던 패키지가 삭제되었을 경우)

# package(https://cran.rstudio.com/bin/windows/contrib/3.6/stringr_1.4.0.zip)
# 이런식으로 버전을 명시한 주소로 수동 설치해줘야 한다.


#패키지 설치 경로
.libPaths()
# [1] "C:/Users/user/Documents/R/win-library/3.6" -확장 패키지
# [2] "C:/Program Files/R/R-3.6.2/library" -기본 30개 패키지

# 3) in memory : 패키지를 메모리에 업로드upload하는 것
#R은 in memory computing 방법임
library(stringr)
library('stringr') #위와 같다.

library(help='stringr') #현재 내가 쓰고 있는 패키지에 대한 설명도 가능

search() #현재 memory에 로딩된 패키지 목록
# "package:stats"    
# "package:graphics"  "package:grDevices" "package:utils"    
# "package:datasets"  "package:methods"
# 이것들은 R스튜디오만 실행하면 기본으로 메모리에 올라가는 7개의 패키지임.

str_extract('홍길동35이순신45', '[가-힣]{3}') #[1] "홍길동"
str_extract_all('홍길동35이순신45', '[가-힣]{3}') #[[1]] [1] "홍길동" "이순신"

# 4) 패키지 삭제
remove.packages('stringr') #물리적 삭제 가능

str_extract()
#stringr을 삭제했음에도 stringr의 미리보기 명령어가 뜨는 이유는
#물리적으로는 지워졌으나, 현 세션에서 사라지지 않고 있기 때문이다.

##############################
## 패키지 설치 Error 해결법 ##
##############################

#1. 최초 패키지 설치
# - Rstudio 관리자 권한으로 실행 후 설치할 것

# 2. 기존 패키지 설치 과정에서 오류
#    1) remove.packages('the name of pacakage')
#    2) rebooting system(컴퓨터 자체를 껐다킴)
#    3) install.packages('패키지')

# 3. 변수와 자료
# 
# 1) 변수 : 메모리 이름
# 2) 변수 작성 규칙
#  - 첫자는 영문, 두번째는 숫자, 특수문자(_, .)
#  - 예문) score2020, score_2020, score.2020 다 가능하다.
#  - 보통 다른 언어에선 .을 변수로 쓰지만 R에서는 그렇지 않기 때문에 가능
#  - 예약어, 함수명 사용 불가
#  - 대소문자 구분
#    ex) NUM = 100, num = 10
#  - 변수 선언시 type선언이 없다.
#  - C의 경우 int score = 90
#    R의 경우 score = 90
#  - 가장 최근값으로 변경됨.
#    R의 모든 변수는 객체(Object)


#할당 연산자(<- , =)
var1 <- 0 #var1 = 0
var1 <- 1
var1
print(var1)

var2 <- 10
var3 <- 20

var1;var2;var3 #한 번에 한문장 안에 쓰려면 ;을 이용한다.

#색인(index) : 저장 위치 / 변수 하나에 수많은 데이터를 저장할 수 있다.
var3 <- c(10, 20, 30, 40, 50)
var3[5] #5번째 것을 꺼내겠다는 의미 / 50

#대소문자
NUM=100
num=200
print(NUM == num) # 관계식 -> T/F
# [1] FALSE

#object.member
member.id = 'hong'
member.name = '홍길동'
member.age = 35

member.id; member.age

#scala(0) vs vector(1)
score <- 95 #scala변수
scores <- c(85, 75, 95, 100) #vector변수
score #  95
scores # [1]  85  75  95 100

# 3). 자료형(data type) : 숫자형, 문자형, 논리형

int <- 100
float <- 125.23
string <- "Korea"
bool <- TRUE # TRUE, T, FALSE, F

# mode : 자료형 변환 함수
mode(int) #"numeric"
mode(float) #"numeric"
mode(string) #"character"
mode(bool) #"logical"

# is.xxxx
is.numeric(int) #TRUE 괄호 안의 것이 numeric인지 묻는 함수
is.character(string) # TRUE
is.logical(bool) #  TRUE
is.numeric(string) # FALSE

datas <- c(84, 85, 62, NA, 45)
datas # 84 85 62 NA 45

is.na(datas) #결측치(값이 없는 상태) -> TURE

# 4)자료형변환 함수 : p.20
# (1) 문자형 -> 숫자형
x <- c(10,20,30, '40') #vector
x # [1] "10" "20" "30" "40"

#벡터는 한개의 자료형만 저장함
#따라서 x에 40이라는 문자형 하나만 있어도 문자형이 된다.
mode(x) # "numeric" -> "character"
x*2 # Error 

x <- as.numeric(x) #무조건 모든 문자를 숫자로 변환할 수 없다.
# 따옴표를 벗겨내면 숫자가 되는 것들이 가능하다.
x*2
x**2

plot(x) #x축이 인덱스, y축이 값으로 해서 그래프 생성

# (2) 요인형(factor)
# 범주형 변수(=집단변수) 생성
# 독립변수(x변수) :더미변수(0, 1)생성

gender <- c('남', '여', '남', '여', '여')
mode(gender) # "character"
plot(gender) # Error 
gender # "남" "여" "남" "여" "여"

#문자형 -> 요인형 변환
fgender <- as.factor(gender) #character
mode(fgender) #numeric
plot(fgender)

fgender
#[1] 남 여 남 여 여
#Levels: 남 여 #가나다순이기 기본값이기 때문에 남자가 여자보다 먼저 나옴.

str(fgender)
#Factor w/ 2 levels "남","여": 1 2 1 2 2 숫자긴 숫자지만 숫자에 의미는 없다.
#그저 분류만 할 뿐이지 이것으로 평균을 내거나 하는 것은 의미가 없다.
#이러한 것을 더미데이터라고 한다.

#mode vs class
mode(fgender) #numeric -> 자료형 확인
class(fgender) #factor -> 자료구조 확인

##########################
##factor 형변환 고려사항##
##########################

#숫자형 변수
x <- c(4,2,4,2)
mode(x) #"numeric"

#숫자형 -> 요인형
f <- as.factor(x)
f #Levels: 2 4 -> 2 1 즉 의미를 가진 숫자가 아니라 범주를 나타내기 위한 숫자일뿐이기 때문
f # Levels: 2 4 -> 2 1 2 1

# 요인형 -> 숫자형
x2 <- as.numeric(f)
x2 #원래대로 복원이 되지 않음.

#요인형을 -> 문자형, 문자형 -> 숫자형 이 순서대로야지 원래대로 복원할 수 있다.

#요인형 -> 문자형
c <- as.character(f)

#문자형 -> 숫자형
x2 <- as.numeric(c)
x2 #4 2 4 2

#4. 기본함수와 작업공간

# 1) 기본함수 : 7개 패키지에 속한 함수들, 바로 사용 가능
sessionInfo() #attached base packages:
# stats     graphics  grDevices utils     datasets  methods   base     

#패키지 도움말
library(help='stats') #패키지의 정보를 볼 수 있음

#함수 도움말 : 패키지 안에는 10개 이상의 함수들이 있다. 도움말을 참고하는 것이 좋다.
help(sum) # sum(..., na.rm = FALSE)
x <- c(10,20,30,NA)
sum(x)
sum(x, na.rm = TRUE) #60
sum(1:5) # sum(1,2,3,4,5) #15
sum(10,20,30,NA, na.rm = TRUE) # 60

?mean #= help(mean)
mean(10,20,30,NA, na.rm = TRUE) #10이라는 값이 나오긴하지만 제대로 된 값이 아니다.
mean(x, na.rm = TRUE) #20

# 2) 기본 데이터셋
data() #기본 데이터셋 목록 보기
data(Nile) # in memory

Nile
length(Nile) #100
mode(Nile) #numeric
plot(Nile)
hist(Nile) #Histogram

# 3) 작업공간
getwd() # "C:/ITWILL/2_Rwork" #현재 작업 경로 확인
setwd("C:/ITWILL/2_Rwork/part-I") #경로 변경
getwd() # "c:/itwill/2_rwork/part-i"
read.csv("emp.csv", header = T)



emp <- read.csv("emp.csv", header = T)
emp











