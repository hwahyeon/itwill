#chap02_Datastructure

# 자료구조의 유형(S)

# 1. vector 자료구조
#  - 동일한 자료형을 갖는 1차원 배열구조
#  - 생성 함수 : c(), seq(), rep()
#  - 처리 함수 : apply()
# 
#  (1) c()

x <- c(1,3,5,7)
y <- c(3,5)
length(x) #4
length(y) #2

# 집합 관련 함수 : base 패키지 
union(x, y) #합집합 1 3 5 7 : x+y
setdiff(x, y) #차집합 1 7 : x-y
x-y
intersect(x, y) #교집합 3 5

#벡터 변수 유형
num <- 1:5 #c(1,2,3,4,5)
num
num <- c(-10:5)
num
num <- c(1,2,3,"4") #벡터는 동일한 자료형만 가능
num #다 문자형으로 "1" "2" "3" "4"

#벡터 원소 이름 지정
names <- c("hong","lee","kang")
names # "hong" "lee"  "kang"
ages <- c(35, 45, 55)
ages

# 나이에 이름 지정
#names()는 함수
help(names)

names(age) <- c("hong", "lee", "kang")  
age  
names(ages) <- names
ages #위에붙은 이름은 데이터가 아니다.
mode(age) # "numeric"
mean(age) # 45 

str(ages) #2개의 벡터인지 하나의 값에 이름을 붙인건지 헷갈려서 구분하기 위해서.
# Named num [1:3] 35 45 55
# - attr(*, "names")= chr [1:3] "hong" "lee" "kang"


# 2) seq()
help("seq")
num <- seq(1, 10, by=2)
num #1 3 5 7 9

num2 <- seq(10, 1, by=2) #error

num2 <- seq(10, 1, by=-2)
num2 #10 8 6 4 2

# 3) rep()
?rep # rep(x, ...)
rep(1:3, times=3) #1 2 3 1 2 3 1 2 3
rep(1:3, each=3) #1 1 1 2 2 2 3 3 3

# 색인(index) : 실제 메모리에 저장된 저장위치를 색인이라 한다.
# 형식) object[n] : 이 객체의 n번째 자료가 무엇이냐 는 뜻

a <- 1:50
a # 전체 원소
a[10] #10 -> 특정 원소 한 개
a[10:20] #10~20
a[10:20, 30:35] #error 1차원은 ,를 쓸 수 없다. [행, 열]로 해석하고 있기 때문이다. # [행, 열] - Error 
a[c(10:20, 30:35)] #1차원의 경우 c함수를 이용해서 구할 수 있다. # 10~20, 30:35

#함수 이용
length(a) #길이 = 원소 개수
a[10:length(a)-5] #5~45
a[10:(length(a)-5)] #10~45 괄호로 잘 묶어야 함.
a[seq(2, length(a), by=2)]

#특정 원소 제외(-)
a[-c(15, 25, 30:35)] #15, 25, 30:35를 제외한 것들이 출력

#조건식 (boolean)
a[a>=10 & a<=30]  #10~30 : &(and)
a[a>=10 | a<=30]  # |(or)
a[!(a>=10)] # !(not)

# 2.Matrix 자료구조
# - 동일한 자료형을 갖는 2차원 배열구조
# - 생성 함수 : matrix(), rbind(), cbind() / r은 row, c는 컬럼을 의미함.
# - 처리 함수 : apply()

# (1) matrix
m1 <- matrix(data=c(1:5)) #1에서 5까지 매트릭스를 만들겠다는 의미.
#1차원 -> 2차원(행 : n, 열 : 1)
m1
dim(m1) # 5x1
mode(m1) # 자료형은 "numeric" 
class(m1) # 자료의 구조(실제 메모리의 구조)는 "matrix"

m2 <- matrix(data = c(1:9), nrow = 3, ncol = 3, byrow = TRUE)
m2
#안에 들어가는 데이터는 행 우선으로 들어감. byrow = TRUE
# byrow = FALSE, 혹은 생략하면 열을 우선으로 데이터가 들어감.

dim(m2) #3x3

# (2) rbind
x <- 1:5
y <- 6:10
x
y
m3 <- rbind(x,y)
m3
dim(m3) # 2x5

#(3) cbind()
m4 <- cbind(x,y)
m4
dim(m4) #5x2

#ADsP 출제 문제
xy <- rbind(x,y)
#다음 보기 중 틀린 것은?
#보기1. xy[1,]는 x와 같다.
#보기2. xy[,1]는 y와 같다. (x)
#보기3. dim(xy)는 2x5구조이다.
#보기4.class(xy)는 matrix이다.

# [행,컬럼]의 구조
# 행자리가 비어있다면 전체 행을 의미하며, 컬럼 자리가 비어있다면 전체 칼럼을 의미함.
xy[1,] #x의 벡터와 같다.
xy[,1] #y하고 같지 않다. 

#색인(index) : matrix
#형식) object[row, column]

m5 <- matrix(data = c(1:9), nrow=3, ncol=3)
m5

#특정 행/열 색인
m5[1,] #열 생략하면 열전체가 나옴
m5[,1]
#box선택 : 어떤 박스 형태만 출력하려면 예를 들어 2,1에서 3,2까지 하려면
m5[2:3, 1:2]

m5[1, 2:3]
m5[2,3] #2행 3열

# - 속성
m5[-2,] #2번째 행을 제외하겠다는 의미
m5[,-3] #3번째 열을 제외하겠다는 의미
m5[,-c(1,3)] #2개 이상 제외할 경우 사용 / 2번째열만 보겠다는 의미

#열(칼럼=변수=변인) 이름 지정
colnames(m5) <- c("one", "two", "three")
m5

m5[,'one'] #열 이름으로 인덱스 사용 가능
m5[,'one':'two'] #error 열이름으로 :을 사용할 수는 없다.
m5[,1:2]

#broadcast 연산
# - 작은 차원 -> 큰 차원 늘어나서 연산

x <- matrix(1:12, nrow = 4, ncol = 3, byrow = T)
dim(x) #4 3
x

# 1) scala(0) vs matrix(2)
0.5 * x

# broadcast연산에서 행렬과 행렬의 곱은 일대일로 연산을 취한다

#2) vactor(1) vs matrix(2)
y <- 10:12
y
y + x
x

x <- matrix(0,ncol=3,nrow=4)
x
y+x

x1 <- matrix(1,nrow=4,ncol=3)
x1
x1*y


#3) 동일한 모양(shape)
x + x
x - x

#4) 전치행렬 : 행->열, 열->행
x
t(x)

# 처리 함수 : apply()
help(apply)
# apply(x, MARGIN(1/2), FUN, ...)
x
apply(x, 1, sum) # 행 단위 합계
apply(x, 2, mean) # 열 단위 평균
apply(x, 1, var) # 행 단위 분산
apply(x, 1, sd) # 행 단위 표준편차

#위는 표본에 대한 분산이 아님.

x

# 3. Array 자료구조
# - 동일한 자료형을 갖는 3차원 배열구조
# - 생성 함수 : array()

#1차원데이터를 3차원으로 만들기
arr <- array(data=c(1:12), dim=c(3,2,2)) #3x2를 2개 더 붙이겠다는 뜻
arr
dim(arr) # 3행 2열 2면

# array에 대한, 3차원에 대한 색인(index)
#arr[r,c,s] = arr[행, 열, 면]
arr[,,1] #1면
arr[,,2] #2면


data()
data("iris3")
iris3 #붓꽃의 종류로 나눈 것.
dim(iris3) #50  4  3
50*4*3

#붗꽃 dataset
iris3[,,1] #첫번째 꽃의 종류
iris3[,,2] #꽃의 종2
iris3[,,3] #꽃의 종3

iris3[10:20,1:2,1] #꽃의 종에서 10~20의 행

# 4. data.frame
# - '열 단위 서로 다른 자료형'을 갖는 2차원(행렬) 배열구조
# - 생성 함수 : data.frame()
 #- 처리 함수 : apply() -> 행렬 처리

# 1) vector 이용
no <- 1:3
name <- c("홍길동", "이순신", "유관순")
pay <- c(250, 350, 200)

emp <- data.frame(NO=no, NAME = name, PAY = pay)
emp
dim(emp) # 3x3
class(emp) # "data.frame"
mode(emp) # list 하나의 자료형으로 통합이 안될 경우 (즉, 2개 이상의 자료형을 포함하고 있을 경우)
#-> list라는 자료형으로 반환함.

# 자료 참조 : 컬럼 참조 or index 참조
# 형식) object$칼럼
pay <- emp$PAY #특정 칼럼 -> vector
mean(pay) #266.6667
sd(pay) # 76.37626

# 형식) object[row,column]
emp_row <- emp[c(1,3),] # emp[-2,]
emp_row

# 2) csb, text file, db table
setwd("C:\\ITWILL\\2_Rwork/Part-I")
getwd()

emp_txt <- read.table("emp_txt", header = T, sep = "")
emp_txt
class(emp_txt) # "data.frame"

emp_csv <- read.csv("emp.csv") # 콤마 구분자 
emp_csv
class(emp_csv) # "data.frame"

#[실습]
sid <- 1:3 #이산형 변수
score <- c(90, 85, 83) #연속형
gender <- c('M', 'F', 'M') #범주형

student <- data.frame(SID=sid, Score = score, Gender = gender, stringsAsFactors = F)
student

#stringsAsFactors = F 스트링을 팩터로 만들지 않겠다는 뜻. (기본값은 T)
#stringsAsFactors = T : 문자형 -> 요인형 변환

#내부의 자료구조 보기
str(student) # 문자형이 data.frame으로 들어오면 DF(요인형)으로 바뀐다.
# 'data.frame':	3 obs. of  3 variables:
# $ SID   : int  1 2 3 -> 이산형
# $ Score : num  90 85 83 -> 연속형
# $ Gender: Factor w/ 2 levels "F","M": 2 1 2


#stringsAsFactors을 하면
# $ Gender: chr  "M" "F" "M"
# 캐릭터로 나온다.

#특정 컬럼 -> vector
score <- student$Score

mean(score)
sum(score)
var(score)
sqrt(var(score))
sd(score)

# 산포도 : 분산, 표준편차
#모집단에 대한 분산, 표준편차
#분산 = SUM(x-산술평균)^2 / n
#표준편차 = sqrt(분산)

#표본에 대한 분산, 표준편차 <- R 함수
#분산 = SUM(x-산술평균)^2 / n-1
#표준편차 = sqrt(분산)

avg <- mean(score)
diff <- (score - avg)^2 #score는 vector, avg는 scala 즉 이 식은 브로드캐스팅 연산

Var1= sum(diff) / length(score)
Var2= sum(diff) / length(score) -1

Var1;Var2


# 5) List 자료구조
 # - key와 value 한쌍으로 자료가 저장된다.
 # - key는 중복불가, value 중복 가능
 # - key를 통해서 값(value)를 참조한다.
 # - 다양한 자료형, 자료구조를 갖는 자료구조이다.

#1) key 생략 : [key1=value, key2=value]
lst <- list('lee','이순신',35,'hong', '홍길동', 30)
lst

# 첫번째 원소 : key+value
# [[1]]     -> 기본키를 생략하고 만들면 이런 기본키(default key)가 자동으로 생성된다.
# [1] "lee" -> 값(value)
# 
# 두번째 원소 : key+value
# [[2]]        -> 기본키(default key)
# [1] "이순신" ->값2(value)
# 
# [[3]]
# [1] 35
# 
# [[4]]
# [1] "hong"
# 
# [[5]]
# [1] "홍길동"
# 
# [[6]]
# [1] 30

lst[1] #첫번째 원소 index
lst[6] #마지막 원소 index

lst[[5]] #키를 통해서 value를 참조하는 것

# 2) key = value
lst2 <- list(first = 1:5, second = 6:10)
lst2

# $first   -> 직접 만든 키는 $가 붙는다.
# [1] 1 2 3 4 5
# 
# $second
# [1]  6  7  8  9 10

# data.frame($) vs list($)
# data.frame$.칼럼명
# list$키명

# key -> value 참조
lst2$first # 1 2 3 4 5
lst2$second # 6  7  8  9 10
lst2$first[3] #3
lst2$second[2:4] #7 8 9

# 3) 다양한 자료형(숫자형, 문자형, 논리형)
lst3 <- list(name=c("홍길동","유관순"),age=c(35,25),gender=c('M','F'))
lst3
mean(lst3$age) #30

# 4) 다양한 자료구조(vector, matrix, array)
lst4 <- list(one = c('one', 'two', 'three'), two = matrix(1:9, nrow=3),
             three = array(1:12, c(2,3,2)))
lst4
# $one : 1차원
# $two : 2차원
# $three : 3차원

# 5) list 형변환
multi_list <- list(r1 = list(1,2,3),
                   r2 = list(10,20,30),
                   r3 = list(100,200,300))
multi_list

# do.call(func, object)
mat <- do.call(rbind, multi_list)
mat

# 6) list 처리 함수
x <- list(1:10) # key 생략 -> [[n]]
x

#list -> vector
v <- unlist(x) # key 제거
v

a <- list(1:5)
b <- list(6:10)

a;b

#list 객체에 함수 적용 lapply(x, function)
lapply(c(a,b), max) #list로 반환
sapply(c(a,b), max) #vector로 반환 즉, 값만 반환


# 6. 서브셋(subset)
# - 특정 행 또는 열 선택 -> 새로운 dataset 생성

letters

x <- 1:5
y <- 6:10
z <- letters[1:5] #"a" "b" "c" "d" "e"

df <- data.frame(x, y, z)
df

help("subset")
#subset(x, subset, select, drop = FALSE, ...)

# 1) 조건식으로 subset 생성 : 행 기준
df2 <- subset(df, x >= 2)
df2

# 2) select로 subset 생성 : 칼럼 기준
df3 <- subset(df, select = c(x,z))
df3

# 3) 조건식 & select
df4 <- subset(df, x>=2 & x<=4, select = c(x,z))
df4

#df2, df3, df4 모두 데이터 프레임이다.
class(df2)
class(df3)
class(df4)

df

#4) 특정 칼럼의 특정 값으로 subset 생성
df5 <- subset(df, z %in% c('a', 'c', 'e'))
#z라는 칼럼 중에 a, c, e라는 문자가 포함된 것을 서브셋으로 만든다.
# %in% 연산자
df5





# [실습] iris dataset 이용 subset 생성
iris
str(iris) #dataset의 구조보기 명령어
#'data.frame':	150 obs. of  5 variables: 데이터프레임이고, 150개의 관측지라는 뜻
#행의 개수 = 관측지

# $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
# $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
# $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
# $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
# $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1

#Factor : 요인형, num : 숫자형

iris_df = subset(iris, Sepal.Length >= mean(Sepal.Length),
                 select = c(Sepal.Length, Petal.Length, Species))
iris_df

str(iris_df)

# 'data.frame':	70 obs. of  3 variables:
# $ Sepal.Length: num  7 6.4 6.9 6.5 6.3 6.6 5.9 6 6.1 6.7 ...
# $ Petal.Length: num  4.7 4.5 4.9 4.6 4.7 4.6 4.2 4 4.7 4.4 ...
# $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 2 2 2 2 2 2 2 2 2 2 ...

# 70개 관측지, 3개의 변수


# 7. 문자열 처리와 정규 표현식
install.packages("stringr")
library(stringr) #메모리에 올리는 것

string <- "hong35lee45kang55유관순25이사도시45"
string
# [1] "hong35lee45kang55유관순25" [1]같은 인덱스가 나오면 벡터구조라고 볼 수 있다.
# 메타문자 : 패턴지정 특수 기호

#1) 반복관련 메타문자 : [x] : x 1개, {n} : n개 연속 반복을 의미
str_extract_all(string, "[a-z]{3}") #영문소문자가 연속해서 3번 오는 것을 추출한다는 의미.

# [[1]]  -> key
# [1] "hon" "lee" "kan"  -> value

str_extract_all(string, "[a-z]{3,}") #{3,} 영문자 3개 이상을 의미, 3자 이상 연속된 것을 추출한다는 의미

# [[1]]
# [1] "hong" "lee"  "kang"

name <- str_extract_all(string, "[가-힣]{3,5}") #한글 3~5자를 추출하고자 할 때
unlist(name) #unlist는 list를 벡터로 만들어줄 때 사용.
#[1] "유관순"   "이사도시" 키 [[1]]가 없어짐.

#숫자(나이) 추출
ages <- str_extract_all(string, "[0-9]{2,}") #0-9 즉 숫자 한자리를 추출하고자 할 때
#[0-9]{2,} 숫자 2자리 이상
ages_vec <- unlist(ages) #문자열에서 추출했기 때문에 문자열로 들어옴. 우선 벡터로 변경.
# [1] "35" "45" "55" "25" "24"

# 숫자형 변환
num_ages <- as.numeric(ages_vec)
num_ages
cat('나이 평균 =', mean(num_ages)) #나이 평균

# 2) 단어와 숫자 관련 메타문자
# 단어 : \\w 영어, 숫자, 한글 다 단어로 봄 -> 특수문자 제외
# 숫자 : \\d (dit의 약자)
# \하나는 이미 약속된 문자이기 때문.
# \n : 줄바꿈
# 경로 : c:\\Rwork


jumin <- "123456-1234567"

str_extract_all(jumin, "[0-9]{6}-[1-4][0-9]{6}")
str_extract_all(jumin, "[0-9]{6}-[1-4]\\d{6}")
#패턴과 일치하면 값을 반환함.
# 둘은 같은 것.

email <- "kp1234@naver.com"
str_extract_all(email, "[a-z]{3,}@[a-z]{3,}.[a-z]{2,}") # character(0)
str_extract_all(email, "[a-z]\\w{3,}@[a-z]{3,}.[a-z]{2,}") #[1] "kp1234@naver.com"

email2 <- "kp$1234@naver.com"
str_extract_all(email2, "[a-z]\\w{3,}@[a-z]{3,}.[a-z]{2,}") #중간에 특수문자로 인해 character(0)

# 3. 접두어(^) / 접미어($) 메타문자
email3 <- "1kp1234@naver.com"
str_extract_all(email3, "[a-z]\\w{3,}@[a-z]{3,}.[a-z]{2,}")
#[1] "kp1234@naver.com" 첫번째 숫자1을 인식하지 못함.

str_extract_all(email3, "^[a-z]\\w{3,}@[a-z]{3,}.[a-z]{2,}") #문장의 맨 처음부터 인식가능.
#character(0)

email4 <- "kp1234@naver.com"
str_extract_all(email4, "^[a-z]\\w{3,}@[a-z]{3,}.com$") #접미어 사용
# com$ 이메일은 반드시 com으로 끝나야한다는 뜻

# 4) 특정 문자 제외 메타문자
string
# [1] "hong35lee45kang55유관순25이사도시45"

str_extract_all(string, "[^0-9]{3,}") #숫자 제외, 나머지 반환
# [1] "hong"     "lee"      "kang"     "유관순"   "이사도시"

result <- str_extract_all(string, "[^0-9]{3,}")
result #[[1]] : 기본키
name <- str_extract_all(result[[1]], "[가-힣]{3,}")
unlist(name)
# [1] "유관순"   "이사도시"


# 2. str_length_all
length(string) #1
str_length(string) #28

#3. str_location/ str_location_all
str_locate(string,'g')
str_locate_all(string,'g')
#       start end
# [1,]     4   4
# [2,]    15  15

# 4. str_replace/str_replace_all
str_replace_all(string, "[0-9]{2}", "") #[0-9]{2}를 찾아 제거하겠다는 뜻. = 숫자 제거

# 5. str_sub : 부분 문자열
str_sub(string, start = 3, end = 5) #3번째부터 5번째 문자까지 꺼내기

# 6. str_split : 문자열 분리(토큰)
string2 = "홍길동,이순신,강감찬,유관순"
result <- str_split(string2, ",") #분리하고자 하는 기준문자(여기선 ,)을 가지고 분리함.
result

name <- unlist(result)
name # [1] "홍길동" "이순신" "강감찬" "유관순"

# 7. 문자열 결합(join) : 기본함수
paste(name, collapse=",") #결합하고자 하는 기준문자를 가지고 결합
# [1] "홍길동,이순신,강감찬,유관순"























