# chap03_DataIO


#1. data 불러오기(키보드 입력, 파일 가져오기)

# 1) 키보드 입력
x <- scan() #blank 상태에서 엔터를 치면 입력이 끝남.
x[3]
sum(x)
mean(x)

#문자 입력
string <- scan(what = character())
string

# 2) 파일 읽기
setwd("C:/ITWILL/2_Rwork/Part-I")

#(1). read.table() : 칼럼 구분(공백, 특수문자)

# 제목없음, 구분자 : 공백 txt file 가져오기
student <- read.table("student.txt") #제목 없음, 각 칸이 공백으로 구분되었을 때 사용하는 함수
student #제목열이 없는 경우 기본 제목열을 제공함.   V1   V2  V3 V4

#제목이 있는 경우, 칼럼 구분자 : 특수문자
read.table("student2.txt", header = TRUE, sep = ";") #sep = 공백일 경우 생략 가능

#결측치 처리하기 : -, &
student3 <- read.table("student3.txt", header = TRUE, sep = "")
student3

student3$'키' #Levels 요인형(문자타입)으로 바꾼 것

student31 <- read.table("student3.txt", header = TRUE, na.string = "-") #특수문자를 na로 바꾸기
student31
student31$'키' #연산이 가능한 상태
mean(student31$'키', na.rm=T)

str(student31)
class(student31) #"data.frame"

# (2) read.csv() : 구분자 : 콤마(,)
student4 <- read.csv("student4.txt", header = T, sep = ",") # header = T, sep = ",") 생략 가능
student4

#탐색기 이용 : 파일 선택
excel <- read.csv(file.choose()) #excel.csv // Ctrl + Enter하면 탐색기 창이 뜬다.
excel

# (3) xls/xlsx : 패키지 설치
install.packages("xlsx") #의존성 있는 패키지가 함께 설치됨.
# package ‘rJava’ successfully unpacked and MD5 sums checked
# package ‘xlsxjars’ successfully unpacked and MD5 sums checked
# package ‘xlsx’ successfully unpacked and MD5 sums checked


library(rJava) # 의존성 있는 패키지를 먼저 올린다.
library(xlsx)

kospi <- read.xlsx("sam_kospi.xlsx", sheetIndex = 1)
kospi

#한글이 포함된 xlsx 파일 읽기
st_excel <- read.xlsx("studentexcel.xlsx", sheetIndex = 1,
          encoding = 'UTF-8')
st_excel #한글을 깨지지 않게 가져오는 방법.

# 3) 인터넷 파일 읽기
# 데이터 셋 제공 사이트 
# http://www.public.iastate.edu/~hofmann/data_in_r_sortable.html - Datasets in R packages
# https://vincentarelbundock.github.io/Rdatasets/datasets.html
# https://r-dir.com/reference/datasets.html - Dataset site
# http://www.rdatamining.com/resources/data

titanic <- read.csv("https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/titanic.csv")
str(titanic)
dim(titanic) #1316    5
head(titanic)

#생존여부
table(titanic$survived) #빈도수 확인
# no yes 
# 817 499 

#성별 구분
table(titanic$sex)

#class 구분
table(titanic$class)

#성별 vs 생존여부 : 교차분할표
tab <- table(titanic$survived, titanic$sex)
#     man women
# no  694   123
# yes 175   324

#막대차트
barplot(tab, col = rainbow(2))


getOption("max.print") #1000 전체 개수를 의미
200*5

# 행 제한 풀기
options(max.print = 99999999) #모든 관측지를 확인할 수 있도록 옵션을 조절하는 방법
titanic

# 2. 데이터 저장(출력)하기

# 1) 화면 출력
x=20
y=30
z=x+y
cat('z=', z)

print(z) #함수 내에서 출력
z
print(z**2) #수식 표현 가능
print('z=', z) #Error cat을 사용할 것

# 2) 파일 저장(출력)
#read.table -> write.table : 구분자(공백, 특수문자)
#read.csv -> write.csv : 구분자(콤마)
#read.xlsx -> write.xlsx : 엑셀(패키지 필요)

# (1) write.table() : 공백
setwd("C:/ITWILL/2_Rwork/output")
write.table(titanic, "titanic.txt", row.names = FALSE, sep = " ")
#앞에 행번호는 저장하지 않겠다는 의미
write.table(titanic, "titanic2.txt", quote = FALSE, row.names = FALSE)


# (2) write.csv() : 콤마
head(titanic)
titanic_df <- titanic[-1] # X 칼럼 제외
titanic_df
str(titanic_df)

write.csv(titanic_df, "titanic_df.csv", row.names = FALSE, quote = F)

# (3) write.xlsx : 엑셀 파일 저장
search() #"package:xlsx"

write.xlsx(titanic, "titanic.xlsx", sheetName = "titanic", row.names = FALSE)





