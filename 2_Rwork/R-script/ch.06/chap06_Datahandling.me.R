# chap06_Datahandling

# 1. dplyr 패키지

install.packages("dplyr")
library(dplyr)

library(help="dplyr")

# 1) 파이프 연산자 : %>%
# 형식) df%>%func1()%>%func2()
iris

iris %>% head() %>% filter(Sepal.Length >= 5.0)
#150개 관측지가 head함수로 6개로 6개는 필터함수로 3개의 관측지로

install.packages("hflights")
library(hflights)
str(hflights)

# 2) tbl_df() : 콘솔창 크기만큼 자료를 구성
hflights_df <- tbl_df(hflights)
hflights_df #콘솔 크기만큼 데이터를 보여줌.

# 3) filter() : 행 추출
# 형식) df %>% filter(조건식)
# head() 앞에 일부만 보여주는 함수
names(iris) #iris의 다섯개 컬럼 이름
iris %>% filter(Species == 'setosa') %>% head() # = head(filter(iris, iris$Species == 'setosa'))
iris %>% filter(Sepal.Width > 3) %>% head()

iris_df <- iris %>% filter(Sepal.Width > 3)
str(iris_df)

# 형식) filter(df, 조건식)
filter(iris, Sepal.Width > 3)
filter(hflights_df, Month == 1 & DayofMonth == 1)
filter(hflights_df, Month == 1 | Month == 2)

# 4) arrange() : 정렬 함수
# 형식) df %>% arrange(칼럼명)
iris %>% arrange(Sepal.Length) %>% head() #오름차순
iris %>% arrange(desc(Sepal.Width)) %>% head() #내림차순순

# 형식) arrange(df, 칼럼명) : 월(1~12) > 도착시간 
arrange(hflights_df, Month, ArrTime)
#월(1~12) > 도착시간 // 월이 같다면 도착시간으로 정렬하겠다는 의미.

arrange(hflights_df, desc(Month), ArrTime) #내림차순

# 5) select() : 열 추출
# 형식) df %>% select()
names(iris)
iris %>% select(Sepal.Length, Petal.Length, Species) %>% head()

#형식) select(df, col1, col2, ...)
select(hflights_df, DepTime, ArrTime, TailNum, AirTime) #내가 원하는 행만 가져오기

select(hflights_df, Year:DayOfWeek)

#문) Month 기준으로 내림차순 정렬하고,
# Year, Month, AirTime 칼럼 선택하기
names(hflights_df)
arrange(hflights_df, desc(Month),
        Year, Month, AirTime)

select(arrange(hflights_df, desc(Month)), 
       Year, Month, AirTime)

# 6) mutate() : 파생변수 생성
# 형식) df %>% mutate(변수 = 함수 or 수식)
iris_diff <- iris %>% mutate(diff = Sepal.Length-Sepal.Width) %>% head()
iris_diff

# 형식) mutate(df, 변수 = 함수 or 수식)
select(mutate(hflights_df,
              diff_delay = ArrDelay-DepDelay), #출발지연-도착지연
              ArrDelay, DepDelay, diff_delay ) 

# 7) summarise() : 통계 구하기
# 형식) df %>% summarise(변수 = 통계함수())
iris %>% summarise(col1_avg = mean(Sepal.Length),
                   col2_sd = sd(Sepal.Width))

# col1_avg   col2_sd
# 1 5.843333 0.4358663

# 형식) summarise(df, 변수 = 통계함수())
summarise(hflights_df,
          delay_avg = mean(DepDelay, na.rm = T),
          delay_tot = sum(DepDelay, na.rm = T))
# 출발지연시간 평균/합계
# delay_avg delay_tot
# <dbl>     <int>
#   1      9.44   2121251


# 8) group_by(dataset, 집단변수)
# 형식) df %>% group_by(집단변수)
names(iris)
table(iris$Species) #도메인의 빈도수를 볼 수 있는 함수.

grp <- iris %>% group_by(Species)
grp
#Groups:   Species [3] // Species로 3개의 집단으로 나눠진다는 의미.

summarise(grp, mean(Sepal.Length))
# <fct>                     <dbl>
# 1 setosa                     5.01
# 2 versicolor                 5.94
# 3 virginica                  6.59

summarise(grp, sd(Sepal.Length))
# <fct>                   <dbl>
# 1 setosa                  0.352
# 2 versicolor              0.516
# 3 virginica               0.636

# group_by() [실습]
install.packages("ggplot2")
library(ggplot2)

data("mtcars") #자동차 연비
head(mtcars)
str(mtcars)

table(mtcars$cyl) # 4  6  8 
table(mtcars$gear) # 3  4  5 

# group : cyl
grp <- group_by(mtcars, cyl)
grp

# 각 cyl 집단별 연비 평균/표준편차
summarise(grp, mpg_avg = mean(mpg),
               mpg_sd = sd(mpg))

# 각 gear 집단별 무게(wt) 평균/표준편차
summarise(grp, wt_avg = mean(wt),
               wt_sd = sd(wt))

# 두 집단변수 -> 그룹화
grp2 <- group_by(mtcars, cyl, gear) #

summarise(grp2, mpg_avg = mean(mpg),
          mpg_sd = sd(mpg))

# 형식) group_by(dataset, 집단변수)
# 예제) 각 항공기별 비행편수가 40편 이상이고,
# 평균 비행거리가 2,000마일 이상인 경우의
# 평균 도착지연시간을 확인하시오.

# 1) 항공기별 그룹화
str(hflights_df)

summarise(hflights_df, n()) #n함수 행갯수
          
planes <- group_by(hflights_df, TailNum) #항공기에 대한 일련번호
planes

# 2) 항공기별 요약 통계 : 비행편수, 평균 비행거리, 평균 도착지연시간
planes_state <-summarise(planes, count = n(),
              dist_avg = mean(Distance, na.rm = T),
              delay_avg = mean(ArrDelay, na.rm = T))
# na.rm = T // 결측지를 제거하겠다.

planes_state

# 3) 항공기별 요약 통계 : 조건에 맞게 필터링
planes_st1 <- filter(planes_state, count >= 40 & dist_avg >= 2000)
planes_st1

# 2. reshape2
install.packages("reshape2")
library(reshape2)

# 1) dcast() : long -> wide 긴 형식을 가로로.

data <- read.csv(file.choose()) #Part-II
data
dim(data) #22  3

# Data : 구매일자(col)
# ID : 고객 구분자(row)
# Buy : 구매수량
names(data)

#형식) dcast(Dataset, row ~ col, func) 포뮬라(~)라는 기호를 사용하는 것이 특징.
wide <- dcast(data, Customer_ID ~ Date, sum)
wide
str(wide) #data.frame
#dcast(data, Customer_ID ~ Date, sum) #Using <<Buy>> as value column: use value.var to override.

library(ggplot2)
data(mpg) #자동차 연비
str(mpg) #Classes ‘tbl_df’
mpg

mpg_df <- as.data.frame(mpg) #tbl_적용된 것을 다시 적용 안되게끔 하는 것
mpg_df
str(mpg_df)

mpg_df <- select(mpg_df, c(cyl, drv, hwy))
head(mpg_df)

# 교차셀에 hwy 합계
tab <- dcast(mpg_df, cyl ~ drv, sum) #sum, mean 등의 함수를 활용할 수 있음.
tab

# 교차셀에 hwy 출현 건수
tab2 <- dcast(mpg_df, cyl ~ drv, length)
tab2

#교차분할표
#table(행집단변수, 열집단변수)
table(mpg_df$cyl, mpg_df$drv)

unique(mpg_df$cyl)# 4 6 8 5
unique(mpg_df$drv)# "f" "4" "r"

# 2) melt() : wide -> long # "기준"을 기준으로 삼고 길게 관측치를 만드는 함수. 
wide
long <- melt(wide, id="Customer_ID")
long

#Customer_ID : 기준 칼럼
#variable value : 열이름
#value : 교차셀의 값

names(long) <- c("user_ID", "Date", "Buy")
long

#example
data("smiths")
smiths

# wide -> long
long <- melt(smiths, id='subject')
long

long2 <- melt(smiths, id=1:2) #데이터셋, 기준으로 삼고자 하는 것
long2

# long -> wide
wide <- dcast(long, subjet ~ ...) #... // subject를 제외한 나머지 컬럼들을 의미
wide

# 3. acast(dataset, 행~열~면)
data("airquality")
str(airquality)

table(airquality$Month)
#  5  6  7  8  9 -> 월
# 31 30 31 31 30 -> 일
table(airquality$Day)
dim(airquality) # 153   6

table(airquality$Month) #월
table(airquality$Day) #일
dim(airquality) #153   6

# wide -> long
air_melt <- melt(airquality, id=c("Month", "Day"), na.rm = T)
air_melt
dim(air_melt) #612   4
table(air_melt$variable)

# [일, 월, variable] 3차원으로 만들기 -> [행, 열, 면]
# acast(dataset, Day ~ Month ~ variable)
air_arr3d <- acast(air_melt, Day ~ Month ~ variable) #[31, 5, 4]
dim(air_arr3d) #31  5  4

# 오존 data
air_arr3d[,,1]
#태양열 data
air_arr3d[,,2]

######추가내용######
#4. URL 만들기 : http://www.naver.com?name='홍길동' // ?뒤에 나오는 것을 쿼리라고 함

# 1) base url 만들기
baseurl <- "http://www.sbus.or.kr/2018/lost/lost_02.htm"

# 2) page query 추가
#http://www.sbus.or.kr/2018/lost/lost_02.htm?Page=1
no <- 1:5
library(stringr)
page <- str_c('?Page=', no)
page #"?Page=1" "?Page=2" "?Page=3" "?Page=4" "?Page=5"

#outer(X(1),Y(n),func)
page_url <- outer(baseurl, page, str_c)
page_url
dim(page_url) #1 5

# reshape : 2d -> 1d
page_url <- sort(as.vector(page_url))


# 3) sear query 추가
# http://www.sbus.or.kr/2018/lost/lost_02.htm?Page=1&sear=2
no <- 1:3
sear <- str_c("&sear=", no)
sear #"&sear=1" "&sear=2" "&sear=3"

final_url <- outer(page_url, sear, str_c) #matrix
final_url <- sort(as.vector(final_url))
final_url










