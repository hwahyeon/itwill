# chap05_DataVisualization

# 차트 데이터 생성
chart_data <- c(305,450, 320, 460, 330, 480, 380, 520) 
names(chart_data) <- c("2016 1분기","2017 1분기","2016 2분기","2017 2분기","2016 3분기","2017 3분기","2016 4분기","2017 4분기")
str(chart_data)
chart_data
max(chart_data) #520

# 1. 이산변수 시각화
# - 정수단위로 나누어지는 수(자녀수, 판매수)

# (1) 막대차트
?barplot
#세로막대차트
barplot(chart_data, ylim = c(0,600), #ylim : y축의 범위
        main = "2016년vs 2017년 판매현황",
        col = rainbow(8))  #rainbow(8)막대의 개수

#가로막대차트 (horiz =TRUE)
barplot(chart_data, xlim = c(0,600),horiz =TRUE, #xlim : x축의 범위
        main = "2016년vs 2017년 판매현황",
        col = rainbow(8))

par(mfrow =c(1,2)) # 1행 2열 그래프 보기
VADeaths
str(VADeaths)

row_names <- row.names(VADeaths)
row_names
col_names <- colnames(VADeaths)
col_names
max(VADeaths) # 71.1

# beside = FALSE, horiz = FALSE
barplot(VADeaths, beside = FALSE,
        horiz = FALSE,
        main = "버지니아 사망비율",
        col =rainbow(4))

# beside = TRUE, horiz = FALSE // beside = TRUE면 그래프를 각각 표현함.
barplot(VADeaths, beside = TRUE,
        horiz = FALSE,
        main = "버지니아 사망비율",
        col =rainbow(5))
        
par(mfrow =c(1,1)) #차트 공간에 차트 한개씩 보이는 것으로 돌려놓는 기능
# beside = FALSE, horiz = FALSE
barplot(VADeaths, beside = FALSE,
        horiz = FALSE,
        main = "버지니아 사망비율",
        col =rainbow(5))

#범례 추가
legend(x=4, y=200,           #범례의 위치
       legend = row_names,
       fill = rainbow(5))

# (2) 점 차트
dotchart(chart_data,
         color = c("green","red"),
         lcolor="black",
         pch=1:2,
         labels=names(chart_data),
         xlab="매출액",
         main="분기별 판매현황",
         cex=1.2  #포인트의 크기(기본은 1)
         )

# (3) 파이 차트 시각화
pie(chart_data, labels = names(chart_data),
    border='blue',
    col=rainbow(8), cex=1.2)

#차트에 제목 추가
title("2014-2015년도 분기별 매출 현황")

table(iris$Species)
# iris의 Species의 빈도값을 나타냄
# setosa versicolor  virginica 
# 50         50         50 

pie(table(iris$Species),
    col=rainbow(3),
    main = "iris 꽃의 종 빈도수")

# 2. 연속변수 시각화
# - 시간, 길이 등의 연속성을 갖는 변수
# 1) 상자 그래프 시각화(Box graph)
summary(VADeaths) #요약 통계
boxplot(VADeaths)
range(VADeaths[,1])
mean(VADeaths[,1])
#사분위수
quantile(VADeaths[,1]) # 데이터를 4등분해서 보이는 것.
# 0%  25%  50%  75% 100% 
# 11.7 18.1 26.9 41.0 66.0 
#     25% : 1사분위수, 75% : 3사분위수, 50% : 중위수

# 2) 히스토그램 시각화 : 대칭성 확인시 사용
hist(iris$Sepal.Width,
     xlab='iris$Sepal.Width',
     col = 'mistyrose',
     main="iris 꽃받침 넓이 histogram",
     xlim =c(2.0,4.5))

#col = mistyrose : 색상(흐릿한 장미) 적용
#xlab : x축 이름, main : 제목, xlim : x축 범위

par(mfrow =c(1,2))
hist(iris$Sepal.Width, xlab="iris$Sepal.Width",
    col="green",
    main="iris 꽃받침 넓이 histogram",
    xlim =c(2.0,4.5))
# 확률 밀도로 히스토그램 그리기 - 연속형변수의 확률
hist(iris$Sepal.Width,
     xlab = "iris$Sepal.Width",
     col = "mistyrose",
     freq = F,  #빈도수를 축의 눈금으로 볼 때, 여기는 FALSE이기 때문에 밀도로 보인다.
     main="iris 꽃받침 넓이 histogram",
     xlim =c(2.0, 4.5))

par(mfrow =c(1,1))                      
hist(iris$Sepal.Width,
     xlab = "iris$Sepal.Width",
     col = "mistyrose",
     freq = F,
     main="iris 꽃받침 넓이 histogram",
     xlim =c(2.0, 4.5))
#밀도를 기준으로 line 을 그려준다. 밀도 곡선 분포
lines(density(iris$Sepal.Width),col ="red")

n <- 10000
x <- rnorm(n, mean=0, sd=1)
hist(x, freq = F) #Density로 표현하는 방법
lines(density(x), col = 'red')

# 3) 산점도 시각화
x <- runif(n=15, min =1, max = 100) #난수 생성
plot(x) #x축엔 인덱스가 들어감. x값은 y축으로 쓰임.

y<- runif(n=15, min =5, max = 120) #난수 생성
y
plot(x,y) #처음오는 것이 x축, 뒤에 오는 것이 y축에 옴
plot(x~y) #위와 순서가 다르다. 즉 plot(x,y)는 plot(y~x)와 같다.

# col 속성 = 범주형
head(iris,10)
plot(iris$Sepal.Length, iris$Petal.Length,
     col = iris$Species) #색을 종마다 넣음

price <- runif (10, min=1, max=100) # 1~100사이 10개 난수 발생
price <- rnorm(10) #난수 발생

par(mfrow =c(2,2)) # 2 행 2 열 차트 그리기
plot(price, type="l") #유형 : 실선
plot(price, type="o") #유형 : 원형과 실선 원형 통과
plot(price, type="h") #직선
plot(price, type="s") #꺾은선

# plot() 함수 속성 : pch : 연결점 문자타입 --> plotting characher 번호 (1~30)
plot(price, type="o", pch =5) # 빈 사각형
plot(price, type="o", pch =15)# 채워진 마름모
plot(price, type="o", pch =20, col ="blue") #color 지정
plot(price, type="o", pch =20, col ="orange", cex =1.5) #character expension 확대
plot(price, type="o", pch =20, col ="green", cex =2.0, lwd =3) #lwd : line width

#plot 만능차트 : 들어오는 데이터의 특성에 맞게 적합한 차트를 만들어 주는 것
methods(plot) #plot에서 지원하는 유형을 볼 수 있음.

# plot.ts : 시계열자료에 적합
WWWusage #분당 인터넷 사용시간을 기록한 R에서 제공하는 기본 테스트용 데이터셋
plot(WWWusage) #추세선

# plot.lm* : 회귀모델
install.packages("UsingR")
library('UsingR')
library(help = 'UsingR')

data(galton)
str(galton)
#유전학자 galton이 만든 데이터 , 이 사람은 회귀라는 용어를 제안함.
#회귀는 평균으로 되돌아간다는 의미.

model <- lm(y~x) #y는 영향을 받는 변수, x는 영향을 주는 변수
model <- lm(child~parent, data = galton)
plot(model) #다음 플랏을 보기 위해서는 <Return>키를 치세요
Return

# 4) 산점도 행렬 : 변수 간의 비교
pairs(iris[-5])

#꽃의 종별 산점도 행렬
#pairs(iris[row,column])
table(iris$Species)
pairs(iris[iris$Species=='setosa',1:4])
pairs(iris[iris$Species=='versicolor',1:4])

# 5) 차트를 파일로 저장하기
setwd("C:/ITWILL/2_Rwork/output") #폴더 지정

##############
jpeg("iris.jpg", width=720, height=480) #픽셀 지정 가능
plot(iris$Sepal.Length, iris$Petal.Length, col = iris$Species) 
title(main="iris 데이터 테이블 산포도 차트")
dev.off() # 장치 종료
############## 한 덩어리이다. 같이 실행해야함.


#########################
### 3차원 산점도 
#########################
install.packages('scatterplot3d')
library(scatterplot3d)

# 꽃의 종류별 분류 
iris_setosa = iris[iris$Species == 'setosa',]
iris_versicolor = iris[iris$Species == 'versicolor',]
iris_virginica = iris[iris$Species == 'virginica',]

# scatterplot3d(밑변, 오른쪽변, 왼쪽변, type='n') # type='n' : 기본 산점도 제외 
d3 <- scatterplot3d(iris$Petal.Length, iris$Sepal.Length, iris$Sepal.Width, type='n')

d3$points3d(iris_setosa$Petal.Length, iris_setosa$Sepal.Length,
            iris_setosa$Sepal.Width, bg='orange', pch=21)

d3$points3d(iris_versicolor$Petal.Length, iris_versicolor$Sepal.Length,
            iris_versicolor$Sepal.Width, bg='blue', pch=23)

d3$points3d(iris_virginica$Petal.Length, iris_virginica$Sepal.Length,
            iris_virginica$Sepal.Width, bg='green', pch=25)
