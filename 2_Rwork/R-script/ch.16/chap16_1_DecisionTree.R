# chap16_1_DecisionTree

library(rpart) # rpart() : 분류모델 생성 
# install.packages("rpart.plot")
library(rpart.plot) # prp(), rpart.plot() : rpart 시각화
# install.packages('rattle')
library('rattle') # fancyRpartPlot() : node 번호 시각화 


# 단계1. 실습데이터 생성 
data(iris)
set.seed(415)
idx = sample(1:nrow(iris), 0.7*nrow(iris))
train = iris[idx, ]
test = iris[-idx, ]
dim(train) # 105 5
dim(test) # 45  5

table(train$Species)

# 단계2. 분류모델 생성 
# rpart(y변수(범주형) ~ x변수(연속형), data)
model = rpart(Species~., data=train) # iris의 꽃의 종류(Species) 분류 
model
#1) root 105 68 setosa (0.35238095 0.31428571 0.33333333) 
#root node : 105는 사용할 데이터셋의 전체 크기, 
#setosa : 3개의 레이블 중에서 가장 많은 비율을 차지하는 레이블이 setosa이며 (105-68)개를 가지고 있다는 의미
# 즉, 68은 setosa를 제외한 레이블의 비율을 의미한다.
# (0.35238095 0.31428571 0.33333333) 각각의 비율을 의미한다.


# 2) Petal.Length< 2.45 37  0 setosa (1.00000000 0.00000000 0.00000000) *
# 앞쪽의 번호 1), 2)는 중요변수를 의미.
# left node : root node의 왼쪽 노드
# 분류 조건 -> 37개 0개로 분류를 하겠다는 의미 / 가장많은 비율 label = setosa
# 37개란 수치는 setosa를 말함. 나머지는 분류할 수 없다는 의미.
# (1.00000000 0.00000000 0.00000000) 각 레이블에 대한 분류 비율
# * 단노드인지 아닌지를 보여준다. *는 마지막 터미널 노드라는 것을 보여준다.
# 왼쪽에는 37개 세토사를 100% 분류하고 오른쪽 노드로 다 넘어갔다는 내용이다.

# 3) Petal.Length>=2.45 68 33 virginica (0.00000000 0.48529412 0.51470588)  
# console 창의 들여쓰기로 노드 레벨을 알 수 있다.
# right node : 분류 조건 (>=2.45) 즉 왼쪽 노드와 조건이 반대이다.
# 68개는 분류조건에 의해 분류할 수 있는 전체 갯수
# 33개는 68개중에서 가장 많은 비율을 차지하고 있는 갯수.
# 가장 많은 비율을 차지하고 있는 레이블의 이름 virginica
# 즉 68개 중 33개는 virginica이다 라는 뜻.
# (0.00000000 0.48529412 0.51470588) 각 레이블의 분류 비율을 보인다.
# * 표시가 없으므로 아래에 노드가 또 있다는 뜻이다.
# 텍스트자료만 보면 혼동이 올 수 있다.


# 분류모델 시각화 - rpart.plot 패키지 제공 
prp(model) # 간단한 시각화   
rpart.plot(model) # rpart 모델 tree 출력
fancyRpartPlot(model) # node 번호 출력(rattle 패키지 제공), 순번이 제공됨.

# *는 자식이 없는 노드를 말함.


# 단계3. 분류모델 평가  
pred <- predict(model, test) # 비율 예측 
pred <- predict(model, test, type="class") # 분류 예측 
pred

# 1) 분류모델로 분류된 y변수 보기 
table(pred)

# 2) 분류모델 성능 평가 
table(pred, test$Species)
# pred         setosa versicolor virginica
# setosa         13          0         0
# versicolor      0         16         3
# virginica       0          1        12

(13+16+12)/nrow(test)
# 0.9111111



##################################################
# Decision Tree 응용실습 : 암 진단 분류 분석
##################################################
# "wdbc_data.csv" : 유방암 진단결과 데이터 셋 분류

# 1. 데이터셋 가져오기 
wdbc <- read.csv('C:\\ITWILL\\2_Rwork\\Part-IV\\wdbc_data.csv', stringsAsFactors = FALSE)
str(wdbc)

# 2. 데이터 탐색 및 전처리 
wdbc <- wdbc[-1] # id 칼럼 제외(이상치) 
head(wdbc)
head(wdbc[, c('diagnosis')], 10) # 진단결과 : B -> '양성', M -> '악성'

# 목표변수(y변수)를 factor형으로 변환 
wdbc$diagnosis <- factor(wdbc$diagnosis, levels = c("B", "M"))
wdbc$diagnosis[1:10]

# 3. 정규화  : 서로 다른 특징을 갖는 칼럼값 균등하게 적용 (0과 1사이로 정규화)
normalize <- function(x){ # 정규화를 위한 함수 정의 
  return ((x - min(x)) / (max(x) - min(x)))
}

# wdbc[2:31] : x변수에 해당한 칼럼 대상 정규화 수행 
wdbc_x <- as.data.frame(lapply(wdbc[2:31], normalize))
wdbc_x
summary(wdbc_x) # 0 ~ 1 사이 정규화 
class(wdbc_x) # [1] "data.frame"
nrow(wdbc_x) # [1] 569

wdbc_df <- data.frame(wdbc$diagnosis, wdbc_x) # 1+30=31
dim(wdbc_df) # 569  31
head(wdbc_df)

# 4. 훈련데이터와 검정데이터 생성 : 7 : 3 비율 
idx = sample(nrow(wdbc_df), 0.7*nrow(wdbc_df))
wdbc_train = wdbc_df[idx, ] # 훈련 데이터 
wdbc_test = wdbc_df[-idx, ] # 검정 데이터 

dim(wdbc_train) #398  31
dim(wdbc_test) #171  31

# 5. rpart 분류모델 생성 
model <- rpart(wdbc.diagnosis ~.,  data = wdbc_train)
model

rpart.plot(model)

# 6. 분류모델 평가 : 분류정확도
#비율로 예측하는지, 클래스로 예측하는지 차이를 보려는 것.
# 비율로 예측하면 matrix형태로 결과가 제공됨.

#y_pred <- precit(model, wdbc_test, type = 'class')
y_pred <- predict(model, wdbc_test) #비율 예측
y_pred
y_pred <- ifelse(y_pred[,1] >= 0.5, 0, 1)
y_pred
y_true <- wdbc_test$wdbc.diagnosis # 0 or 1

table(y_true, y_pred)
#        y_pred
# y_true  B  M
#      B 99  6
#      M 14 52

acc <- (99+52) / nrow(wdbc_test)
acc #0.8830409

M <- 52 / (52+14)
M #0.7878788

B <- 99 / (99+6)
B

########################################
## 교차검정
########################################

# 단계1 : k겹 교차검정을 위한 샘플링
install.packages("cvTools")
library(cvTools)
?cvFolds
# cvFolds(n, K = 5, R = 1, #K값으로 쪼개겠다는 의미, R은 몇세트로 만들겠느냐 하는 것.
# type = c("randon", "consecutive", "interleaved")

cvFolds(n=nrow(iris), K = 3, R = 1, type = "random")
# 3등분하여(K=3) d1=50, d2=50, d3=50개씩으로 만듦.
# d2, d3로 학습해서 d1과 검증하고, d1, d2로 학습해서 d3에 적용하고 하는 식이다.
nrow(iris) #150

cross <- cvFolds(n=nrow(iris), K = 3, R = 1, type = "random")
str(cross)
# Fold:dataset 어느 데이터셋인지, d1, d2, d3인지를 보여줌
# Index:row // Index는 subsets에 들어가 있다.

# set1
d1 <- cross$subsets[cross$which == 1, 1] #[K, r]
# set2
d2 <- cross$subsets[cross$which == 2, 1]
# set3
d3 <- cross$subsets[cross$which == 3, 1]

length(d1)
length(d2)
length(d3)

# d1, d2, d3 각각 50개씩 만들어진 것을 볼 수 있다.

K <- 1:3 #k겹
R <- 1:2 #r세트

# for(r in R){ #set = 열 index(2회)
#   cat('R=', r, '\n')
#   for(k in K){ # k겹 = 행 index(3회)
#     idx <- cross$subsets[cross$which == k, r]
#     cat('k=', k, '\n')
#     print(idx)
#   }
# }
#cross <- cvFolds(n=nrow(iris), K = 3, R = 2, type = "random")인 경우에 실행 

K <- 1:3
R <- 1
ACC <- numeric()
cnt <- 1
for(r in R){ #set = 열 index(1회)
  cat('R=', r, '\n')
  for(k in K){ # k겹 = 행 index(3회)
    idx <- cross$subsets[cross$which == k, r]
    #cat('k=', k, '\n')
    #print(idx)
    test <- iris[idx,] #검정용(50)
    train <- iris[-idx,] #훈련용(100)
    model <- rpart(Species ~ ., data=train)
    pred <- predict(model, test, type = 'class')
    tab <- table(test$Species, pred)
    ACC[cnt] <- (tab[1,1]+tab[2,2]+tab[3,3])/sum(tab)
    cnt <- cnt + 1 #카운터
  }
}

ACC #0.94 0.96 0.96
mean(ACC) #0.9533333


#####################
## 타이타닉
#####################


# titanic3.csv 변수 설명
#'data.frame': 1309 obs. of 14 variables:
#1.pclass : 1, 2, 3등석 정보를 각각 1, 2, 3으로 저장
#2.survived : 생존 여부. survived(생존=1), dead(사망=0)
#3.name : 이름(제외)
#4.sex : 성별. female(여성), male(남성)
#5.age : 나이
#6.sibsp : 함께 탑승한 형제 또는 배우자의 수
#7.parch : 함께 탑승한 부모 또는 자녀의 수
#8.ticket : 티켓 번호(제외)
#9.fare : 티켓 요금
#10.cabin : 선실 번호(제외)
#11.embarked : 탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)
#12.boat     : (제외)Factor w/ 28 levels "","1","10","11",..: 13 4 1 1 1 14 3 1 28 1 ...
#13.body     : (제외)int  NA NA NA 135 NA NA NA NA NA 22 ...
#14.home.dest: (제외)


titanic <- read.csv(file.choose()) #titanic3.csv
str(titanic)

#<조건1> 6개 변수 제외 -> subset
#<조건2> survived : int -> factor 변환(0,1)
#<조건3> train vs test : 7 : 3
#<조건4> 가장 중요한 변수 ?
#<조건5> model 평가 : 분류정확도

titanic_df <- titanic[,-c(3,8,10,12:14)]
dim(titanic_df)

titanic_df$survived <- factor(titanic_df$survived)
str(titanic_df)

idx <- 
train
test

model <- 
model








































