#chap16_2

##################################################
#randomForest
##################################################
# 결정트리(Decision tree)에서 파생된 모델 
# 랜덤포레스트는 앙상블 학습기법을 사용한 모델
# 앙상블 학습 : 새로운 데이터에 대해서 여러 개의 Tree로 학습한 다음, 
# 학습 결과들을 종합해서 예측하는 모델(PPT 참고)
# DT보다 성능 향상, 과적합 문제를 해결


# 랜덤포레스트 구성방법(2가지)
# 1. 결정 트리를 만들 때 데이터의 일부만을 복원 추출하여 트리 생성 
#  -> 데이터 일부만을 사용해 포레스트 구성 
# 2. 트리의 자식 노드를 나눌때 일부 변수만 적용하여 노드 분류
#  -> 변수 일부만을 사용해 포레스트 구성 
# [해설] 위 2가지 방법을 혼용하여 랜덤하게 Tree(학습데이터)를 구성한다.

# 새로운 데이터 예측 방법
# - 여러 개의 결정트리가 내놓은 예측 결과를 투표방식(voting) 방식으로 선택 


install.packages('randomForest')
library(randomForest) # randomForest()함수 제공 

data(iris)

# 1. 랜덤 포레스트 모델 생성 //랜덤포레스트에선 훈련셋 검정셋을 만들지 않고 시작한다.
# 형식) randomForest(y ~ x, data, ntree, mtry) #기본적으로 ntree는 500이 기본이다.
#즉 모델이 500개가 탄생하는 것.
model = randomForest(Species~., data=iris)
model
#만들어진 트리와 같은 정보가 나온다.
# 
# Call:
#   randomForest(formula = Species ~ ., data = iris) 
# Type of random forest: classification
# Number of trees: 500 (만들어진 트리가 500개)
# No. of variables tried at each split: 2 (4개의 x변수 중 가장 중요한 2개의 변수를 사용해 분류했다는 의미.)
# 
# OOB estimate of  error rate: 5.33%
# Confusion matrix:
#               setosa versicolor virginica class.error
# setosa         50          0         0        0.00 
# versicolor      0         47         3        0.06 (6%오차)
# virginica       0          5        45        0.10 (10%오차)


# Number of trees: 500
# No. of variables tried at each split: 2
model = randomForest(Species~., data=iris, ntree = 500, mtry =2)


# node 분할에 사용하는 x변수 개수
mtry <- sqrt(4) #전체 x변수의 수:4 #2 #범주형
p <- 14
mtrt <- 1/3*p #4~5개 #y변수가 연속형인 회계트리에서 사용하는 로드의 갯수


# 2. 파라미터 조정 300개의 Tree와 4개의 변수 적용 모델 생성 
model = randomForest(Species~., data=iris, 
                     ntree=300, mtry=4, na.action=na.omit )
model


# 3. 최적의 파리미터(ntree, mtry) 찾기
# - 최적의 분류모델 생성을 위한 파라미터 찾기

ntree <- c(400, 500, 600)
mtry <- c(2:4)

# 2개 vector이용 data frame 생성 
param <- data.frame(n=ntree, m=mtry)
param

for(i in param$n){ # 400,500,600
  cat('ntree = ', i, '\n')
  for(j in param$m){ # 2,3,4
    cat('mtry = ', j, '\n')
    model = randomForest(Species~., data=iris, 
                         ntree=i, mtry=j, 
                         na.action=na.omit )    
    print(model)
  }
}


# 4. 중요 변수 생성  
model3 = randomForest(Species ~ ., data=iris, 
                      ntree=500, mtry=2, 
                      importance = T,
                      na.action=na.omit )
model3 

importance(model3)
# MeanDecreaseAccuracy : 분류 정확도 개선에 기여하는 변수.
# 숫자가 크면 클수록 그만큼 y를 분류하는데 기여도가 높다는 것.
# MeanDecreaseGini : 지니계수
# 노드 불순도(불확실성) 개선에 기여하는 변수, 크면 클수록 정확도에 기여하는 것이 크다는 것.

varImpPlot(model3)





#########################
## 회귀 tree
#########################
library(MASS)

data("Boston")
str(Boston)

#crim : 도시 1인당 범죄율 
#zn : 25,000 평방피트를 초과하는 거주지역 비율
#indus : 비상업지역이 점유하고 있는 토지 비율  
#chas : 찰스강에 대한 더미변수(1:강의 경계 위치, 0:아닌 경우)
#nox : 10ppm 당 농축 일산화질소 
#rm : 주택 1가구당 평균 방의 개수 
#age : 1940년 이전에 건축된 소유주택 비율 
#dis : 5개 보스턴 직업센터까지의 접근성 지수  
#rad : 고속도로 접근성 지수 
#tax : 10,000 달러 당 재산세율 
#ptratio : 도시별 학생/교사 비율 
#black : 자치 도시별 흑인 비율 
#lstat : 하위계층 비율 
#medv(y) : 소유 주택가격 중앙값 (단위 : $1,000)

ntree <- 500
p <- 13
mtry <- 1/3*p
mtry # 4 or 5

boston_model <- randomForest(medv ~ ., data = Boston,
                              ntree = 500, mtry = 5,
                              importance = T)
# boston_model
# randomForest(formula = medv ~ ., data = Boston, ntree = 500,      mtry = 5, importance = T) 
# Type of random forest: regression
# Number of trees: 500
# No. of variables tried at each split: 5
# 
# Mean of squared residuals: 9.602952
# % Var explained: 88.62

#boston_model
importance(boston_model)
varImpPlot(boston_model)

names(boston_model)

boston_model#omportance

y_pred <- boston_model$predicted
y_true <- boston_model$y

#표준화(ㅇ_)
err <- y_true - y_pred
mse <- mean(err**2)
mse

#표준화(x)
cor(y_true, y_pred) #0.9438779

# model
# 분류 tree





##########################
##
##########################

titanic <- read.csv(file.choose()) #titanic3.csv

titanic_df <- titanic[,-c(3,8,10,12:14)]
dim(titanic_df)

titanic_df$survived <- factor(titanic_df$survived)
str(titanic_df)

idx <- sample(nrow(titanic_df), nrow(titanic_df)*0.7)
train <- titanic_df[idx,]
test <- titanic_df[-idx,]

ntree <- 500
mrty <- round(sqrt(7))

model <- randomForest(survived ~., data=titanic_df,
             ntree=ntree, mtry=mtry,
             importance = T,
             na.action = na.omit)

model
#         OOB estimate of  error rate: 21.15%
# Confusion matrix:
#     0   1 class.error
# 0 529  89   0.1440129
# 1 132 295   0.3091335

varImpPlot(model)

#중요한 변수들 sex, pclass, age, fare / sex, fare, age, pclass







############################
## entropy : 불확실성 척도
############################
# - tree model에서 중요 변수 선정 기준

# 1. x1 : 앞면, x2 : 동전이 뒷면이 나올 확률
# 불확실성이 가장 높은 경우는?
x1=0.5
x2=0.5
# -> 둘중에 뭐가 나올지 모름. 이런 경우가 불확실성이 가장 큰 경우라고 할 수 있다.

e1 <- -x1 * log2(x1) - x2 * log2(x2)  #자연로그
e <- exp(1)
e #2.718282
e1 #1

# 2. x1=0.8, x2=0.2
x1 = 0.8
x2 = 0.2

e2 <- -x1 * log2(x1) - x2 * log2(x2)
e2 #0.7219281 숫자가 작으면 작을수록 불확정성이 낮다고 볼 수 있다.

e2 <- -(x1 * log2(x1) + x2 * log2(x2)) #같은 식이다.
e2









