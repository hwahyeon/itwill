# chap15_2_Logistic_Regression

###############################################
# 15_2. 로지스틱 회귀분석(Logistic Regression) 
###############################################

# 목적 : 일반 회귀분석과 동일하게 종속변수와 독립변수 간의 관계를 나타내어 
# 향후 예측 모델을 생성하는데 있다.

# 차이점 : 종속변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때
# 해당 데이터의결과가 특정 분류로 나눠지기 때문에 분류분석 방법으로 분류된다.
# 유형 : 이항형(종속변수가 2개 범주-Yes/No), 다항형(종속변수가 3개 이상 범주-iris 꽃 종류)
# 다항형 로지스틱 회귀분석 : nnet, rpart 패키지 이용 
# a : 0.6,  b:0.3,  c:0.1 -> a 분류 

# 분야 : 의료, 통신, 기타 데이터마이닝

# 선형회귀분석 vs 로지스틱 회귀분석 
# 1. 로지스틱 회귀분석 결과는 0과 1로 나타난다.(이항형)
# 2. 정규분포 대신에 이항분포를 따른다.
# 3. 로직스틱 모형 적용 : 변수[-무한대, +무한대] -> 변수[0,1]사이에 있도록 하는 모형 
#    -> 로짓변환 : 출력범위를 [0,1]로 조정
# 4. 종속변수가 2개 이상인 경우 더미변수(dummy variable)로 변환하여 0과 1를 갖도록한다.
#    예) 혈액형 AB인 경우 -> [1,0,0,0] AB(1) -> A,B,O(0)


# 단계1. 데이터 가져오기
weather = read.csv("C:/ITWILL/2_Rwork/Part-IV/weather.csv", stringsAsFactors = F) 
# stringsAsFactors = F : 순수한 문자형으로 가져오기 

dim(weather)  # 366  15
head(weather)
str(weather)
# 숫자형만 x변수로 쓸 수 있다. 여기선 범주형 RainTomorrow를 y변수로 사용할 것임.
# 팩터형은 자동으로 더미변수를 만들어준다. stringsAsFactors = T로 하면
# 혹은 아예 안쓰면, 아래 1, 0으로 바꿔줄 필요가 없다.
# 문자형으로 일단 가져오겠다는 것을 보여주기 위함임.

# chr 칼럼, Date, RainToday 칼럼 제거 
weather_df <- weather[, c(-1, -6, -8, -14)] #x변수로 부적절한 것들을 제거
str(weather_df)

# RainTomorrow 칼럼 -> 로지스틱 회귀분석 결과(0,1)에 맞게 더미변수 생성      
weather_df$RainTomorrow[weather_df$RainTomorrow=='Yes'] <- 1
weather_df$RainTomorrow[weather_df$RainTomorrow=='No'] <- 0 #base = 0
weather_df$RainTomorrow <- as.numeric(weather_df$RainTomorrow)
head(weather_df)

# y 빈도수
table(weather_df$RainTomorrow)
#   0   1 
# 300  66 # 0인 경우가 훨씬 많은 것을 볼 수 있다.
prop.table(table(weather_df$RainTomorrow))
#         0         1 
# 0.8196721 0.1803279 비율로 확인해본 것.

#  단계2.  데이터 셈플링
idx <- sample(1:nrow(weather_df), nrow(weather_df)*0.7)
train <- weather_df[idx, ]
test <- weather_df[-idx, ]

#  단계3.  로지스틱  회귀모델 생성 : 학습데이터 
weater_model <- glm(RainTomorrow ~ ., data = train, family = 'binomial')
# glm은 로지스틱 회귀모델을 만들어주는 함수.
# . :나머지 10개 변수는 x로 하겠다. family = 'binomial'는 y에 대한 변수가 이항이란 뜻이다.
# 즉 이항분류형식으로 모델을 만들겠다는 뜻.
weater_model 
summary(weater_model) 


# 단계4. 로지스틱  회귀모델 예측치 생성 : 검정데이터 
# newdata=test : 새로운 데이터 셋, type="response" : 0~1 확률값으로 예측 (시그모이드 함수)
pred <- predict(weater_model, newdata=test, type="response")  #예측치를 만듦.
pred
range(pred, na.rm = T) #0.0008622668   0.9867003537 #0.5 이상이면 비가 온다, 미만이면 안 온다.
summary(pred)
str(pred)

#cut off =0.5를 이용해서 이항 분류
cpred <- ifelse(pred >= 0.5, 1, 0) #0.5이상이면 1로 보고 그렇지 않으면 0으로 보겠다.
table(cpred) 
# 0  1 
# 92 17 -> 즉 108개. 총 366개이지만 30%만 검정데이터로 썼기 때문에 108개만 나온 것.

y_true <- test$RainTomorrow #이제 0과 1로 구성되어있다.

# 단계5 : 모델 검증

# 교차분할표
# table 집단 변수를 행으로 변환
table(y_true, cpred) #(행, 렬)
tab <- table(y_true, cpred)
tab

# cpred
# y_true  0  1
#      0 82  8  // 82 예측치 총 90개 중
#      1 10  9  // 비가 오는 경우엔 19개 중 10개만 예측한 것임.
# 실제 0인 경우는 82+8로 총 90
# 비가 오는 경우인 1인 경우는 10+9 총 19경우다. 대각선 82, 9 정분류(accuracy), 8, 10을 오분류라고 봄.

# 단계5 : 모델 평가 

# 1) 정분류 : 분류정확도 
acc <- (82+9) / nrow(test)

acc <- (tab[1,1]+tab[2,2]) / nrow(test) #좌대각선
cat('accuracy =', acc) # accuracy = 0.8545455

no <- 84 / (84+6)
no # 0.9333333

yes <- 10 / (10+10)
yes # 0.5

# no_acc <- (tab[1,2] + tab[2,1]) / nrow(test) #우대각선

cat('accuracy =', acc) #accuracy = 0.8272727
# 즉 비가 안오는 경우를 잘 맞혔지만, 비가 오는 경우는 상당히 예측력이 떨어지는 것을 보인다.
# 비가 왔을 때
yes <- 9 / (10+9)
yes #0.4736842
no <- 82 / (82+8)
no #0.9111111

# 2) 오분류
no_acc <- (tab[1,2]+tab[2,1]) / nrow(test)
no_acc

# 3) 특이도 : 관측치(NO) -> NO
tab[1,1] / (tab[1,1] + tab[1,2]) # 0.9333

# 4) 민감도 = 재현률, recall : 관측치(YES) -> YES
recall <- tab[2,2] / (tab[2,1] + tab[2,2]) #0.5

# 5) 정확률 : 예측치를 기준으로 함. 예측치(yes) -> yes
precision <- tab[2,2] / (tab[1,2] + tab[2,2])
precision   #0.625

# 6) F1_score : 불균형 비율
F1_score = 2*((recall * precision) / (recall + precision)) #조합 평균


### ROC Curve를 이용한 모형평가(분류정확도)  #### 모델을 평가하는 것임.
# Receiver Operating Characteristic

install.packages("ROCR")
library(ROCR)

# ROCR 패키지 제공 함수 : prediction() -> performance
pr <- prediction(pred, test$RainTomorrow)  #prediction로 객체를 하나 먼저 만듦.
prf <- performance(pr, measure = "tpr", x.measure = "fpr") #기본속성을 넣음.
plot(prf)

#그래프에서 그래프가 포함하지 않는 면적(부분)이 error를 나타내는 것이다.



########################################
## 다항형 로지스틱 회귀분석 : nnet
########################################
# install.packages("nnet")
library(nnet)

set.seed(123)
idx <- sample(nrow(iris), nrow(iris)*0.7)
train <- iris[idx, ]
test <- iris[-idx, ]


# 활성함수
# 이항 : sigmoid function : 0~1 확률값
# 다항 : softmax function : 0~1 확률값(sum=1)
# y1=0.1 y2=0.1 y3=0.8 확률의 합을 1로 맞추는 작업하고 가장 높은 확률인 y3을 결과로 예측하는 과정

names(iris) #"Species" 범주형
model <- multinom(Species ~ ., data =train)

names(model)
model$fitted.values
range(model$fitted.values)
#0 1 // 0과 1사이 확률값 예측
rowSums(model$fitted.values)
# 모든 값 : 1

str(model$fitted.values)
#num [1:105, 1:3(y변수에 대한 범주)] - 2차원 즉 martix이다.

model$fitted.values[1,] #첫번째 예측치 # versicolor : 예측치 
train[1,] # 실제 관측치 # versicolor : 관측치 

#비율 예측 # 예측치 : 범주로 예측  
y_pred <- predict(model, test, type = 'probs')
y_pred
str(y_pred)

#cut off 적용


# y가 가진 범주를 예측
y_pred <- predict(model, test)
y_pred
str(y_pred)

y_true <- test$Species
table(y_true, y_pred)

#              y_pred
# y_true       setosa versicolor virginica
# setosa         15          0         0
# versicolor      0          9         1
# virginica       0          0        20

(15+9+20)/(15+9+20+1) #0.9777778

# 교차분할표(confusion matrix)
tab <- table(y_true, y_pred)
tab

acc <- (tab[1,1] + tab[2,2] + tab[3,3]) / nrow(test)
cat('분류정확도 =', acc)
# 분류정확도 = 0.9777778














