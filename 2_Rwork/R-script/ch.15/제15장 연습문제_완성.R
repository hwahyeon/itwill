#################################
## <제15장 연습문제>
################################# 

###################################
## 선형 회귀분석 연습문제 
###################################

# 01. ggplot2패키지에서 제공하는 diamonds 데이터 셋을 대상으로 
# carat, table, depth 변수 중 다이아몬드의 가격(price)에 영향을 
# 미치는 관계를 다음과 같은 단계로 다중회귀분석을 수행하시오.

library(ggplot2)
data(diamonds)

# 변수 선택 : carat depth table price
cols <- names(diamonds)
cols
dia_data <- diamonds[c(cols[1], cols[5:7])]

# 단계1 : 다이아몬드 가격 결정에 가장 큰 영향을 미치는 변수는?
dia_model <- lm(price ~ ., data = dia_data)
dia_model$coefficients # 회귀계수 확인 

summary(dia_model)
# 1. model 유의성 검정 : F-statistic(p-value: < 2.2e-16)
# 2. model 설명력 : Adjusted R-squared:  0.8537 
# 3. x의 유의성 검정 : x변수 모두 유의한 수준에서 영향 있음 
# carat        555.36   <2e-16
# depth        -31.38   <2e-16
# table        -33.26   <2e-16

# 단계2 : 다중회귀 분석 결과를 정(+)과 부(-) 관계로 해설
# depth와 table은 부의 관계, carat은 정의 관계 


# 02. mtcars 데이터셋을 이용하여 다음과 같은 단계로 다중회귀분석을 수행하시오.

library(datasets)
str(mtcars) # 연비 효율 data set 

df <- mtcars[c('mpg','hp','wt')]

# 단계1 : 연비(mpg)는 마력(hp), 무게(wt) 변수와 어떤 상관관계를 갖는가? 
cor(df)
# mpg  1.0000000 -0.7761684 -0.8676594
# [해설]  모두 음의 상관관계를 보인다.

# 단계2 : 마력(hp)과 무게(wt)는 연비(mpg)에 어떤 영향을 미치는가? 
cars_model <- lm(mpg ~ ., data = df)
summary(cars_model)
# hp           -3.519  0.00145
# wt           -6.129 1.12e-06 -> 0.00000112
# [해설] x변수 모두 유의미한 수준에서 음의 영향이 다고 볼 수 있다.
# 특히 wt는 매우 유의미한 수준에서 y에 영향을 미친다고 볼 수 있다.  

# 단계3 : hp = 90, wt = 2.5t일 때 회귀모델의 예측치는?
x_data <- data.frame(hp=90, wt=2.5) # x 데이터 : model 생성 시 동일 이름 
# predict(model, x)
y_pred <- predict(cars_model, x_data) # y 예측치  
y_pred # 24.67313


# 03. product.csv 파일의 데이터를 이용하여 다음과 같은 단계로 다중회귀분석을 수행하시오.
setwd("D:/ITWILL/2_Rwork/Part-IV")
product <- read.csv("product.csv", header=TRUE)

#  단계1 : 학습데이터(train),검정데이터(test)를 7 : 3 비율로 샘플링
idx <- sample(nrow(product), 0.7*nrow(product), replace = F)
idx

train <- product[idx, ]
test <- product[-idx, ]

#  단계2 : 학습데이터 이용 회귀모델 생성 
#        변수 모델링) y변수 : 제품_만족도, x변수 : 제품_적절성, 제품_친밀도
model <- lm(제품_만족도 ~ ., data = train) 

#  단계3 : 검정데이터 이용 모델 예측치 생성 
y_pred <- predict(model, test)
y_true <- test$'제품_만족도'

#  단계4 : 모델 평가 : MSE, cor()함수 이용  
mse = mean((y_pred - y_true)**2)
cat('MSE =', mse) # MSE = 0.2994912
cat('cor =', cor(y_pred, y_true))
# cor = 0.7646438

###################################
## 로지스틱 회귀분석 연습문제 
###################################
# 04.  admit 객체를 대상으로 다음과 같이 로지스틱 회귀분석을 수행하시오.
# <조건1> 변수 모델링 : y변수 : admit, x변수 : gre, gpa, rank 
# <조건2> 7:3비율로 데이터셋을 구성하여 모델과 예측치 생성 
# <조건3> 분류 정확도 구하기 

# 파일 불러오기
admit <- read.csv('admit.csv')
str(admit) # 'data.frame':	400 obs. of  4 variables:
#$ admit: 입학여부 - int  0 1 1 1 0 1 1 0 1 0 ...
#$ gre  : 시험점수 - int  380 660 800 640 520 760 560 400 540 700 ...
#$ gpa  : 시험점수 - num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
#$ rank : 학교등급 - int  3 3 1 4 4 2 1 2 3 2 ...

# 1. data set 구성 
idx <- sample(1:nrow(admit), nrow(admit)*0.7)
train_admit <- admit[idx, ]
test_admin <- admit[-idx, ]

# 2. model 생성 

# 3. predict 생성 

# 4. 모델 평가(분류정확도) : 혼돈 matrix 이용/ROC Curve 이용
