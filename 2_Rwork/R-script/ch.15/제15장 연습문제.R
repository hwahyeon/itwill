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
head(diamonds)

# 단계1 : 다이아몬드 가격 결정에 가장 큰 영향을 미치는 변수는?

# 단계2 : 다중회귀 분석 결과를 정(+)과 부(-) 관계로 해설


#(1) 변수 선택 : 적절성 + 친밀도 -> 만족도 
y = diamonds$price # 종속변수
x1 = diamonds$carat # 독립변수1
x2 = diamonds$table # 독립변수2
x3 = diamonds$depth # 독립변수3

df_dia <- data.frame(x1, x2, x3, y)

result.lm <- lm(formula=y ~ ., data=df_dia)

# 계수 확인 
result.lm
#  Coefficients:
#   (Intercept)           x1           x2           x3  
#       13003.4       7858.8       -104.5       -151.2  

b <- 13003.4 
a1 <- 7858.8
a2 <- -104.5
a3 <- -151.2
head(df_dia)

X1 <- 0.23
X2 <- 55
X3 <- 61.5
# y = 326

# 다중회귀방정식
y = a1*X1 + a2*X2 + a3*X3 + b
err = y-Y
abs(err) #238.376

# 분석결과 확인
summary(result.lm)
# 1. F-statistic: 1.049e+05,  p-value: < 2.2e-16
# 2. Adjusted R-squared:  0.8537
# 3. x의 유의성 검정

# x1          555.36   <2e-16 ***
# x2          -33.26   <2e-16 ***
# x3          -31.38   <2e-16 ***



# 02. mtcars 데이터셋을 이용하여 다음과 같은 단계로 다중회귀분석을 수행하시오.

library(datasets)
str(mtcars) # 연비 효율 data set 
head(mtcars)

# 단계1 : 연비(mpg)는 마력(hp), 무게(wt) 변수와 어떤 상관관계를 갖는가? 

# 단계2 : 마력(hp)과 무게(wt)는 연비(mpg)에 어떤 영향을 미치는가? 

# 단계3 : hp = 90, wt = 2.5t일 때 회귀모델의 예측치는?


#(1) 변수 선택 : 적절성 + 친밀도 -> 만족도 
y = mtcars$mpg # 종속변수
x1 = mtcars$hp # 독립변수1
x2 = mtcars$wt # 독립변수2

df_mt <- data.frame(x1, x2, y)

result.lm <- lm(formula=y ~ ., data=df_mt)

# 계수 확인 
result.lm
#  Coefficients:
# (Intercept)           x1           x2  
#   37.22727     -0.03177     -3.87783  

b <- 37.22727 
a1 <- -0.03177
a2 <- -3.87783
head(df_mt)

X1 <- 110
X2 <- 2.620
# y = 21.0

# 다중회귀방정식
y = a1*X1 + a2*X2 + b
err = y-Y
abs(err) #238.376

# 분석결과 확인
summary(result.lm)
# 1. F-statistic: 69.21,  p-value: 9.109e-12
# 2. Adjusted R-squared:  0.8148 
# 3. x의 유의성 검정

# x1 -3.519  0.00145 ** 
# x2 -6.129 1.12e-06 *** -> 0.00000112

# hp = 90, wt = 2.5t
y = -0.03177*(90) + -3.87783*(2.5) + 37.22727
y #24.67339




x_data <- data.frame(hp=90, wt=2.5) # x데이터 : model 생성 시 동일 이름// x변수와 이름이 동일해야함.
#predict(model, x)
y_pred <- predict(cars_model, x_data) # y 예측치
y_pred #24.67313



# 03. product.csv 파일의 데이터를 이용하여 다음과 같은 단계로 다중회귀분석을 수행하시오.
product <- read.csv("product.csv", header=TRUE)
names(product) #"제품_친밀도" "제품_적절성" "제품_만족도"

nrow(product) #264

#  단계1 : 학습데이터(train),검정데이터(test)를 7 : 3 비율로 샘플링
x <- sample(nrow(product), 0.7*nrow(product), replace = F) #replace = F중복추출을 하지 않겠다는 뜻.
train <- product[x, ] # 학습데이터 추출
test <- product[-x, ] # 검정데이터 추출
dim(train) # 184 3 
dim(test) # 80 3


#  단계2 : 학습데이터 이용 회귀모델 생성 
#        변수 모델링) y변수 : 제품_만족도, x변수 : 제품_적절성, 제품_친밀도

model <- lm(제품_만족도 ~., data = train)

#  단계3 : 검정데이터 이용 모델 예측치 생성 
y_pred <- predict(model, test)
y_pred
y_true <- test$'제품_만족도'

#  단계4 : 모델 평가 : MSE, cor()함수 이용  
mse = mean((y_pred - y_true)**2)
cat('MSE =', mse) #0.3646024






###################################
## 로지스틱 회귀분석 연습문제 
###################################
# 04.  admit 객체를 대상으로 다음과 같이 로지스틱 회귀분석을 수행하시오.
# <조건1> 변수 모델링 : y변수 : admit, x변수 : gre, gpa, rank 
# <조건2> 7:3비율로 데이터셋을 구성하여 모델과 예측치 생성 
# <조건3> 분류 정확도 구하기 

# 파일 불러오기
admit <- read.csv(file.choose())
#admit <- read.csv('admit.csv')
str(admit) # 'data.frame':	400 obs. of  4 variables:
#$ admit: 입학여부 - int  0 1 1 1 0 1 1 0 1 0 ...
#$ gre  : 시험점수 - int  380 660 800 640 520 760 560 400 540 700 ...
#$ gpa  : 시험점수 - num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
#$ rank : 학교등급 - int  3 3 1 4 4 2 1 2 3 2 ...
table(admit$admit)
# 실패0 성공1 
#   273   127 

# 1. data set 구성
library(nnet)
idx <- sample(1:nrow(admit), nrow(admit)*0.7)
train_admit <- admit[idx, ]
test_admin <- admit[-idx, ]

dim(train_admit) #280   4
dim(test_admin) #120   4

# 2. model 생성 
model <- glm(admit ~., data = train_admit)
model

# 3. predict 생성 
pred <- predict(model, test_admin, type = 'response')
range(pred)

y_pred <- ifelse(pred >= 0.5, 1, 0)
y_true <- test_admin$admit #실제항


# 4. 모델 평가(분류정확도) : 혼돈 matrix 이용/ROC Curve 이용
tab <- table(y_true, y_pred)
tab
#         y_pred
# y_true  0  1
#      0 72  3
#      1 37  8

# 5. model 평가
acc <- (tab[1,1] + tab[2,2]) / sum(tab)
cat('분류 정확도 =', acc)

recall <- tab[2,2] / (tab[2,1]+tab[2,2])
precision <- tab[2,2] / (tab[2,1]+tab[2,2])

F1 = 2*((recall*precision)/(recall+precision))
cat('F1 score =', F1)











