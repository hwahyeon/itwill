#####################################
## 제16-1장 DecisionTress 연습문제 
#####################################

library(rpart) # rpart() : 분류모델 생성
library(rpart.plot) # prp(), rpart.plot() : rpart 시각화
library(rattle) # fancyRpartPlot() : node 번호 시각화 

# 01. Spam 메시지 데이터 셋을 이용하여 DT 분류모델을 생성하고.
# 정분류율, 오분류율, 정확률을 구하시오. 

# 실습 데이터 가져오기
sms_data = read.csv(file.choose()) # sms_spam_tm.csv
dim(sms_data) # [1] 5558(row) 6824(word/ x변수)
sms_data$sms_type #희소행렬 형태 y변수
table(sms_data$sms_type)
#  ham spam 
# 4811  747 


# 1. train과 test 데이터 셋 생성 (7:3)

idx = sample(1:nrow(sms_data), 0.7*nrow(sms_data))
train <- sms_data[idx,]
test <- sms_data[-idx,]
dim(train) # 105 5
dim(test) # 45  5  

# 2. model 생성 - train_sms
rpart(sms_type~., data=train)

# 3. 예측치 생성 - test_sms


# 4. 분류정확도 : 정분류율, 오분류율, 정확률, 재현율 





# 02. weather 데이터를 이용하여 다음과 같은 단계별로 의사결정 트리 방식으로 분류분석을 수행하시오. 

# 조건1) y변수 : RainTomorrow, x변수 : Date와 RainToday 변수 제외한 나머지 변수로 분류모델 생성 
# 조건2) 모델의 시각화를 통해서 y에 가장 영향을 미치는 x변수 확인 
# 조건3) 비가 올 확률이 50% 이상이면 ‘Yes Rain’, 50% 미만이면 ‘No Rain’으로 범주화

# 단계1 : 데이터 가져오기
library(rpart) # model 생성 
library(rpart.plot) # 분류트리 시각화 

weather = read.csv(file.choose(), header=TRUE) # weather.csv
names(weather)


# 단계2 : 데이터 샘플링
weather.df <- weather[, c(-1,-14)]
idx <- sample(1:nrow(weather.df), nrow(weather.df)*0.7)

weather_train <- weather.df[idx, ]
weather_test <- weather.df[-idx, ]

# 단계3 : 분류모델 생성

# 단계4 : 분류모델 시각화 - 중요변수 확인 

# 단계5 : 예측 확률 범주화('Yes Rain', 'No Rain') 

# 단계6 : 혼돈 matrix 생성 및 분류 정확도 구하기
table(y_true, y_pred)










