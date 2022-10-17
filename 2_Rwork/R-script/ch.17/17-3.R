##################################################
#Support Vector Machine 
##################################################
# SVM 알고리즘 - 두 범주를 직선으로 분류(이진분류) 
# 선형분리 - 2개의 집합을 직선으로 분리(초평면 분리) 
# 초평면(Separating Hyperplane) : 2차원 이상 공간에서 평면 
# 가상의 직선을 중심으로 거리를 계산하여 직사각형 형태로 영역 확장
# -> 가장 가까운 점이 만날때 까지 영역  확장 

# 바이오인포매틱스의 유전자 데이터 분류 
# 용도 : 인간의 얼굴, 문자, 숫자 인식(이미지 데이터 패턴 인식) 
# 예) 스캐너로 스캔된 문서 이미지를 문자로 인식 


###############################################
####### e1071 패키지 
###############################################
# 관련 패키지(e1071, kernlab, klaR 등) 4개 중 e1071 가장 많이 사용함 

library(e1071)  

# 1. SVM 기본 개념 익히기 - Support Vector, Margin
df = data.frame(
  x1 = c(1,2,1,2,4,5,6),
  x2 = c(8,7,5,6,1,3,2),
  y=factor(c(1,1,1,1,0,0,0))
)

# 2. svm 모델 생성 : 비선형 SVM
# 형식) svm(y ~ x, data, type, kernel) 
?svm
model_svm = svm(y ~ ., data = df, na.action =na.omit)
model_svm

# default 속성(기본값) :  kernel="radial"
# kernel : 비선형(non linear) 관계를 선형적(linear)으로 변환하는 역할 
# kernel 종류 : linear, polynomial, radial, sigmoid
# cost = 1 : 오분류 조절 속성(값이 큰 경우 -> 오분류가 감소, 과적합 증가)
# cost의 값을 낮추게 되면 오분류는 증가한다.
# gamma = 0.5 : 결정경계 모양 조절 소성 (값이 큰 경우 -> 공간이 작아지고, 오분류 감소)
# 선형 분류가 불가한 


# svm 모델 시각화 
par(mfrow=c(1,1))
plot(df$x1, df$x2, col=df$y)  
X11()
plot(model_svm, df) # 분류 Factor levels에 의해서 2개 분류 (초평면)
# 초평면 그래프 해석 -> 2개의 레이블을 시각적으로 해석. 가운데가 가상의 선

pred <- predict(model_svm, df)
pred
# 1 2 3 4 5 6 7 -> 관측치에 대한 index
# 1 1 1 1 0 0 0 -> 예측치


# 3. kernel="linear" 변경 
model_svm2 = svm(y~., data=df,  kernel="linear") #커널을 선형 SVM으로 변경했을 때.
model_svm2 # model info
summary(model_svm2)
predict(model_svm2, df)
# 1 2 3 4 5 6 7 
# 1 1 1 1 0 0 0
# 선형 분류던 비선형분류던 결과 값이 같게 나오는 데이터셋이다.
# cost =1 : 오분류 조절 속성




############################
# iris 데이터 실습 
############################

# 1. 데이터셋 생성 
data(iris)
set.seed(415) # random 결과를 동일하게 지
idx = sample(1:nrow(iris), 0.7*nrow(iris))
training = iris[idx, ]
testing = iris[-idx, ]
training
testing
dim(training) # 105
dim(testing) # 45


# 2. 분류모델 생성 : 비선형 SVM
model_svm = svm(Species ~ ., data = training) # na.action =na.omit
summary(model_svm)

model_svm2 = svm(Species ~ ., data = training,
                kernel = 'linear')


# 3. 분류모델 성능 평가(testing set 적용 예측값 생성)  
pred <- predict(model_svm, testing)
pred2 <- predict(model_svm2, testing)

# 혼돈 matrix 작성 
table(pred, testing$Species)

table(pred2, testing$Species)

# 분류정확도
42 / 45 #0.9333333

43 / 45 #0.9555556 선형분류한 경우

###################################
## svm model tuning
###################################
# - 가장 최적의 속성값을 찾아서 최적의 model 생성

params <- c(0.001, 0.01, 0.1, 1, 10, 100, 1000) #10^-3 ~ 1^-3 # 파라미터를 지정할 수 있는 벡터변수
length(params) #7

#tuning
tuning <- tune.svm(Species ~ ., data=training,
                   gamma = params, cost = params)

tuning
# - best parameters:
#   gamma cost
#    0.01 1000
# 최적의 값들.
# 최적의 값을 사용하면 오차율은 0.02909091가 나온다.

best_model <- svm(Species ~ ., data=training,
                  gamma = 0.01, cost = 1000)
pred <- predict(best_model, testing)
table(testing$Species, pred)
44 / 45 #0.9777778

##################################################
# Support Vector Machine 문제 : spamfiltering
##################################################
# 단계1. 실습 데이터 가져오기
load(file.choose()) # sms_data_total.RData
ls() #올라와있는 데이터셋을 보여줌.

# 단계2. 데이터 탐색 
dim(train_sms) # train 데이터 
dim(test_sms) # test 데이터 
names(train_sms)
table(train_sms$type) # sms 메시지 유형 
table(test_sms$type)

# 단계3. 분류모델 생성 : 기본 파라미터 사용 
model_sms <- svm(type ~ ., data = train_sms) #h/s
model_sms
summary(model_sms)

model_sms2 <- svm(type ~ ., data = train_sms, kernel = 'linear')

# 단계4. 분류모델 평가 : test_sms
pred <- predict(model_sms, test_sms)
pred2 <- predict(model_sms2, test_sms)

# 단계5. 분류정확도  : table()
tab <- table(pred, test_sms$type)
acc <- (tab[1,1]+tab[2,2]) / sum(tab)
acc #0.8715925

# 단계6. 분류모델 수정 : linear kernel 방식 적용(radial과 linear 방식 비교) 

tab2 <- table(test_sms$type, pred2)
tab2
acc2 <- (tab2[1,1]+tab2[2,2]) / sum(tab2)
acc2 #0.9067432


#######################################
### 스캔된 이미지 문자 인식 
#######################################
# 1. 파일 가져오기 
letterdata = read.csv(file.choose())	#letterdata.csv
str(letterdata) # 'data.frame':	20000 obs. of  17 variables:
# y : letter, x : 나머지 16

# 2. 데이터 셋 생성 
set.seed(415)
idx = sample(1:nrow(letterdata), 0.7*nrow(letterdata))
training_letter = letterdata[idx, ]
testing_letter  = letterdata[-idx, ]

# 3. NA 제거 
training_letter2 = na.omit(training_letter)
testing_letter2 = na.omit(testing_letter)

# 4. 분류모델 생성 : 비선형 svm
model_letter <- svm(letter~., data = training_letter2)

# 5. 분류모델 평가 
pred_letter <- predict(model_letter, testing_letter2)

# 혼돈 matrix 
table(pred_letter, testing_letter2$letter)


re <- (pred_letter == testing_letter2$letter)
table(re)
# FALSE  TRUE 
# 319  5681 

prop.table(table(re))
#      FALSE       TRUE 
# 0.05316667 0.94683333 




