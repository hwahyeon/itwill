##########################
## 제17-3장 SVM 연습문제 
##########################

# 문1) 기상데이터를 다음과 같이 SVM에 적용하여 분류하시오. 
# 조건1> 포물라 적용 : RainTomorrow ~ .  
# 조건2> kernel='radial', kernel='linear' 각 model 생성 및 평가 비교 

# 1. 파일 가져오기 
weatherAUS = read.csv(file.choose()) #weatherAUS.csv
weatherAUS = weatherAUS[ ,c(-1,-2, -22, -23)] # 칼럼 제외 
str(weatherAUS)

# 2. 데이터 셋 생성 
set.seed(415)
idx = sample(1:nrow(weatherAUS), 0.7*nrow(weatherAUS))
training_w = weatherAUS[idx, ]
testing_w  = weatherAUS[-idx, ]

# 결측치 제거
training_w <- na.omit(training_w)
testing_w <- na.omit(testing_w)

# 3. 분류모델 생성 : kernel='radial', kernel='linear' 
model1 <- svm(RainTomorrow ~ ., data = training_w, kernel = 'radial')
model2 <- svm(RainTomorrow ~ ., data = training_w, kernel = 'linear')

# 4. 분류모델 평가 
pred1 <- predict(model1, testing_w)
pred2 <- predict(model2, testing_w)


tab1 <- table(testing_w$RainTomorrow, pred1)
tab2 <- table(testing_w$RainTomorrow, pred2) #x,y 위치 바뀌어도 값은 상관없다. 주로 예측값을 y에 둔다.
acc1 <- (tab1[1,1]+tab1[2,2]) / sum(tab1)
acc1 
acc2 <- (tab2[1,1]+tab2[2,2]) / sum(tab2)
acc2 #0.9067432


# 문2) 문1에서 생성한 모델을 tuning하여 최적의 모델을 생성하시오.
params <- c(0.001, 0.01, 0.1, 1, 10, 100, 1000)
length(params)

tuning <- tune.svm(RainTomorrow ~ ., data = training_w,
                   gamma = params, cost = params)
tuning

best_model <- tune.svm(RainTomorrow ~ ., data = training_w,
                          gamma = , cost = )

pred <- predict(best_model, testing)
table(testing$Species, pred)



#tuning
tuning <- tune.svm(Species ~ ., data=training,
                   gamma = params, cost = params)


