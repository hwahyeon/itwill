################################
## 제16-3장 XGBboost 연습문제 
################################

# 01. UniversalBank.csv 데이터셋을 이용하여 다음과 같이 XGBoost에 적용하여 분류하시오. 
# 조건1> 포물라 : formula <- Personal.Loan ~.
# 조건2> 모델 성능 평가
# 조건3> 중요변수 보기

# 대출 수락 or 거절 prediction
setwd("c:/ITWILL/2_Rwork/Part-IV")
Data = read.csv('UniversalBank.csv',  stringsAsFactors = F)
str(Data)
Data = Data[c(-1, -5)] # 1, 5번 칼럼 제외 


# Personal.Loan -> y변수(대출여부) 
str(Data)

# 1. data set 생성 
idx <- sample(nrow(Data), nrow(Data)*0.7)
train <- Data[idx, ]
test <- Data[-idx, ]
dim(train) # 3500   12
dim(test) # 1500  12

# 2. xgb.DMatrix 생성 : data(x):matrix, label(y):vecor 
train_mat <- as.matrix(train[-8]) # matrix
train_mat[1,]

# y변수 : vector 
train_y <- train$Personal.Loan

# 3. model 생성 : xgboost matrix 객체 이용   
dmatrix <- xgb.DMatrix(data=train_mat, label = train_y)

model <- xgboost(data = dmatrix, max_depth = 2, eta = 1,
                 nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)

# eta 0으로 가깝게하면 그만큼 결과를 도출해내는데 많은 시간이 필요하다.(꼼꼼하게 많이 연산하니까)
# nrounds 공부한(학습한) 횟수, 일정한 숫자를 넘어가면 의미가 없다.
# 집단변수 이항변수일 경우 objective = "binary:logistic"
# verbose = 0 실행과정에서 결과를 출력하지 않겠다.

# 4. prediction 
test_mat <- as.matrix(test[-8])
test_y <- test$Education

pred <- predict(model, test_mat) #x변수는 뒤에 넣는다.

# 5. cut off 적용 (= 0.5)
y_pred <- ifelse(pred>=0.5, 1, 0)

# 6. confusion matrix
tab <- table(test_y, y_pred)
tab
#        y_pred
# test_y   0   1
#      1 610  19
#      2 376  50
#      3 372  73

# 7. 모델 성능평가 : Accuracy
acc <- (tab[1,1]+tab[2,2]) / sum(tab)
cat('accuracy =', acc) #accuracy = 0.44

# 8. 중요변수 보기(model 비교) 
import <- xgb.importance(colnames(train_mat), model = model)
import
# Feature       Gain      Cover Frequency
# 1:    Income 0.39354349 0.50000000 0.3333333
# 2: Education 0.36768237 0.07704812 0.1666667
# 3:    Family 0.16736787 0.03795968 0.1666667
# 4:     CCAvg 0.07140627 0.38499221 0.3333333 #평균 신용카드 실적



xgb.plot.importance(importance_matrix = import)





