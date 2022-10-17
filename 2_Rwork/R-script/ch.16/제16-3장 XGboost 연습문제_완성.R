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

dmatrix <- xgb.DMatrix(data = train_mat, label = train_y)

# 3. model 생성 : xgboost matrix 객체 이용   
model <- xgboost(data = dmatrix, max_depth = 2, eta = 0.5, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)

# 4. prediction 
test_mat <- as.matrix(test[-8])
test_y <- test$Personal.Loan

pred <- predict(model, test_mat)

# 5. cut off 적용 = 0.5
y_pred <- ifelse(pred >= 0.5, 1, 0)

# 6. confusion matrix
tab <- table(test_y, y_pred)
tab
# y_pred
# test_y    0    1
#      0 1341   15
#      1   36  108

# 7. 모델 성능평가 : Accuracy
acc <- (tab[1,1]+tab[2,2]) / sum(tab)
cat('accuracy=', acc) # accuracy= 0.966

# 8. 중요변수 보기(model 비교) 
import <- xgb.importance(colnames(train_mat), 
                         model = model)
import 
#      Feature       Gain      Cover Frequency
# 1:    Income 0.40317529 0.50000000 0.3333333:수입
# 2: Education 0.38798972 0.05515101 0.1666667:교육수준
# 3:    Family 0.16699035 0.05869475 0.1666667:가족수
# 4:     CCAvg 0.04184464 0.38615424 0.3333333:평균 신용카드 

xgb.plot.importance(importance_matrix = import)

