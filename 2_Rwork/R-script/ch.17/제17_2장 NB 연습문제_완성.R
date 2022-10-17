##########################
## 제2-2장 NB 연습문제 
##########################

# 문) Spam 메시지 데이터 셋을 이용하여 NB 분류모델을 생성하고,
# 분류정확도와 F 측정치를 구하시오. 

# 실습 데이터 가져오기(TM에서 전처리한 데이터)
setwd("C:/ITWILL/2_Rwork/Part-IV")
sms_data <- read.csv('sms_spam_tm.csv')
dim(sms_data) # [1] 5558(row) 6824(word) - 6157
str(sms_data)

# X 칼럼 제외 
sms_data.df <- sms_data[-1] # 행번호 제외 
head(sms_data.df)
str(sms_data.df) # 5558 obs. of  6823 variables:

sms_data.df$sms_type # y

# 1. train과 test 데이터 셋 생성 (7:3)
idx <- sample(nrow(sms_data.df), nrow(sms_data.df)*0.7)
train_sms <- sms_data.df[idx, ]
test_sms <- sms_data.df[-idx, ]

# 2. model 생성 - train_sms
model <- naiveBayes(sms_type ~ ., data = train_sms)
model

# 3. 예측치 생성 - test_sms
pred <- predict(model, test_sms)
table(pred)
tab <- table(test_sms$sms_type, pred)
tab
# pred
#       ham spam
# ham  1444   11
# spam   27  186

# 4. 정분류율(Accuracy)
acc <- (tab[1,1] + tab[2,2]) / sum(tab)
acc # 0.9772182


# 5. F measure(f1 score)
r <- tab[2,2] / sum(tab[2,]) # 0.8732394(관측치)
p <- tab[2,2] / sum(tab[,2]) # 0.9441624(예측치)

F1 <- 2*((r*p) / (r+p))
F1 # 0.9073171





