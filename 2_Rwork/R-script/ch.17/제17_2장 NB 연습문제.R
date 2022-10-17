##########################
## 제2-2장 NB 연습문제 
##########################

# 문) Spam 메시지 데이터 셋을 이용하여 NB 분류모델을 생성하고,
# 분류정확도와 F 측정치를 구하시오. 

# 실습 데이터 가져오기(TM에서 전처리한 데이터)
setwd("C:/ITWILL/2_Rwork/Part-IV")
sms_data <- read.csv('sms_spam_tm.csv')
dim(sms_data) # [1] 5558(row) 6824(word) - 6157
str(sms_data) #5558 obs. of  6824 variables: 6824개의 x변수
sms_data$sms_data.type
colnames(sms_data)[1] <- "sms_type"

sms_data$sms_type[1:5]

# X 칼럼 제외 
sms_data.df <- sms_data[-1] # 행번호 제외 
head(sms_data.df)
str(sms_data.df) # 5558 obs. of  6823 variables:

# 1. train과 test 데이터 셋 생성 (7:3)
sms_data.df
train <- [idx,]
test <- [-idx,]

# 2. model 생성 - train_sms
model <- 

# 3. 예측치 생성 - test_sms
pred <- predic(model, test_sms)
table(pred)

# 4. 정분류율(Accuracy)
acc <- (tab[1,1]+tab[2,2]) / sum(tab)
acc

# 5. F measure(f1 score)
r <- tab[2,2] / sum(tab[2,])
p <- tab[2,2] / sum(tab[,2])

F1 <- 2*((r*p) / (r+p))
F1





