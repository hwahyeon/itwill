####################################
## 제16-2장 RandomForest 연습문제 
####################################

# 01. 400개의 Tree와 4개의 분류변수를 파라미터로 지정하여 모델을 생성하고, 
#       분류정확도를 구하시오.
#  조건> 1,2,22,23 칼럼을 제외하여 데이터 셋 구성 

weatherAUS = read.csv(file.choose()) # weatherAUS.csv 
weatherAUS = weatherAUS[ ,c(-1,-2, -22, -23)]
str(weatherAUS)
# 'data.frame':	36881 obs. of  20 variables:
table(weatherAUS$RainTomorrow, useNA = 'ifany')
#    No   Yes  <NA> 
# 27732  8529   620 <- 결측치 

sqrt(19) # 4.35
model <- randomForest(RainTomorrow ~ ., 
                      data = weatherAUS,
                      ntree=400, mtry = 4,
                      importance = T,
                      na.action = na.omit) # 결측치 제거 
model
# Number of trees: 400
# No. of variables tried at each split: 4
# 
# OOB estimate of  error rate: 14.08%
# Confusion matrix:
#   No  Yes class.error
# No  12573  853  0.06353344
# Yes  1593 2359  0.40308704


# 02. 변수의 중요도 평가를 통해서 가장 중요한 변수를 확인하고, 시각화 하시오. 
varImpPlot(model)
