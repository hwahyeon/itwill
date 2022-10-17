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


# 02. 변수의 중요도 평가를 통해서 가장 중요한 변수를 확인하고, 시각화 하시오. 
varImpPlot(model)
# 중요한 변수는 그래프 위쪽에 있다.