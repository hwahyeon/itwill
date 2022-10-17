# chap13_Ttest_Anova(연습문제)

#############################################
# 추론통계분석 - 1-1. 단일집단 비율차이 검정
#############################################

# 01. 중소기업에서 생산한 HDTV 판매율을 높이기 위해서 프로모션을 진행한 결과 
# 기존 구매비율 15% 보다 향상되었는지를 각 단계별로 분석을 수행하여 검정하시오.

#연구가설(H1) : 기존 구매비율과 차이가 있다.
#귀무가설(H0) : 기존 구매비율과 차이가 없다.


#조건) 구매여부 변수 : buy (1: 구매하지 않음, 2: 구매)

#(1) 데이터셋 가져오기
setwd("C:/ITWILL/2_Rwork/Part-III")
hdtv <- read.csv("hdtv.csv", header=TRUE)

# (2) 빈도수와 비율 계산
buy <- hdtv$buy
table(buy)
# 1  2  (1:구매하지 않음, 2:구매)
# 40 10 
prop.table(table(buy))
# 1   2  
# 0.8 0.2 (구매 비율)

# (3)가설검정 : (x=성공횟수, n=시행횟수, p=구매비율)
binom.test(x=10, n=50, p=0.15,
           alternative = 'two.sided', conf.level = 0.95)
# p-value = 0.321
# 95 percent confidence interval : 95% 신뢰수준
# 0.1003022 ~ 0.3371831 채택역
# probability of success 
#                    0.2 -> 실제 성공 확률
# [해설] 구매비율은 15%를 넘지 못한다.

#################################################
# 추론통계학 분석 - 1-2. 단일집단 평균차이 검정
#################################################

# 02. 우리나라 전체 중학교 2학년 여학생 평균 키가 148.5cm로 알려져 있는 상태에서 
# A중학교 2학년 전체 500명을 대상으로 10%인 50명을 표본으로 선정된 데이터 셋을 이용하여
# 모집단의 평균과 차이가 있는지를 각 단계별로 분석을 수행하여 검정하시오.

#(1) 데이터셋 가져오기
sheight<- read.csv("student_height.csv", header=TRUE)

# (2) 기술통계량 평균 계산
height <- sheight$height
length(height) #50
mean(height) #149.4

# (3) 정규성 검정
shapiro.test(height)
#p-value = 0.0001853
hist(height, freq = T)

# (4) 가설검정 
wilcox.test(height, mu=148.5)
# V = 826, p-value = 0.067 >= 0.05


#################################################
# 추론통계학 분석 - 2-1. 두집단 비율 차이 검정
#################################################

# 03. 대학에 진학한 남학생과 여학생을 대상으로 진학한 대학에 
# 대해서 만족도에 차이가 있는가를 검정하시오.

# 힌트) 두 집단 비율 차이 검정
#  조건) 파일명 : two_sample.csv, 변수명 : gender(1,2), survey(0,1)
# gender : 남학생(1), 여학생(2)
# survey : 불만(0), 만족(1)
# prop.test('성공횟수', '시행횟수')


##################################################
# 추론통계학 분석 - 2-2. 두집단 평균 차이 검정
##################################################

# 04. 교육방법에 따라 시험성적에 차이가 있는지 검정하시오.

#힌트) 두 집단 평균 차이 검정
#조건1) 파일 : twomethod.csv
#조건2) 변수 : method : 교육방법, score : 시험성적
#조건3) 모델 : 교육방법(명목)  ->  시험성적(비율)
#조건4) 전처리 : 결측치 제거 : 평균으로 대체 


#############################################
# 추론통계분석 - 2-1. 두집단 비율차이 검정
#############################################

# 1. 실습데이터 가져오기
data <- read.csv("two_sample.csv", header=TRUE)
data
head(data) # 변수명 확인


# 2. 두 집단 subset 작성
data$method # 1, 2 -> 노이즈 없음
data$survey # 1(만족), 0(불만족)

# - 데이터 정체/전처리
x<- data$method # 교육방법(1, 2) -> 노이즈 없음
y<- data$survey # 만족도(1: 만족, 0:불만족)
x;y

# 1) 데이터 확인
# 교육방법 1과 2 모두 150명 참여
table(x) # 1 : 150, 2 : 150
# 교육방법 만족/불만족
table(y) # 0 : 55, 1 : 245

# 2) data 전처리 & 두 변수에 대한 교차분석
table(x, y, useNA="ifany") 
# 1  40 110 110명이 만족
# 2  15 135 135명이 만족

# 3. 두집단 비율차이검증 - prop.test(성공횟수, 시행횟수)

# 양측가설 검정
prop.test(c(110,135), c(150, 150)) # 14와 20% 불만족율 기준 차이 검정
prop.test(c(110,135), c(150, 150), alternative="two.sided", conf.level=0.95)

# # 방향성이 있는 대립가설 검정 : PT > CODE라고 했을 때 세워지는 가설
prop.test(c(110,135), c(150, 150), alternative="greater", conf.level=0.95)
# p-value = 0.9998 -> PT가 CODE보다 훨씬 만족도가 높다는 것에 99%가 옳지 않다는 뜻

#방향성이 있는 대립가설 검정
prop.test(c(110,135), c(150, 150), alternative="less", conf.level=0.95)
# p-value = 0.0001711 즉, PT < CODE


#############################################
# 추론통계분석 - 2-2. 두집단 평균차이 검정
#############################################

# 1. 실습파일 가져오기
data <- read.csv("two_sample.csv")
data 
head(data) #4개 변수 확인
summary(data) # score - NA's : 73개

# 2. 두 집단 subset 작성(데이터 정제,전처리)
#result <- subset(data, !is.na(score), c(method, score))
dataset <- data[c('method', 'score')]
table(dataset$method)


# 3. 데이터 분리
# 1) 교육방법 별로 분리
method1 <- subset(dataset, method==1)
method2 <- subset(dataset, method==2) # 교육방법이 2인 경우의 서브셋.

# 2) 교육방법에서 점수 추출
method1_score <- method1$score
method2_score <- method2$score

# 3) 기술통계량 
length(method1_score); # 150
length(method2_score); # 150
mean(method1_score, na.rm = T) #5.556881
mean(method2_score, na.rm = T) #5.80339

# 4. 분포모양 검정 : 두 집단의 분포모양 일치 여부 검정
var.test(method1_score, method2_score) 
# 동질성 분포 : t.test()
# 비동질성 분포 : wilcox.test()
# p-value = 0.3002


# 5. 가설검정 - 두집단 평균 차이검정
t.test(method1_score, method2_score)
t.test(method1_score, method2_score, alter="two.sided", conf.int=TRUE, conf.level=0.95)
# p-value = 0.0411 - 두 집단간 평균에 차이가 있다.

# # 방향성이 있는 연구가설 검정 : method1 > method2
t.test(method1_score, method2_score, alter="greater", conf.int=TRUE, conf.level=0.95)
# p-value = 0.9794

# # 방향성이 있는 연구가설 검정 : method1 < method2
t.test(method1_score, method2_score, alter="less", conf.int=TRUE, conf.level=0.95)
# p-value = 0.02055 < 0.05


################################################
# 추론통계분석 - 2-3. 대응 두 집단 평균차이 검정
################################################
# 조건 : A집단  독립적 B집단 -> 비교대상 독립성 유지
# 대응 : 표본이 짝을 이룬다. -> 한 사람에게 2가지 질문
# 사례) 다이어트식품 효능 테스트 : 복용전 몸무게 -> 복용후 몸무게 

# 1. 실습파일 가져오기
getwd()
setwd("c:/Rwork/Part-III")
data <- read.csv("paired_sample.csv", header=TRUE)

# 2. 두 집단 subset 작성

# 1) 데이터 정제
#result <- subset(data, !is.na(after), c(before,after))
dataset <- data[ c('before',  'after')]
dataset

# 2) 적용전과 적용후 분리
before <- dataset$before# 교수법 적용전 점수
after <- dataset$after # 교수법 적용후 점수
before; after

# 3) 기술통계량 
length(before) # 100
length(after) # 100
mean(before) # 5.145
mean(after, na.rm = T) # 6.220833 -> 1.052  정도 증가


# 3. 분포모양 검정 
var.test(before, after, paired=TRUE) 
# 동질성 분포 : t.test()
# 비동질성 분포 : wilcox.test()

# 4. 가설검정
t.test(before, after, paired=TRUE) # p-value < 2.2e-16 

# 방향성이 있는 연구가설 검정 
t.test(before, after, paired=TRUE,alter="greater",conf.int=TRUE, conf.level=0.95) 
#p-value = 1 -> x을 기준으로 비교 : x가 y보다 크지 않다.

#  방향성이 있는 연구가설 검정
t.test(before, after, paired=TRUE,alter="less",conf.int=TRUE, conf.level=0.95) 
# p-value < 2.2e-16 -> x을 기준으로 비교 : x가 y보다 적다.





# 05. iris 데이터셋을 이용하여 다음과 같이 분산분석(anova)을 수행하시오.
#즉, 집단과 집단 사이의 평균의 차이가 있냐 없냐를 보려는 것.
# 독립변수 : Species(집단변수)
# 종속변수 : 전제조건을 만족하는 변수(1번 칼럼 ~ 4번 칼럼) 선택
# 1번부터 4번가지 전제조건을 만족하는 것을 추려서 종속변수로 삼을 것.
# 분산분석 해석 -> 사후검정 해석

str(iris)
# $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
# $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
# $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
# $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
# 4개 중 1개 선택

# 1. 동질성 검정 : 전제조건
bartlett.test(iris$Sepal.Length, iris$Species)
# p-value = 0.0003345
bartlett.test(iris$Sepal.Width, iris$Species)
# p-value = 0.3515

# 2. 변수 선택 (변수 선택은 아주 중요함!)
x <- iris$Sepal.Width
y <- iris$Species

# 3. 분산분석
result <- aov(Sepal.Width ~ Species, data = iris)
summary(result)
#              Df Sum Sq Mean Sq F value Pr(>F)    
# Species       2  11.35   5.672   49.16 <2e-16 ***
#  [해설] 매우 유의미한 수준(***)에서 적어도 한 집단의 평균 차이

# 4. 사후검정
TukeyHSD(result)
# $Species
#                        diff         lwr        upr     p adj
# versicolor-setosa    -0.658 -0.81885528 -0.4971447 0.0000000
# virginica-setosa     -0.454 -0.61485528 -0.2931447 0.0000000
# virginica-versicolor  0.204  0.04314472  0.3648553 0.0087802

# [해설]
# 95% 신뢰수준에서 3집단(꽃의 종별) 모두 평균 차이(p adj<0.05)
# 꽃잎의 넓이(Sepal.Width) 변수는 versicolor와 setosa 집단
# 가장 평균 차이를 보인다.

plot(TukeyHSD(result))
# 3집단 모두 신뢰구간이 0을 포함하지 않는다. 즉, 세 집단 모두 평균의 차이가 있다.

#plot은 만능함수
methods(plot) #패키지를 인스톨할 때마다 계속 늘어남.
#plot.TukeyHSD* 분산검정 사후검정을 plot으로 시각화 할 수 있다는 의미.

library(dplyr) # df %>% function()

iris %>% group_by(Species) %>% summarise(age = mean(Sepal.Width))
#   Species      age
#    <fct>      <dbl>
# 1 setosa      3.43
# 2 versicolor  2.77
# 3 virginica   2.97

2.77 - 3.43 #versicolor-setosa








