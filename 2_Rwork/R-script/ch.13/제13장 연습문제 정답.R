#################################
## <제13장 연습문제>
################################# 

# 01. 중소기업에서 생산한 HDTV 판매율을 높이기 위해서 프로모션을 진행한 결과 
#  기존 구매비율  보다 15% 향상되었는지를 각 단계별로 분석을 수행하여 검정하시오.
#연구가설(H1) : 기존 구매비율과 차이가 있다.
#귀무가설(H0) : 기존 구매비율과 차이가 없다.

#조건) 구매여부 변수 : buy (1: 구매하지 않음, 2: 구매)

# 단계1 : 데이터셋 가져오기
hdtv <- read.csv("hdtv.csv", header=TRUE)

# 단계2 :  빈도수와 비율 계산
summary(hdtv) 
length(hdtv$buy) # 50개

library(prettyR) # freq() 함수 사용
freq(hdtv$buy) # 1:40, 2:10
table(hdtv$buy)
table(hdtv$buy, useNA="ifany") # NA 빈도수 표시 
# 1  2   buy (1: 구매하지 않음, 2: 구매)
#40 10  


# 단계3 : 가설검정
binom.test(10, 50, p=0.15) #15% 비교 -> p-value = 0.321
binom.test(10, 50, p=0.15, alternative="two.sided", conf.level=0.95)
# 해설 : 기존 구매비율(20%)과 차이가 없다. -> 귀무가설 채택

# [실습] 방향성이 있는 연구가설 검정
binom.test(10, 50, p=0.15, alternative="greater", conf.level=0.95) # p-value = 0.2089
binom.test(10, 50, p=0.15, alternative="less", conf.level=0.95) # p-value = 0.8801
# <해설> 방향성이 있는 연구가설은 모두 기각된다.


# 02. 우리나라 전체 중학교 2학년 여학생 평균 키가 148.5cm로 알려져 있는 상태에서  
# A중학교 2학년 전체 500명을 대상으로 10%인 50명을 표본으로 선정하여 표본평균신장을 
# 계산하고 모집단의 평균과 차이가 있는지를 각 단계별로 분석을 수행하여 검정하시오.

# 단계1 : 데이터셋 가져오기
stheight<- read.csv("student_height.csv", header=TRUE)
stheight
height <- stheight$height
head(height)

# 단계2 : 기술통계량/결측치 확인
length(height) #50
summary(height) # 149.4 

x1 <- na.omit(height)
x1 # 정제 데이터 
mean(x1) # 149.4 -> 평균신장

# 단계3 : 정규성 검정
shapiro.test(x1) # p-value = 0.0001853 -> 정규분포 아님
# 정규분포(모수검정) - t.test()
# 비정규분포(비모수검정) - wilcox.test()

# 단계4 : 가설검정 - 양측검정
wilcox.test(x1, mu=148.5) # p-value = 0.067
wilcox.test(x1, mu=148.5, alter="two.side", conf.level=0.95) # p-value = 0.067
## p-value = 0.067 : 148.5와 차이가 없다.


# 03. 대학에 진학한 남학생과 여학생을 대상으로 진학한 대학에 대해서
# 만족도에 차이가 있는가를 검정하시오.

# 힌트) 두 집단 비율 차이 검정
#  조건) 파일명 : two_sample.csv, 변수명 : gender(1,2), survey(0,1)
# gender : 남학생(1), 여학생(2)
# survey : 불만(0), 만족(1)
# prop.test('성공횟수', '시행횟수')

# 단계1 : 실습데이터 가져오기
getwd()
data <- read.csv("two_sample.csv", header=TRUE)
data
head(data) # 변수명 확인

# 단계2 : 두 집단 subset 작성
data$gender # 1, 2 -> 노이즈 없음
data$survey # 1(만족), 0(불만족)
# - 데이터 정체/전처리
x<- data$gender # 1, 2 -> 노이즈 없음
y<- data$survey # 1(만족), 0(불만족)

x;y

# 1) 데이터 확인
# 성별 구분 : 300명
table(x) # 1 : 174, 2 : 126
# 대학진학 만족도
table(y) # 0 : 55, 1 : 245

# 2) data 전처리 & 기술통계량 -> 빈도수 -> 정규성 검정 필요 없음
table(x) # 174, 126
table(y) # 55, 245

#- 두 변수에 대한 교차분석
table(x, y, useNA="ifany") # 결측치 까지 출력
#######################
#  y
#x    0    1  
#  1  36   138  -> 남학생 - 138 만족(174)
#  2  19   107  -> 여학생 - 107 만족(126)
######################

# 단계3 : 두집단 비율차이검증 : prop.test()
help(prop.test) # prop.test(x,n,p, alternative, conf.level, correct)
prop.test(c(138,107),c(174,126)) # 남학생과 여학생의 만족도 차이 검정
# p-value = 0.2765

#sample estimates:
#  prop 1    prop 2 
#0.7931034 0.8492063 

prop.test(c(138,107),c(174,126), alternative="two.sided", conf.level=0.95)
# p-value = 0.2765 - 남학생과 여학생의 만족도에 차이가 없다.

# 04. 교육방법에 따라 시험성적에 차이가 있는지 검정하시오.

#힌트) 두 집단 평균 차이 검정
#조건1) 파일 : twomethod.csv
#조건2) 변수 : method : 교육방법, score : 시험성적
#조건3) 모델 : 교육방법(명목)  ->  시험성적(비율)
#조건4) 전처리 : 결측치 제거 : 평균으로 대체 

#단계1. 실습파일 가져오기
Data <- read.csv("twomethod.csv", header=TRUE)
head(Data) #3개 변수 확인 -> id method score

#단계2. 두 집단 subset 작성(데이터 정제, 전처리)
# 데이터 전처리(score의 NA 처리)
Data <- Data[c('method', 'score')]
Data$score <- ifelse(is.na(Data$score), mean(Data$score, na.rm=T), Data$score)
summary(Data)

#단계3. 데이터 분리
# 1) 교육방법별로 분리
a <- subset(Data,method==1)
b <- subset(Data,method==2)

# 2) 교육방법에서 영업실적 추출
a1 <- a$score
b1 <- b$score

# 3) 기술통계량 
length(a1); # 22
length(b1); # 35

#단계4 : 분포모양 검정
var.test(a1, b1) # p-value = 0.7302 : 차이가 없다. 
#<해설> 동질성 분포와 차이가 없다. 모수검정 방법 수행
# 동질성 분포 :　t.test()
# 비동질성 분포 : wilcox.test()

#단계5: 가설검정
t.test(a1, b1) # p-value = 1.859e-06

t.test(a1, b1, alter="greater", conf.int=TRUE, conf.level=0.95) # p-value = 1

t.test(b1, a1, alter="greater", conf.int=TRUE, conf.level=0.95) # p-value=6.513e-07
#<해설> b1 교육 방법이 a1 교육방법 보다 시험성적이 더 좋다.



