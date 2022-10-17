##################################
###  Part-I ~ Part-III 총정리  ###
##################################

setwd("C:/ITWILL/2_Rwork/Part-IV")
election <- read.csv("election_2012.csv") # 2012년도 미국 대통령 선거 후원금 현황 
election <- read.csv(file.choose())
str(election) # 'data.frame':	1001731 obs. of  16 variables:
# 2.cand_id : 대선후보자
# 3.cand_nm : 대선 후보자 이름
# 4.contbr_nm  : 후원자 이름 
# 9.contbr_occupation : 후원자 직업군 
# 10.contb_receipt_amt : 후원금


### 전처리 ###
# [문제1] 위 7개 칼럼으로 data.frame 생성 clean_election2 변수에 저장하시오.
library(dplyr)
clean_election2 <- election %>% select(c(cand_id, cand_nm, contbr_nm, contbr_occupation, contb_receipt_amt))
head(clean_election2)

# [문제2] 직업군 칼럼을 대상으로 NA를 포함하는 관측치를 제외하여 clean_election2 변수에 저장하시오.
# <조건> 전처리 전과 후 관측치 수의 차이는? 
#   힌트) subset(), is.na() 함수 이용 
dim(clean_election2) # 1001731       5
clean_election2 <- clean_election2 %>% filter(!is.na(contbr_occupation))
dim(clean_election2) # 1001603       5
an1 <- 1001731 - 1001603
an1 # 128

# [문제3] 5만개 관측치만 샘플링하여 clean_election2 변수에 저장하시오.
#  힌트) sample() 함수 이용
a <- sample(nrow(clean_election2), 50000)
clean_election2 <- clean_election2[a,]
dim(clean_election2) # 50000     5

# [문제4] 'Romney, Mitt'와 'Obama, Barack' 후보자별 서브셋을 작성하여 romney, obama 변수에 저장하시오
romney <- clean_election2 %>% filter(cand_nm == "Romney, Mitt")
dim(romney) # 5365    5
obama <- clean_election2 %>% filter(cand_nm == 'Obama, Barack')
dim(obama) # 29697     5

# [문제5] romney, obama 변수를 대상으로 병합하여 obama_romney 변수에 저장하시오.
# 힌트) rbind()
obama_romney <- rbind(romney, obama)
dim(obama_romney) # 35062     5

## 교차분석 ## 

# [문제6] 후원자의 직업군과 대통령 당선 유무에 따라서 다음과 같이 파생변수를 만드시오.
# <조건1> 대상 변수 : obama_romney

# <조건2> 다음과 같은 후원자의 직업군을 job1, job2, job3로 리코딩하여 contbr_occupation2 칼럼 저장 
# job1 : INVESTOR - 투자자, EXECUTIVE - 경영진, PRESIDENT  - 회장 
# job2 : LAWYER - 변호사, PHYSICIAN  - 내과의사, ATTORNEY   - 변호사
# job3 : RETIRED  - 퇴직자, HOMEMAKER  - 주부

# job1 리코딩 
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='INVESTOR'] <- 'job1'
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='EXECUTIVE'] <- 'job1'
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='PRESIDENT'] <- 'job1'

# job2 리코딩 
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='LAWYER'] <- 'job2'
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='PHYSICIAN'] <- 'job2'
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='ATTORNEY'] <- 'job2'

# job3 리코딩 
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='RETIRED'] <- 'job3'
obama_romney$contbr_occupation2[obama_romney$contbr_occupation=='HOMEMAKER'] <- 'job3'

# <조건3> contbr_occupation2 칼럼에 NA를 포함한 관측( 제거 후 서브셋 작성(obama_romney2 저장)  
table(is.na(obama_romney$contbr_occupation2))

# FALSE  TRUE 
# 13042 22020

obama_romney2 <- obama_romney %>% filter(!is.na(contbr_occupation2))
table(is.na(obama_romney2$contbr_occupation2))
# FALSE 
# 13042 

# <조건4> obama_romney2 변수를 대상으로 cand_nm 칼럼이 'Obama, Barack'이면 '당선', 
#               'Romney, Mitt'이면 '낙선'으로 파생변수를 생성하여 cand_pass 칼럼 추가
obama_romney2 <- obama_romney2 %>% mutate(cand_pass = 
                                            ifelse(cand_nm == "Obama, Barack", "당선", "낙선"))
table(obama_romney2$cand_pass)

# [문제7]  [문제6]에서 만든 파생변수(직업유형과 당선유무)를 이용하여 교차분할표를 작성하시오.  
library(gmodels)
CrossTable(obama_romney2$contbr_occupation2, obama_romney2$cand_pass)

## chisquare 검정 ##

# [문제8] 후원자의 직업군과 대통령 당선 유무 여부와 관련성이 있는가를 검정하시오.
# <조건1> 대상 변수 : obama_romney2
# <조건2> 귀무가설과 대립가설 수립 
# <조건3> 변수 모델링 : x변수(contbr_occupation2), y변수(cand_pass)
# <조건4> 검정결과 해석

# 귀무가설 : 직업의 유형과 대통령 당선과 관련성이 없다.
# 대립가설 : 직업의 유형과 대통령 당선과 관련성이 있다.
chisq.test(obama_romney2$contbr_occupation2, obama_romney2$cand_pass)
# p-value < 2.2e-16 < 0.05
# [해설] 귀무가설 기각 대립가설 채택

## lattice 패키지 ## 

# [문제9] lattice 패키지의 densityplot()함수를 이용하여 후원자의 직업유형별로 다음과 같이 후원금을 시각화하시오.
# <조건1> 대상 변수 : obama_romney2
# <조건2> 후원금이 300달러 ~ 3000달러 사이의 관측치 선정 -> obama_romney3 변수 저장  
# <조건3> obama_romney3 변수 대상 밀도그래프 시각화(x축 : 후원금, 조건 : 당선유무, 그룹 : 직업유형) 
library(lattice)
# obama_romney3
str(obama_romney2)
obama_romney3 <- obama_romney2 %>% filter(contb_receipt_amt >= 300 & contb_receipt_amt <= 3000)
densityplot(~ contb_receipt_amt | factor(cand_pass), data = obama_romney2, groups = contbr_occupation2)
range(obama_romney3$contb_receipt_amt) # 300 3000

## 평균차이 검정 ## 

# [문제10] romney와 obama 후보자를 대상으로 후원자의 직업군이 'RETIRED'인 후원금에 차이가 있는지 검정하시오.
# <조건1> 대상 변수 : obama_romney3
# <조건2> 두집단 평균차이 검정
# y(종속변수) =  후원금(contb_receipt_amt), x(독립변수) = 후보자(cand_nm)
obama_romney4 <- obama_romney3 %>% filter(contbr_occupation == "RETIRED")
str(obama_romney4)
boxplot(obama_romney4$contb_receipt_amt)
bartlett.test(contb_receipt_amt ~ cand_nm, data = obama_romney4 )
# p-value = 6.659e-05 < 0.05 : 동질성 x
# kruskal.test() 사용
kruskal.test(contb_receipt_amt ~ cand_nm, data = obama_romney4 )
# p-value < 2.2e-16 : 비모수 통계 사용시 차이가 있음
