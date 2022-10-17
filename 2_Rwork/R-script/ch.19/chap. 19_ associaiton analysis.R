# chap. 19_ associaiton analysis

###################################################
# 연관분석(Association Analysis)
###################################################
# 군집분석 : 그룹핑 하는 작업
#     ↓
# 연관분석 : 그룹에 대한 특성 분석(장바구니 분석)
#--------------------------------------------------

# 특징
# - 데이터베이스에서 사건의 연관규칙을 찾는 무방향성 데이터마이닝 기법                                            
# - 무방향성(x -> y변수 없음) -> 비지도 학습에 의한 패턴 분석 방법
# - 사건과 사건 간 연관성(관계)를 찾는 방법(예:기저귀와 맥주)
# - A와 B 제품 동시 구매 패턴(지지도)
# - A 제품 구매 시 B 제품 구매 패턴(신뢰도)

# 예) 장바구니 분석 : 장바구니 정보를 트랜잭션(상품 거래 정보)이라고 하며,
# 트랜잭션 내의 연관성을 살펴보는 분석기법
# 분석절차 : 거래내역 -> 품목 관찰 -> 상품 연관성에 대한 규칙(Rule) 발견

# 활용분야
# - 대형 마트, 백화점, 쇼핑몰 등에서 고객의 장바구니에 들어있는 품목 간의 
#   관계를 탐구하는 용도
# ex) 고객들은 어떤 상품들을 동시에 구매하는가?
#   - 맥주를 구매한 고객은 주로 어떤 상품을 함께 구매하는가?

# - 대형 마트,백화점, 쇼핑몰 판매자 -> 고객 대상 상품추천, 마케팅
# 1) 고객 대상 상품추천 및 상품정보 발송  
#     -> ex) A고객에 대한 B 상품 쿠폰 발송
# 2) 텔레마케팅를 통해서 패키지 상품 판매 기획 및 홍보 
# 3) 상품 진열 및 show window 상품 display

# 연관규칙(Association Rule)
# - 상업 데이터베이스에서 가장 흔히 쓰이는 도구로, 
# - 어떤 사건이 얼마나 자주 동시에 발생하는가를 표현하는 규칙(조건)

#########################################
# 1. 연관규칙 평가 척도
#########################################

# 연관규칙의 평가 척도
# 1. 지지도(support) : 전체자료에서 A를 구매한 후 B를 구매하는 거래 비율 
#  A->B 지지도 식 
#  -> A와 B를 포함한 거래수 / 전체 거래수
#  -> n(A, B) : 두 항목(A,B)이 동시에 포함되는 거래수
#  -> n : 전체 거래 수

# 2. 신뢰도(confidence) : A가 포함된 거래 중에서 B를 포함한 거래의 비율(조건부 확률)
# A->B 신뢰도 식
#  -> A와 B를 포함한 거래수 / A를 포함한 거래수

# 지지도는 순서를 고려하지 않고, 신뢰도는 순서를 고려한다.
# 영수증만보고 신뢰도를 구하긴 어렵다. -> 조건부확률 적용
# 즉 장바구니에서의 순서는 큰 의미가 없다. 순서를 고려한다는 것은
# 조건부확률을 적용한다는 의미인 셈이다.

# 3. 향상도(Lift) : 하위 항목들이 독립에서 얼마나 벗어나는지의 정도를 측정한 값
# 향상도 식
#  -> 신뢰도 / B가 포함될 거래율

# <지지도와 신뢰도 예시>
# t1 : 라면,맥주,우유
# t2 : 라면,고기,우유
# t3 : 라면,과일,고기
# t4 : 고기,맥주,우유
# t5 : 라면,고기,우유
# t6 : 과일,우유

#    A -> B             지지도         신뢰도          향상도
#  맥주 -> 고기         1/6=0.166       1/2=0.5      0.5/0.66(4/6)=0.75
# 맥주가 포함된 트랜젝션은 2개
# 고기가 포함된 트랜젝션은 4개 즉, 총 6개 중에 4개 (4/6)
# 4/6를 향상도의 분모로 둔다.
# 맥주와 고기가 같이 있는 트랜젝션은 1개

#    A -> B             지지도         신뢰도          향상도
# 라면,우유-> 맥주        1/6             1/3           0.33/(2/6)    
# 먼저 라면과 우유를 선행사건으로 보고 맥주를 후행사건으로 봤을 때,
# 이것의 순서를 비교하지 않고 보는 비율을 지지도라고 보는 것이다.
# 지지도는 전체 트랜젝션의 개수는 6개로 분모가 6이 된다.
# 라면과 우유와 맥주가 동시에 포함된(순서가 상관없으니까)
# 트랜젝션의 개수는 1개(t1)이니 지지도는 1/6이다.
# 신뢰도는 모두 포함된 것을 분자/
# 분모는 A가 포함될 조건부확률(A를 포함한 거래수)이므로
# A가 포함된 것은 3개니까 신뢰도는 1/3이 된다.
# 후행사건인 맥주를 포함한 트랜젝션은 2개니 확률은 2/6이다. 
# 향상도가 1이상이면 상관도가 어느정도 있다고 보는 것이다.


## 연관성 규칙 분석을 위한 패키지
install.packages("arules") # association Rule
# read.transactions(),  apriori(), Adult 데이터셋 제공
library(arules) #read.transactions()함수 제공


# 1. transaction 객체 생성(파일 이용)
setwd("C:\\ITWILL\\2_Rwork\\Part-IV")
tran<- read.transactions("tran.txt", format="basket", sep=",")
tran

#read.transaction 읽어와서 트랜젝션 형태로 만들어주는 것.
# transactions in sparse format with
# 6 transactions (rows) and
# 5 items (columns)

# 2. transaction 데이터 보기
inspect(tran) #inspect로 트랜젝션을 볼 수 있다.

# 3. rule 발견(생성) - 지지도,신뢰도 = 0.1
# apriori(트랜잭션 data, parameter=list(supp, conf))

# 연관성 규칙 평가 척도 - 지지도와 신뢰도
rule<- apriori(tran, parameter = list(supp=0.3, conf=0.1))
# 16 rule 16개의 룰이 만들어졌다.
# 지지도는 0.3, 신뢰도는 0.1로 해서 룰을 만들겠다는 의미
rule<- apriori(tran, parameter = list(supp=0.1, conf=0.1))
# 35 rule 룰이 더 늘어난 것을 볼 수 있다.
# 즉 평가척도들이 룰의 갯수를 정하는 것.
# 숫자가 클수록 필터링을 많이하는 것이니 룰의 개수가 줄어듦.
rule
inspect(rule) # 룰(규칙) 보기
# 6번째 사건부터 시작한다고 보면 된다.


# 지지도, 신뢰도, maxlen 인수  
help("apriori") # support 0.1, confidence 0.8, and maxlen 10 
# maxlen 최대 길이를 의미함.
# 선행사건과 후행사건의 아이템을 다 더해서 10개 이하인 것만 보겠다는 의미.
# ex) {고기,과일} => {라면} 이 경우엔 3개이다.

rule <- apriori(tran) 
rule<- apriori(tran, parameter = list(supp=0.1, conf=0.8, maxlen=10)) 
inspect(rule) 

#########################################
# 2. 트랜잭션 객체 생성 
#########################################

#형식)
#read.transactions(file, format=c("basket", "single"),
#      sep = NULL, cols=NULL, rm.duplicates=FALSE,encoding="unknown")
#------------------------------------------------------
#file : file name
#format : data set의 형식 지정(basket 또는 single)
# -> single : 데이터 구성(2개 칼럼) -> transaction ID에 의해서 상품(item)이 대응된 경우
# 상품과 아이디가 일대일 대응일 때.
# -> basket : 데이터 셋이 여러개의 상품으로 구성 -> transaction ID 없이 여러 상품(item) 구성
#sep : 상품 구분자
#cols : single인 경우 읽을 컬럼 수 지정, basket은 생략(transaction ID가 없는 경우)
#rm.duplicates : 중복 트랜잭션 항목 제거
#encoding : 인코딩 지정
#------------------------------------------------------

# (1) single 트랜잭션 객체 생성
## read demo data - sep 생략 : 공백으로 처리, single인 경우 cols 지정 
# format = "single" : 1개의 transaction id에 의해서 item이 연결된 경우 

# 하나의 트랜젝션은 한 사람이 구매한 아이템들의 정보만 쭉 해놓은 것
# (사과, 바나나, 복숭아) 이런 것은 basket

setwd("C:\\ITWILL\\2_Rwork\\Part-IV")
stran <- read.transactions("demo_single",format="single",cols=c(1,2)) 
inspect(stran)

#       items       transactionID
# [1] {item1}       trans1       
# [2] {item1,item2} trans2   

# <실습> 중복 트랜잭션 객체 생성
stran2<- read.transactions("single_format.csv", format="single", sep=",", 
                           cols=c(1,2), rm.duplicates=T)
stran2
# transactions in sparse format with
# 248 transactions (rows) and
# 68 items (columns)

summary(stran2) # 248개 트랜잭션에 대한 '기술통계' 제공
# transactions as itemMatrix in sparse format with
# 248 rows (elements/itemsets/transactions) and
# 68 columns (items) and a density of 0.06949715 

# most frequent items: 빈도 아이템 수
# 10001519 10003364 10003373 10093119 10003332  (Other) 
#      186      183       91       84       71      557 

# element (itemset/transaction) length distribution:
# sizes
#  1   2   3   4   5   6   7  8 
# 12  25  16  20 119  12  37  7
#즉, 1,2,3,4,5,6,7,8은 아이템의 수, 아래는 트랜젝션에서 출현한 빈도수를 말한다.
# 6개의 상품에 의해 만들어진 것은 12개의 트렌젝션이 만들어진다는 의미.

# 트랜잭션 보기
inspect(stran2) # 248 트랜잭션 확인 


# 규칙 발견
astran2 <- apriori(stran2) # supp=0.1, conf=0.8와 동일함 
#astran2 <- apriori(stran2, parameter = list(supp=0.1, conf=0.8))
astran2 # set of 102 rules
attributes(astran2)
inspect(astran2)

# 향상도가 높은 순서로 정렬 
inspect(sort(astran2, by="lift"))
inspect(head(sort(astran2, by="lift")))

# (2) basket 트랜잭션 데이터 가져오기
btran <- read.transactions("demo_basket",format="basket",sep=",") 
inspect(btran) # 트랜잭션 데이터 보기


##############################################
# 3. 연관규칙 시각화(Adult 데이터 셋 이용)
##############################################

data(Adult) # arules에서 제공되는 내장 데이터 로딩
str(Adult) # Formal class 'transactions' , 48842(행)
Adult

attributes(Adult)# 트랜잭션의 변수와 범주 보기
################ Adult 데이터 셋 #################
# 32,000개의 관찰치와 15개의 변수로 구성되어 있음
# 종속변수에 의해서 년간 개인 수입이 5만달러 이상 인지를
# 예측하는 데이터 셋으로 transactions 데이터로 읽어온
# 경우 48,842행과 115 항목으로 구성된다.
##################################################


# [data.frame 형식으로 보기] - 트랜잭션 데이터를 데이터프레임으로 변경 
adult<- as(Adult, "data.frame") # data.frame형식으로 변경 
str(adult) # 'data.frame':	48842 obs. of  2 variables:
head(adult) # 칼럼 내용 보기 

# 요약 통계량
summary(Adult)

#---------------------------------------------------------------
# 신뢰도 80%, 지지도 10%이 적용된 연관규칙 6137 발견   
#----------------------------------------------------------------
ar<- apriori(Adult, parameter = list(supp=0.1, conf=0.8))
ar1<- apriori(Adult, parameter = list(supp=0.2)) # 지도도 높임
ar2<- apriori(Adult, parameter = list(supp=0.2, conf=0.95)) # 신뢰도 높임
ar3<- apriori(Adult, parameter = list(supp=0.3, conf=0.95)) # 신뢰도 높임
ar4<- apriori(Adult, parameter = list(supp=0.35, conf=0.95)) # 신뢰도 높임
ar5<- apriori(Adult, parameter = list(supp=0.4, conf=0.95)) # 신뢰도 높임
# -> 룰이 점점 줄어드는 것을 볼 수 있다. 

# 결과보기
inspect(head(ar5)) # 상위 6개 규칙 제공 -> inspect() 적용

# confidence(신뢰도) 기준 내림차순 정렬 상위 6개 출력
inspect(head(sort(ar5, decreasing=T, by="confidence")))

# lift(향상도) 기준 내림차순 정렬 상위 6개 출력
inspect(head(sort(ar5, by="lift"))) 


## 연관성 규칙에 대한 데이터 시각화를 위한 패키지
install.packages("arulesViz") 
library(arulesViz) # rules값 대상 그래프를 그리는 패키지

plot(ar3) # 지지도(support), 신뢰도(conf) , 향상도(lift)에 대한 산포도
plot(ar4, method="graph") #  연관규칙 네트워크 그래프
# 각 연관규칙 별로 연관성 있는 항목(item) 끼리 묶여서 네트워크 형태로 시각화

# 원이 클수록 지지도가 큰 것.
# 색상의 농도가 진할수록 상품과 상품의 상관성이 높다
# 선행사건이 다 다르지만 특정 후행사건으로 모인다면 중심어로 생각할만하다.

ar5 #set of 36 rules
plot(ar5, method="graph")
inspect(ar5)

ar5_sub <- subset(ar5, lhs %in% 'workclass=Private')
ar5_sub #set of 9 rules 

plot(ar5_sub, method="graph")


#########################################
# 4. <<식료품점 파일 예제>> 
#########################################

library(arules)

# transactions 데이터 가져오기
data("Groceries")  # 식료품점 데이터 로딩
str(Groceries) # Formal class 'transactions' [package "arules"] with 4 slots
Groceries

# [data.frame 형 변환]
Groceries.df<- as(Groceries, "data.frame")
head(Groceries.df)
summary(Groceries)

itemFrequencyPlot(Groceries, topN=20, type="absolute") # 상위 20개 토픽
# 지지도 0.001, 신뢰도 0.9

rules <- apriori(Groceries, parameter=list(supp=0.001, conf=0.8)) #maxlen=10 기본값

inspect(rules) 
# 규칙을 구성하는 왼쪽(LHS) -> 오른쪽(RHS)의 item 빈도수 보기  
plot(rules, method="grouped")

# 최대 길이 3이내로 규칙 생성
rules <- apriori(Groceries, parameter=list(supp=0.001, conf=0.80, maxlen=3))
rules #set of 410 rules 
inspect(rules) # 29개 규칙

# confidence(신뢰도)를 기준 내림차순으로 규칙 정렬
rules <- sort(rules, decreasing=T, by="confidence") #by="기준이 되는 값"
inspect(rules) #즉, 신뢰도가 가장 높은 룰부터 보여줌.


library(arulesViz) # rules값 대상 그래프를 그리는 패키지
plot(rules, method="graph", control=list(type="items"))

#other vegetables과 whole milk로 몰리는 것을 볼 수 있다.
# 이 둘을 기준으로 두 서브셋을 만들어 볼 수 있어서, 각각을 선행사건으로 두고
# 후행사건을 탐색해볼 수 있다.

#######################################
### 특정 item 서브셋 작성/시각화
#######################################

# 오른쪽 item이 전지분유(whole milk)인 규칙만 서브셋으로 작성
wmilk <- subset(rules, rhs %in% 'whole milk') # lhs : 왼쪽 item
# rhs 후행사건
wmilk # set of 18 rules // 29개중에 18개
inspect(wmilk)
plot(wmilk, method="graph") #  연관 네트워크 그래프

# 오른쪽 item이 other vegetables인 규칙만 서브셋으로 작성
oveg <- subset(rules, rhs %in% 'other vegetables') # lhs : 왼쪽 item
oveg # set of 10 rules 
inspect(oveg)
plot(oveg, method="graph") #  연관 네트워크 그래프

# 왼쪽 item 'butter', 'yogurt'
but_yog <- subset(rules, lhs %in% c('butter', 'yogurt'))
#두 아이템을 포함(or, 둘중에 하나 포함)하는 선행사건
but_yog #set of 4 rules 
inspect(but_yog)
plot(but_yog, method="graph")


