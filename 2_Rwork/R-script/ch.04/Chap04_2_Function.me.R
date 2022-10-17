# chap04_2_Function

# 1. 사용자 정의함수 

# 형식)
# 함수명 <- function([인수]){
#    실행문
#    실행문
#    [return 값]  
# }

# 1) 매개변수없는 함수 
f1 <- function(){
  cat('f1 함수')
}

f1() # 함수 호출 

# 2) 매개변수 있는 함수 
f2 <- function(x){ # 가인수=매개변수 
  x2 <- x^2
  cat('x2 =', x2)
}

f2(2)
f2(10) # 실인수 

# 3) 리턴(return)이 있는 함수 
f3 <- function(x, y){
  add<-x+y
  return(add) #계산된 결과를, add를 반환하는 것
}

#함수 호출 -> 반환값
add_re <- f3(10, 5) #return된 것을 앞의 변수에 넣는 과정
add_re # 15

avg <- tot / n


num <- 1:10

tot_func <- function(x){
  tot<-sum(x)
  return(tot)
}

tot_re<-tot_func(num)
tot_re # 55

avg <- tot_re / length(num)
avg #5.5

#값을 그냥 출력할거면 cat, 값을 이용할 거면 return을 사용

# 문) calc 함수를 정의하기
#100 + 20 = 120
#100 - 20 = 80
#100 * 20 = 2000
#100 / 20 = 5

calc <- function(x,y){
  cat(x+y, '\n')
  cat(x-y, '\n')
  cat(x*y, '\n')
  cat(x/y, '\n')
}
calc(100,20)

calcr <- function(x,y){
  add <- x+y
  sub <- x-y
  mul <- x*y
  div <- x/y
  
  #return(add, sub, mul, div) #다중인자 반환은 허용하지 않는다.
  #(r에서는 하나의 값만 반환할 수 있다.python은 가능)
  
  calc_df <- data.frame(add, sub, mul, div)
  return(calc_df)
  }

#함수 호출
df <- calcr(100,20)
df

#구구단의 단을 인수로 받아서 구구단 출력하는 함수
gugu <- function(dan){
  cat('***', dan, '단***\n')
  for(i in 1:9){
    cat(dan, '*', i, '=', dan*i, '\n')
    }
}

gugu(2)
gugu(5)

state <- function(fname, data){
  switch(fname,
         SUM = sum(data),
         AVG = mean(data),
         VAR = var(data),
         SD = sd(data)
         )  #스위치를 활용해서 원하는 것을 골라서 활용가능한 함수
}

data <- 1:10
state("SUM", data) #55
state("AVG", data) #5.5
state("VAR", data) #9.166667
state("SD", data) #3.02765


#결측치(NA) 처리 함수
na <- function(x){
  # 1. NA 자체를 제거해서 처리하는 방법
  x1 <- na.omit(x)  #Object에 NA가 포함되어 있으면 이를 제외한다.
  cat('x1 =', x1, '\n')
  cat('x1 =', mean(x1), '\n')
  
  # 2. NA -> 평균으로 대체하는 방법
  x2 <- ifelse(is.na(x), mean(x, na.rm = T), x)
  cat('x2 =', x2, '\n')
  cat('x2 =', mean(x2), '\n')
  
  # 3. NA -> 0
  x3 <- ifelse(is.na(x), 0, x)
  cat('x3 =', x3, '\n')
  cat('x3 =', mean(x3), '\n')
}

x <- c(10, 5, NA, 4.2, 6.3, NA, 7.5, 8, 10)
x
length(x) # 9
mean(x, na.rm = T) #결측치를 제외한 값들의 평균

#함수 호출
na(x)

###################################
### 몬테카를로 시뮬레이션 
###################################
# 현실적으로 불가능한 문제의 해답을 얻기 위해서 난수의 확률분포를 이용하여 
# 모의시험으로 근사적 해를 구하는 기법

# 동전 앞/뒤 난수 확률분포 함수 
coin <- function(n){
  r <- runif(n, min=0, max=1)
  #print(r) # n번 시행 
  
  result <- numeric()
  for (i in 1:n){
    if (r[i] <= 0.5)
      result[i] <- 0 # 앞면 
    else 
      result[i] <- 1 # 뒷면
  }
  return(result)
}

#코인
# coin(5)
# n<-5
# r <- runif(n, min=0, max=1)
# r

# 몬테카를로 시뮬레이션 
montaCoin <- function(n){
  cnt <- 0
  for(i in 1:n){
    cnt <- cnt + coin(1) # 동전 함수 호출 
  }
  result <- cnt / n
  return(result)
}

# 만약에 동전이 0, 1, 0으로 나왔다고 가정하면 확률은 0.3333.
# 0이니까 0/1 = 0
# 1이니까 1/2 = 0.5
# 0이니까 (1+0)/3 = 0.3333.

montaCoin(5) # 0.6
montaCoin(1000) # 0.504
montaCoin(10000) # 0.501

#중심극한정리 - 데이터가 많으면 많을수록 평균에 밀집되는 성질 (정규분포와 연관이 있음)

# 1) 기술통계함수 

vec <- 1:10          
min(vec)                   # 최소값
max(vec)                   # 최대값
range(vec)                  # 범위
mean(vec)                   # 평균
median(vec)                # 중위수
sum(vec)                   # 합계
prod(vec)                  # 데이터의 곱
1*2*3*4*5*6*7*8*9*10
summary(vec)               # 요약통계량 


rnorm(10) #mean=0, sd=1
sd(rnorm(10))      # 표준편차 구하기
sd(rnorm(100))   #10개보다 100개가 더 1에 근사한 값이 나온다.
factorial(5) # 팩토리얼=120
sqrt(49) # 루트

n <- rnorm(100) # mean=0, sd=1
mean(n) #  0.08993331
sd(n) # 1.094573



install.packages('RSADBE') #RSADBE에선 Bug_Metrics_Software데이터셋을 제공함.

library(RSADBE)

library(help="RSADBE")
data(Bug_Metrics_Software)
str(Bug_Metrics_Software)
#num [1:5, 1:5, 1:2] 
Bug_Metrics_Software

# 소프트웨어 발표 전 버그 수
Bug_Metrics_Software[,,1] #Before

# 소프트웨어 발표 후 버그 수
Bug_Metrics_Software[,,2] #After

# 행 단위 합계 : 소프트웨어 별 버그 수 합계
rowSums(Bug_Metrics_Software[,,1])
# JDT     PDE Equinox  Lucene   Mylyn 
# 23750   10552    1959    3428   31014 

# 열 단위 합계 : 버그 별 합계
colSums(Bug_Metrics_Software[,,1])
# Bugs    NT.Bugs      Major   Critical H.Priority 
# 34024      24223       2245        838       9373 

# 행 단위 평균
rowMeans(Bug_Metrics_Software[,,1])

# 열 단위 평균
colMeans(Bug_Metrics_Software[,,1])

#3차원의 어레이에 면을 더 추가하는 방법
#Before에서 After를 빼고 그 값을 3면에 넣기
Bug_Metrics_Software[,,1] #Before
Bug_Metrics_Software[,,2] #After
bug <- Bug_Metrics_Software #복제하기
bug.new <- array(bug, dim = c(5,5,3)) #면 추가 / 원래 5, 5, 2인데 3으로 변경
dim(bug.new) #5 5 3
bug.new[,,1]
bug.new[,,2]
bug.new[,,3] #1면이랑 똑같은데 물어보기.

bug.new[,,3] = bug[,,1] - bug[,,2]
bug.new[,,3]


# 2) 반올림 관련 함수 
x <- c(1.5, 2.5, -1.3, 2.5)
round(mean(x)) # 1.3 -> 1
ceiling(mean(x)) # x보다 큰 정수 
floor(mean(x)) # 1보다 작은 정수 


# 3) 난수 생성과 확률 분포

# (1) 정규분포를 따르는 난수 - 연속확률분포(실수형)
# 형식) rnorm(n, mean=0, sd=1)
n<-1000
r <- rnorm(n, mean = 0, sd = 1) #표준정규분포
r
mean(r) #0.007612918
sd(r) #0.9969745
hist(r) #좌우 균등한 대칭성

# (2) 균등분포를 따르는 난수 - 연속확률분포(실수형)
# 형식) runif(n, min=, max=)
r1 <- runif(n, min=10, max=100)
r2 <- runif(n, min=0, max=1)
r1
r2
hist(r1)
hist(r2)

# (3) 이항분포를 따르는 난수 - 이산확률분포(정수형) /소수점이 없는 이산된 형태의 정수
# 형식)
n<-10
r3 <- rbinom(n, size = 1, 0.5) #전체에서 1이 나오는 확률이 0.5정도
r3

set.seed(123)
n<-10
r4 <- rbinom(n, size = 1, 0.5)
r4 #0 1 0 1 1 0 1 1 1 0

set.seed(123)
n<-10
r4 <- rbinom(n, size = 1, 0.5)
r4 #0 1 0 1 1 0 1 1 1 0 시드값이 같으면 같은 난수가 만들어진다.

#시드값을 생성하지 않으면 매번 새로운 난수가 만들어진다.

n<-10
r5 <- rbinom(n, size = 1, 0.25)
r5

# (4) sample 함수
#sample(모집단, 임의추출하려는 갯수)
sample(10:20, 5) #이것도 난수이기 때문에 실행할 때마다 바뀐다.
sample(c(10:20,50:100), 10)

# train(70%)/test(30%) 데이터셋 / 하나의 데이터셋에서 70%를 훈련에, 30%를 테스트에 활용하는 것.
#홀드아웃방식 일반적으로 7:3, 8:2 비율로 쪼개서 학습용 데이터와 검증용 데이터로 활용하는 것.
dim(iris) #150   5

idx <- sample(nrow(iris), nrow(iris)*0.7)
range(idx)
idx #행번호
length(idx) #105
train <- iris[idx, ] # 학습용 
test <- iris[-idx, ] # 검정용 

dim(train) # 105개
dim(test) # 45개

# 4) 행렬연산 내장함수
x <- matrix(1:9, nrow=3, byrow=T)
dim(x) #3 3
y <- matrix(1:3, nrow=3)
dim(y) #3 1
x;y


#행렬곱 x%*%y
z<-x%*%y
dim(z)

#행렬곱의 전제조건
# 1. x와 y가 모두 행렬이어야함.
# 2. x의 열의 수와 y의 행의 수가 같아야 한다. (수일치)




















