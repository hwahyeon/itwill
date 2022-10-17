#################################
## <제6장 연습문제>
################################# 

# <dplyr 패키지 관련 연습문제> 
library(dplyr)
data(iris)


# 01. iris의 꽃잎의 길이(Petal.Length) 칼럼을 대상으로 1.5 이상의 값만 필터링하시오.
iris01 <- filter(iris, Petal.Length>=1.5)
iris01
# 02. 01번 결과에서 1,3,5번 칼럼을 선택하시오.
iris02 <- select(iris01, Sepal.Length, Petal.Length, Species)
head(iris02)
# 03. 02번 결과에서 1번 - 3번 칼럼의 차를 구해서 diff 파생변수를 만들고, 앞부분 6개만 출력하시오.
iris03 <- iris %>% mutate(diff = Sepal.Length-Petal.Length) %>% head()
iris03
# 04. 03번 결과에서 꽃의 종(Species)별로 그룹화하여
#Sepal.Length와 Petal.Length 변수의 평균을 계산하시오.
# 힌트 : group_by()
iris04 <- iris03 %>% group_by(Species)


iris05 <-summarise(iris04, SL_avg = mean(Sepal.Length, na.rm = T),
                         PL_avg = mean(Petal.Length, na.rm = T))
iris05

# <reshape2 패키지 관련 연습문제> 
library('reshape2')

# 05. reshape2 패키지를 적용하여 각 다음 조건에 맞게 iris 데이터 셋을 처리하시오. 

# 조건1) 꽃의 종류(Species)를 기준으로 ‘넓은 형식’을 ‘긴 형식’으로 변경하기(melt()함수 이용)
iris_long <- melt(iris, id="Species")
iris_long
dim(iris_long) # 600   3

# 조건2) 꽃의 종별로 나머지 4가지 변수의 합계 구하기(dcast()이용)
head(iris_long)
#Species     variable value
iris_wide <- dcast(iris_long, Species ~ variable, sum)
iris_wide


















