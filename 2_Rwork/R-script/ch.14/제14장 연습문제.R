#################################
## <제14장 연습문제>
################################# 

# 01. 다음 mtcars 데이터셋을 대상으로 연비효율(mpg), 실린더수(cyl), 엔진크기(disp),
#마력(hp), 무게(wt) 변수를 대상으로 서브셋을 작성하시오.
library(datasets)
data(mtcars)
head(mtcars)
summary(mtcars)

data <- mtcars[c('mpg', 'cyl', 'disp', 'hp', 'wt')]
summary(data)
str(data)

#02. 작성된 서브셋을 대상으로 상관분석을 수행하여 연비효율(mpg)과 가장 상관계수가 
# 높은 변수를 확인하시오. 
cor(data)

(abs(cor(data)))



wt


# 03. 연비효율과 가장 상관계수가 높은 변수와 산점도로 시각화하시오.
x <- data$mpg
y <- data$wt
plot(x,y)







