#연습장

sex <- factor("m", c("m", "f"))
sex
nlevels(sex)
levels(sex)
levels(sex) <- c("male", "female")
sex

ex <- c(1, 2, 3, 4, 5)
ex
ex[-5]
ex[-1]
ex[1:5]   #ex[start:end]

length(ex)
NROW(ex)


"1" %in% ex
"7" %in% ex

ex2 <- c(1, 2, 3, 4, 5, 5, 5)
identical (ex,ex2)
setequal (ex,ex2)

seq_along(ex2)
ex3 <- c(7, 7, 7, 7, 7, 7, 7, 7)
seq_along(ex3)

1:NROW(ex3)

exlist <- list(l.name="J", f.name="L")
exlist

exlist[1]
exlist[[1]]
exlist$l.name

#행렬
M<-matrix(c(1:9), ncol=3)
M
M[]
e1<-M[c(1,3),c(1,3)]
e1

e2<-M[1:2,]
e2
e3<-t(e2) #전치행렬
e3

solve(M)

x <- matrix(c(1:4),ncol=2)
x
x1<-solve(x)
I<-x%*%x1


print(
  'Hello'
)


string <- "hong35lee45kang55유관순25이사도시45"
str_locate(string,'o')
str_locate_all(string,'ng')


x<-c(1:5)
dim(x)

x<-data.frame(2:5)
x

View(x)

x <- c(11,31,3100,21,41)
median(x)
?median

y=seq(2,10,2)
y

##피보나치 수열
fibo <- function(n){
  if(n==1||n==2){
    return(1)
  }
  return(fibo(n-1)+fibo(n-2))
}
fibo(5)
fibo(3)


#1 1 2 3 5 8 13


head(iris)

pairs(head(iris[1:2]))

x<-head(iris[1])
y<-head(iris[2])
plot(head(iris[1:2]))


D<-head(iris[1:2])
D$Sepal.Length2<-D$Sepal.Length
D
D1<-D[-1]
plot(D1)







name <- c("hong", "lee", "kang")
name # "hong" "lee"  "kang"
age <- c(35, 45, 55)
age  

# 나이에 이름 지정 
names(age) <- c("hong", "lee", "kang")  
age  

names(age) <- name












































################################
###########6장 그래프###########
################################

install.packages("mlbench")
library(mlbench)
data("Ozone")

head(Ozone)
plot(Ozone$V8, Ozone$V9) #x축, y축 순서

plot(Ozone$V8, Ozone$V9, xlab='Sandburg Temperature', ylab='El Monte Temperature',
     main="Ozone", pch=20)
plot(Ozone$V8, Ozone$V9, xlab='Sandburg Temperature', ylab='El Monte Temperature',
     main="Ozone", pch='+', cex=0.5, col='red')

min(Ozone$V8, na.rm=T) #최솟값 : 25
max(Ozone$V8, na.rm=T) #최댓값 : 93
min(Ozone$V9, na.rm=T) #최솟값 : 27.68
max(Ozone$V9, na.rm=T) #최댓값 : 82.58

data(cars)
str(cars)
head(cars)

plot(cars, type='b')
plot(cars, type='o')
plot(cars, type='l')

plot(cars, type='l', lty='dotdash') #선 유형 옵션

opar <- par(mfrow=c(2,2)) #1행 2열로 그래프 그리기
opar
plot(Ozone$V8, Ozone$V9, xlab='Sandburg Temperature', ylab='El Monte Temperature',
     main="Ozone", pch=20)
plot(Ozone$V8, Ozone$V9, xlab='Sandburg Temperature', ylab='El Monte Temperature',
     main="Ozone", pch='+', cex=0.5, col='red')



##########################Google map############################
library(ggmap)
get_googlemap('Losangeles',zoom=15,maptype="roadmap")
# 에러: Google now requires an API key.
# See ?register_google for details.



library(mlbench)
data(Ozone)
plot(Ozone$V8, Ozone$V9, xlab="Sandburg Temperture", ylab = "El Monte Temperature", main="Ozone",
     col.axis = "blue")

plot(Ozone$V8, Ozone$V9, xlab="Sandburg Temperture", ylab = "El Monte Temperature", main="Ozone",
     col.lab = "red")


plot(Ozone$V8, Ozone$V9, xlab="Sandburg Temperture", ylab = "El Monte Temperature", main="Ozone",
     col.axis = "blue", col.lab = "red", ylim = c(0,200))


plot(Ozone$V8, Ozone$V9, xlab="Sandburg Temperture", ylab = "El Monte Temperature", main="Ozone",
     col.axis = "blue", col.lab = "red", ylim = c(0,200), type='p')



data(cars)
str(cars)

plot(cars, type='b')

plot(tapply(cars$dist, cars$speed, mean))



tapply(cars$dist, cars$speed, mean)

par(mfrow=c(1,2))

opar <- par(mfrow=c(2,2))

par(opar)





plot(Ozone$V6, Ozone$V7, pch=20, cex=.5)
plot(jitter(Ozone$V6), jitter(Ozone$V7), pch=20, cex=.5)














