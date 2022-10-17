# Chap09_Formal_InFormal(Oracle)

# chap09_1_Formal(1_Oracle)

########################################
## Chapter09-1. 정형데이터 처리 
########################################

# Oracle DB 정형 데이터 처리

# 1. 패키지 설치
# - RJDBC 패키지를 사용하기 위해서는 우선 java를 설치해야 한다.
#install.packages("rJava")
install.packages("DBI")
install.packages("RJDBC")

# 2. 패키지 로딩
#DBI는 의존성이 없기 때문에 먼저 올린다.
library(DBI)
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_151') #가상머신 경로
library(rJava)
library(RJDBC) # rJava에 의존적이다.(rJava 먼저 로딩)
#RJDBC는 자바로 개발된 패키지이다. 이 패키지를 쓰려면 위의 설정이 필요한 것.


# 3) Oracle 연동   

############ Oracle 11g ##############
# driver  object
drv<-JDBC("oracle.jdbc.driver.OracleDriver", 
          "C:/oraclexe/app/oracle/product/11.2.0/server/jdbc/lib/ojdbc6.jar")
# db연동(driver, url,uid,upwd)  object 
conn<-dbConnect(drv, "jdbc:oracle:thin:@//127.0.0.1:1521/xe","scott","tiger")
####################################

query <- "select * from tab"
dbGetQuery(conn, query) #scott이 만든 db를 보여줌.

# table 생성
query <- "create table db_test(sid int, pwd char(4), name varchar(20), age int)"
dbSendUpdate(conn, query)

dbGetQuery(conn, "select * from tab")     # DB_TEST   TABLE        NA

# db 내용 수정 : insert, update, delete

# 1. insert
query <- "insert into db_test values(1001, '1234', '홍길동', 35)"
dbSendUpdate(conn, query)

dbGetQuery(conn, "select * from db_test")

# 2. update
dbSendUpdate(conn, "update db_test set name = '김길동' where sid = 1001")
dbGetQuery(conn, "select * from db_test")

# 3. delete
dbSendUpdate(conn, "delete db_test where sid = 1001")
dbGetQuery(conn, "select * from db_test")

# 4. table drop
dbSendUpdate(conn, "drop table db_test purge")


EMP <- dbGetQuery(conn, "select * from emp")
str(EMP) #'data.frame':	14 obs. of  8 variables:
mean(EMP$SAL) #2073.214
summary(EMP)


# 문1) SAL 2500이상이고, 직책(JOB)이 MANAGER인 사원만 검색하기

query <- "(select * from emp where SAL >= 2500 and JOB='MANAGER')"
manager_2500 <- dbGetQuery(conn, query)
str(manager_2500)
manager_2500

# 문2) sub query 관련 문제
# 부서가 'SALES'인 전체 사원의 이름, 급여, 직책 조회하기
# sub : DEPT, main : EMP

dbGetQuery(conn, "select * from DEPT")

query <- "select ename, sal, job from EMP
          where DEPTNO = (select DEPTNO from DEPT where DNAME ='SALES')"

dbGetQuery(conn, query)

# 문3) join 쿼리
dbGetQuery(conn, "select * from product")
dbGetQuery(conn, "select * from sale")

query <- "select p.code, p.name, s.price, s.sdate
          from product p, sale s
          where p.code = s.code and p.name like '%기'"

dbGetQuery(conn, query)

# db 연결 종료
dbDisconnect(conn) #TURE가 나오면 끊어졌다는 것.


