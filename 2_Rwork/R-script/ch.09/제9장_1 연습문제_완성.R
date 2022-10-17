#################################
## <제9장-1 연습문제>
################################# 

# 01. 다음과 같은 단계를 통해서 테이블을 호출하고, SQL문을 이용하여 레코드를 조회하시오.
# (DBMS : Oracle 사용)

# [단계 1] 사원테이블(EMP)을 검색하여 결과를 EMP_DF로 변수로 불러오기
EMP_DF <- dbGetQuery(conn, "select * from EMP")

# [단계 2] EMP_DF 변수를 대상으로 부서별 급여의 합계를 막대차트로 시각화(col)
library(dplyr)
# 부서별 그룹 생성 
emp_g <- group_by(EMP_DF, DEPTNO) # 부서번호 기준 그룹 
# 부서별 급여 합계 
emp_sal_tot <- summarise(emp_g, dept_tot = sum(SAL))

# [단계 3] 막대차트를 대상으로 X축의 축눈금을 부서명으로 표시하기
# 부서 테이블 조회 
dept <- dbGetQuery(conn, "select * from dept")
# 부서명 추출 
dname <- dept$DNAME[1:3]
# x축 눈금(names.arg) : 부서명 표시 
barplot(emp_sal_tot$dept_tot, 
        col = rainbow(3),
        names.arg = dname)




