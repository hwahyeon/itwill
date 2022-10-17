#################################
## <제4장 연습문제>
#################################   
# 01. 다음 3개의 vector 데이터를 이용하여 client 데이터프레임을 
# 생성하고, 조건에 맞게 처리하시오.

# vector 준비 
name <-c("유관순","홍길동","이순신","신사임당")
gender <- c("F","M","M","F")
price <-c(50,65,45,75)

client <- data.frame(name, gender, price)
client

# <조건1> price가 65만원 이상인 고객은 "Best" 미만이면 
#     "Normal" 문자열을 result 변수에 넣고, client의 객체에 컬럼으로 추가하기

result <- ifelse(client$price>=65, "Best", "Normal")
client$result <- result

# <조건2> result의 빈도수를 구하시오. 힌트) table()함수 이용
table(client$result)
# Best Normal 
# 2      2 

# <조건3> gender가 'M'이면 "Male", 'F'이면 "Female" 형식으로 client의 객체에
#  gender2 컬럼을 추가하고 빈도수 구하기

gender2 <- ifelse(client$gender=='M',"Male", "Female")
client$gender2 <- gender2

# 02. 다음 벡터(EMP)는 '입사년도이름급여'순으로 사원의 정보가 기록된 데이터 있다.
# 이 벡터 데이터를 이용하여 다음과 같은 출력결과가 나타나도록 함수를 정의하시오. 

# <Vector 준비>
EMP <- c("2014홍길동220", "2002이순신300", "2010유관순260")
length(EMP)

# <출력 결과>
# 전체 급여 평균 : 260
# 평균 이상 급여 수령자
# 이순신 => 300 
# 유관순 => 260 

library(stringr)
# 1차 급여 추출
name <- str_extract_all(EMP, '[가-힣][0-9]{3}')
vname <- unlist(name)
pay <- str_replace_all(vname, '[가-힣]', '')
pay_avg <- mean(as.numeric(pay))
cat('전체 급여 평균 :', pay_avg)

#내가 한 것
# na1<-str_extract_all(EMP, '[가-힣][0-9]{3}')
# na1
# name<-str_extract_all(na1, '[0-9]{3}')
# unlist(name)

#이름 추출
name <- str_extract_all(EMP, '[가-힣]{3}') #리스트로 반환된다.


#사원수 만큼 반복
for(i in 1:length(x)){
  if(as.numeric(pay[i]) >= pay_avg){
    cat(name[i], '=>',)
  }
}





#힌트) 사용 함수
#stringr 패키지 : str_extract(), str_replace()함수
#숫자변환 함수 : as.numeric()함수
#한글 문자 인식 정규표현식 : [가-힣]  

# 함수 정의 : for()함수 이용 예  
emp_pay <- function(x) {
  
    
}

# 함수 호출 
emp_pay(EMP)

# 03. RSADBE 패키지에서 제공되는 Bug_Metrics_Software 데이터 셋을 대상으로 
# 소프트웨어 발표 후 행 단위 합계와 열 단위 평균을 구하고, 
# 칼럼 단위로 요약통계량을 구하시오.  

# 단계1: 패키지 설치 
# 단계2: 메모리 로딩 
# 단계3 : 데이터셋 가져오기 
# 단계4: 통계량 구하기 : 행 단위 합계와 열 단위 평균 

rowSums(Bug_Metrics_Software[,,2])
colMeans(Bug_Metrics_Software[,,2])
















