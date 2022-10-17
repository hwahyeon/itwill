#################################
## <제8장 연습문제>
################################# 

#01. 다음 조건에 맞게 airquality 데이터 셋의 Ozone과 Wind 변수를 대상으로  
# 다음과 같이 산점도로 시각화 하시오.

#조건1) y축 : Ozone 변수, x축 : Wind 변수 
#조건2) Month 변수를 factor형으로 변경  
#조건3) Month 변수를 이용하여 5개 격자를 갖는 산점도 그래프 그리기

head(airquality)
str(airquality)

# transform() : base 제공 
convert <- transform(airquality, Month=factor(Month))
str(convert) # Month 변수의 Factor값 확인
# $ Month  : Factor w/ 5 levels "5","6","7","8"

convert # Ozone Solar.R Wind Temp Month Day
xyplot(Ozone ~ Wind | Month, data=convert, layout=c(5,1))
# 컬럼 제목 : Month 값으로 출력



# 02. 서울지역 4년제 대학교 위치 정보(Part-II/university.csv) 파일을 이용하여 레이어 형식으로 시각화 하시오.

# 조건1) 지도 중심 지역 SEOUL, zoom=11
# 조건2) 위도(LAT), 경도(LON)를 이용하여 학교의 포인트 표시
# 조건3) 각 학교명으로 텍스트 표시
# 조건4) 완성된 지도를 "university.png"로 저장하기(width=10.24,height=7.68) 

library(ggmap)
setwd("C:/ITWILL/2_Rwork/Part-II") # 파일 경로 지정 

# 서울지역 4년제 대학교 위치 정보 자료 가져오기 
university <- read.csv("university.csv")
university # # 학교명","LAT","LON"

# 지도정보 생성
seoul <- c(left = 126.77, bottom = 37.40, 
           right = 127.17, top = 37.70)

map <- get_stamenmap(seoul, zoom=11,  maptype='watercolor')


# (1)레이어1 : 정적 지도 생성
layer1 <- ggmap(map)
layer1

# (2)레이어2 : 지도위에 포인트
layer2 <- layer1 + geom_point(data=university, 
                              aes(x=LON,y=LAT, color=학교명), size=3)
layer2

# (3)레이어3 : 지도위에 텍스트 추가
layer3 <- layer2 + geom_text(data=university, 
                             aes(x=LON+0.01, y=LAT+0.01,label=학교명), size=5)
layer3

# (4)지도 저장
# 넓이, 폭 적용 파일 저장
ggsave("university.png",width=10.24,height=7.68)

