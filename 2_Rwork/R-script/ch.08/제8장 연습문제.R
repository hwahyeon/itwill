#################################
## <제8장 연습문제>
################################# 

#01. 다음 조건에 맞게 airquality 데이터 셋의 Ozone과 Wind 변수를 대상으로  
# 다음과 같이 산점도로 시각화 하시오.

#조건1) y축 : Ozone 변수, x축 : Wind 변수 
#조건2) Month 변수를 factor형으로 변경  
#조건3) Month 변수를 이용하여 5개 격자를 갖는 산점도 그래프 그리기
library(lattice)
library(datasets)

head(airquality)
str(airquality)


dotplot(Ozone ~ Wind | factor(Month) , airquality)


# 자료형 변환
air_df <- transform(airquality, Month = factor(Month))
str(air_df) # Month : Factor

xyplot(Ozone ~ Wind | Month, data = air_df)





# 02. 서울지역 4년제 대학교 위치 정보(Part-II/university.csv) 파일을 이용하여 레이어 형식으로 시각화 하시오.

# 조건1) 지도 중심 지역 SEOUL, zoom=11
# 조건2) 위도(LAT), 경도(LON)를 이용하여 학교의 포인트 표시
# 조건3) 각 학교명으로 텍스트 표시
# 조건4) 완성된 지도를 "university.png"로 저장하기(width=10.24,height=7.68) 

library(ggmap)

# [단계1]
uni <- read.csv(file.choose())
str(uni)

library(stringr)
head(uni)
names <- uni$'학교명'
lat <- uni$LAT
lon <- uni$LON

uni_df <- data.frame(names, lon, lat)
uni_df

# [단계2] 지도 정보 생성
map <- get_stamenmap(seoul, zoom=11,  maptype='watercolor')

# [단계3] 레이어1 : 정적 지도 시각화
layer1 <- ggmap(map) #maptype : terrain, watercolor

# [단계4] 레이어2 : 각 지역 포인트 추가
layer2 <- layer1+ geom_point(data = uni_df, aes(x=lon, y=lat, color=factor(lon)))
layer2

# [단계5] 레이어3 : 각 지역별 포인트 옆에 지명 추가
layer3 <- layer2 + geom_text(data = uni_df, aes(x=lon+0.01, y=lat+0.08, label=region), size=3)
layer3

# 지도 이미지 file save
ggsave("university.png", scale = 1, width = 10.24, height = 7.68)

