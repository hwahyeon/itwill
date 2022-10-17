'''
문1) 다음 emp '입사년도이름급여'순으로 사원의 정보가 기록된 데이터 있다.
      이 벡터 데이터를 이용하여 사원의 이름만 추출하시오. 

# <출력 결과>
 names = ['홍길동', '이순신', '유관순']
'''

from re import findall

# <Vector data>
emp = ["2014홍길동220", "2002이순신300", "2010유관순260"]

# for문 이용한 경우
name2 = []
for exa in emp:
    name1 = findall('[가-힣]{3,}',exa)
    name2.append(name1[0])
    print(name1)

print('names = ', name2)

# list + for 이용한 경우 // 이러한 형식이 많이 쓰인다.
# 변수 = [실행문 for 변수 in 열거행객체]
name2 = [findall('[가-힣]{3,}',exa)[0] for exa in emp]
print('names = ', name2)

