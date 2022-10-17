'''
반복문(while)

while 조건식 :
    실행문
    실행문
'''

# 카운터, 누적 변수
cnt = tot = 0 #변수 초기화

while cnt < 5: #조건식 이 조건식이 Ture이면 Loop(명령문의 집합) 실행._
    pass #블록에 아무 내용도 없을 때.

while cnt < 5:
    cnt += 1 # 카운터 변수
    tot += cnt #누적변수
    print(cnt, tot)

# 문1) 1~100까지 합 출력하기
cnt = tot = 0
while cnt < 101:
    cnt += 1
    tot += cnt
print("1~100까지 합 : %d" %(tot))

# 1~100까지 짝수 값만 빈리스트에 저장하기.
cnt = tot = 0
data = [] # 빈 리스트

while cnt < 100:
    cnt += 1
    tot += cnt
    if cnt % 2 == 0: # 2의 배수(짝수)
        data.append(cnt) #짝수 값 추가

print("1~100까지 합 : %d" %(tot))
print("1~100 중 짝수들 : ", data)

#문2) 1~100 사이에서 5의 배수이면서 3의 배수가 아닌 값만 append 하기
cnt = tot = 0
data = [] #5의 배수이면서 (and) 3의 배수가 아닌 값 저장

while cnt < 100:
    cnt += 1
    tot += cnt
    if cnt % 5 == 0 and cnt % 3 != 0:
        data.append(cnt)
print("1~100 중 5의 배수이면서 3의 배수가 아닌 수 : ", data)

# 무한 loop -> 종료 조건(0)
while True:
    num = int(input("숫자 입력 : "))
    if num == 0:
        print("프로그램 종료")
        break #탈출(exit) : 종료 조건
    print("num = ", num)

# random : 난수 생성
import random #난수 생성 모듈

help(random.random) # 0~1 난수 실수
#random() -> x in the interval [0, 1).
help(random.choice)
#method of random.Random instance
help(random.randint) #난수 정수
#Return random integer in range [a, b], including both end points.

r=random.random() #모듈.함수(0~1 난수)
print('r = ', r)

# 문3) 난수 0.01 미만이면 종료, 아니면 난수 개수 출력
# 종료 조건 : 0.01 미만

cnt = 0
while True :
    r = random.random()
    if r < 0.01 :
        break
    else:
        cnt += 1
print('난수 개수 = ', cnt)

r = random.randint(1,5) # 1 ~ 5 난수 정수
print(r)

print(">>>숫자 맞추기 게임<<<")
'''
숫자 범위 : 1 ~ 10
myInput == computer : 성공(exit) -> 종료 조건
myInput > computer : '더 작은 수를 입력하시오.' 
myInput < computer : '더 큰 수를 입력하시오.'
'''
computer = random.randint(1, 10)
while True:
    myInput = int(input("예상 숫자 입력 : ")) #사용자 입력
    if myInput == computer:
        print('성공')
        break #exit
    elif myInput > computer:
        print('더 작은 수를 입력하시오.')
    else:
        print('더 큰 수를 입력하시오.')

'''
continue vs break
 - 반복문에서 사용되는 명령어
 - continue : 반복을 지속, 다음 문장 skip
 - break : 반복 멈춤
'''

i = 0
while i < 10:
    i += 1

    if i == 3:
        continue # 다음 문장 skip // 컨티뉴 뒤에 있는 문장들 실행 x 처음으로 돌아감.
    if i == 6:
        break # 블록을 빠져나감.
    print(i, end='  ') # 1  2  4  5



