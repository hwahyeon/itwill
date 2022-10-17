'''
제어문 : 조건문(if), 반복문(while, for)
Python 블럭 : 콜론과 들여쓰기

형식1)
if 조건식 :
    실행문
    실행문
'''

var = 10
if var >= 10 :
    print('var =', var)
    print('var는 10보다 크거나 같다.')

print('항상 실행되는 영역')

if var >= 15:
    print('var =', var)
    print('var는 10보다 크거나 같다.')

print('항상 실행되는 영역')

'''
형식2)
if 조건식 :
    실행문1
else :
    실행문2
'''

var = 2
if var >= 5:
    print('var는 5보다 크거나 같다.')
else :
    print('var는 5보다 작다.')

#var는 5보다 작다.

#키보드 점수 입력 -> 60점 이상 : 합격, 미만: 불합격

score = int(input('점수를 입력해주세요.'))
if score >= 60:  #함수로 보지 않고 명령어로 보기 때문에 R에서처럼 ()를 해주지 않는다.
    print('합격')
else :
    print('불합격')

import datetime # module 임포트
today = datetime.datetime.now() #module.class.method()
print(today)

# 요일 반환
week = today.weekday()
print(week) #0~4 : 평일, 5~6: 휴일

if week >= 5:
    print("오늘은 휴일")
else :
    print("오늘은 평일")
#오늘은 평일

'''
형식3)
if 조건식1:
   실행문
elif 조건식2:
   실행문
else :
   실행문
'''

# 문2) 키보드로 score입력 받고 : A(100~90), B(80점대), C, D, F(59점 미만)
score = int(input('점수를 입력해주세요.'))
if score >= 90:
    print('A')
elif score >= 80:
    print('B')
elif score >= 70:
    print('C')
elif score >= 60:
    print('D')
else :
    print('F')

# 전역변수 : score, grade
# 다른 언어는 grade = ""식으로 빈공간을 블록 전에 만들어놓지만, 파이썬에선 블록이 끝나도 전역변수로 활용이 가능하다.
if score >= 90:
    grade = 'A학점'
elif score >= 80:
    grade = 'B학점'
elif score >= 70:
    grade = 'C학점'
elif score >= 60:
    grade = 'D학점'
else :
    grade = 'F학점'
print("당신의 점수는 %d이고, 등급은 %s이다."%(score, grade))

# 블럭형 if문 vs 라인형 if문

# 블럭 if
num = 9
if num >= 5:
    result = num * 2
else:
    result = num + 2
print(result)

# 라인 if
# 형식) 변수 = 참 if 조건문 else 겆시
result = num * 2 if num >= 5 else num + 2
print(result)



