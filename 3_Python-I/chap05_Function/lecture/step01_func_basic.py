'''
함수(function)
 - 중복 코드 제거
 - 재사용 가능
 - 함수 하나에 많은 기능을 넣는 것보다 특정한 한개의 기능을 정의하는 것이 좋다.
 - 유형) 사용자 정의 함수, 라이브러리 함수
'''

# 1. 사용자 정의 함수
'''
형식)
def 함수명(매개변수) : 
    실행문
    실행문
    return 값1, 값2, ...
'''

#1) 인수가 없는 함수
def userFunc1():
    print('인수가 없는 함수')
    print('userFunc1')

userFunc1() #함수 호출

#2) 인수가 없는 함수
def userFunc2(x, y):
    adder = x+y
    print('adder = ', adder)

userFunc2(10, 20) #adder =  30

#3) 리턴 있는 함수
def userFunc3(x,y):
    add = x+y
    sub = x-y
    mul = x*y
    div = x/y
    return add, sub, mul, div

a, s, m, d = userFunc3(100, 20)
print('add =', a)
print('sub =', s)
print('mul =', m)
print('div =', d)

# 2. 라이브러리 함수 : python에 이미 만들어진 함수
'''
1) built-in : 기본함수()
2) import : 모듈.함수()
'''

# 1) built-in : 기본함수()
dataset = list(range(1,6))
print(dataset) #[1, 2, 3, 4, 5]

print('sum = ', sum(dataset)) #sum
print('max = ', max(dataset)) #최댓값
print('min = ', min(dataset)) #최솟값
print('len = ', len(dataset))
print(mean(dataset)) #mean는 built-in이 아니라서 오류가 나는 것이다.

# 2) import : 모듈.함수()
import statistics # 통계관련 함수 제공 방법1)
'''
Ctrl + 클릭 : module or function source 보기
'''
help(statistics)
from statistics import mean # 방법2) 두번째방법을 많이 씀.

print(dir(statistics)) #해당 모듈의 정보 (제공 함수, 기능)

avg1 = statistics.mean(dataset)
avg2 = mean(dataset)
print(avg1, avg2)

print('평균 =', avg1)
print('평균 =', avg2)













