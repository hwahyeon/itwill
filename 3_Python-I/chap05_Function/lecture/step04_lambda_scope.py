'''
1. 축약함수(lambda)
 - 한 줄로 함수를 표현하는 방법
 형식) 변수 = lambda 인수(외부에서 인수를 받을 수 있는 실인수) : return값
ex) lambda x, y : x+y

2. scope
 - 전역변수, 지역변수(함수)
'''

# 1. 축약함수
def adder(x, y):
    add = x + y
    return add

# 이 함수를 lambda로 한줄로 표현
add = lambda x, y : x + y

#[실행문 for 변수 in 열거행객체]여기에다가
#[lambda x, y : x+y for 변수 in 열거행객체] 이런식으로 사용할 수 있다.

result = add(10, 20)
print(result) # 30

# 2. scope
'''
전역변수 : 전지역에서 사용되는 변수
지역변수 : 특정 지역(함수)에서만 사용되는 변수
'''

x = 50 # 전역변수 (함수 바깥에서 선언된 변수)

def local_func(x) :
    x += 50 # x = 100 : 지역변수
    # 함수내에서 새롭게 만들어진 변수는 지역변수이므로 해당 함수가 종료되면 자동으로 소멸.

local_func(x) # x=50
print('x =', x) # x = 50

def global_func() :
    global x # 전역변수(global 키워드를 넣으면 변수를 전역변수로 사용하겠다는 의미가 됨)
    x += 50 # x = 100
#전역변수에 더한 것이기 때문에 함수 호출이 끝나도 x의 값은 100이 된다.

global_func() # 함수
print('x=', x) # x = 100







