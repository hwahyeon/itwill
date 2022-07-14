'''
연산자(Operator)
1. 변수에 값 할당(=)
2. 연산자 : 산술,관계,논리
3. print 형식
4. input : 키보드 입력
'''

# 1. 변수에 값 할당(=)
i = tot = 10
i += 1 # i = i + 1 : 카운터 변수
tot += i # tot = tot + i : 누적 변수
print('i=', i) # i = 11
print('tot =', tot) #tot = 21

v1, v2 = 100, 200
print('v1=',v1 , 'v2=',v2) #v1= 100 v2= 200
v1, v2 = v2, v1 #변수 값 교체
print('v1=',v1 , 'v2=',v2) #v1= 200 v2= 100

'''
tmp = v1
v1 = v2
v2 = tmp
'''

#패킹(packing) 할당
lst = [1, 2, 3, 4, 5] #R에서 vector느낌
v1, *v2 = lst #맨 앞쪽의 1이 v1으로 들어가고 나머지는 다 묶여서 *표시된 v2에 넣어진다.
print('v1=',v1 , 'v2=',v2) #v1= 1 v2= [2, 3, 4, 5]

*v1, v2 = lst
print('v1=',v1 , 'v2=',v2) #v1= [1, 2, 3, 4] v2= 5

# 2. 연산자 : 산술, 관계, 논리
print("산술 연산자")
num1 = 100 #피연산자1
num2 = 20.5 #피연산자2

add = num1 + num2
sub = num1 - num2
mul = num1 * num2
div1 = num1 / num2 #실수 몫
div2 = num1 // num2 #정수 몫
div3 = num1 % num2 # 나머지 계산
square = num1**2 #pow 계산
print(add, sub, mul)
print(div1, div2, div3)
print(square)

print("관계연산자") #True or False
# 1) 동등비교
bool_re = num1 == num2
print(bool_re) #False

bool_re = num1 != num2
print(bool_re) #True

# 2) 대소 관계 : >, >=, <, <=
bool_re = num1 >= num2
print(bool_re) #True

bool_re = num1 <= num2
print(bool_re) #False

print("논리 연산자") #or, and, not
bool_re = num1 >= num2 or num1 <= 10 # or 둘 중 하나라도 true면 true를 반환
#우선순위는 왼쪽의 연산식부터 시작함. 왼쪽에서 True가 나오면 바로 True로 넘기는 것.
print(bool_re) #True

bool_re = num1 >= num2 and num1 <= 10
print(bool_re) #False

bool_re = not(num1 <= 10) #False -> True
print(bool_re) #True

#3. print 형식
help(print) #함수 도움말
# package > module > function or class #module은 R에는 없는 개념. R에서는 패키지 다음에 함수가 바로 나온다.
# Help on built-in(내장 함수) function print in module builtins:
# print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False) #기본 파라미터에 대한 정보
# module builtins : 기본 모듈이라는 뜻. 기본 모듈은 파이썬 설치시 기본으로 설치되는 모듈.
'''
함수의 인수
매개변수 : 값을 넘겨 받는 변수
파라미터 : 값을 갖는 변수 ex) end='\n' , sep=' '
'''

# 1) 기본 인수
print("values = ", 10+20+30) #values = 60 #sep=' '이 기본값이기 때문에 첫번째와 두번째 값 사이에 공백이 있게 된다.
print("출력1", end = ', ') # \n 이 기본으로 설정되어 있음. # 출력1, 출력2
print("출력2")
print("010", "1111", "2222", sep = "-") #010-1111-2222

# 2) format(value, '형식')
print("원주율 =", format(3.14159, "8.3f")) # f : 실수 (숫자는 자릿수를 말함) (전체자릿수.소숫점 이하 자릿수)
print("금액 =", format(10000, "10d")) # 금액 =      10000 (10자리를 이용해서 정수를 표현하겠다 : 10d)
print("금액 =", format(125000, "3,d")) #세번째 마다 ,를 찍겠다는 뜻. #금액 = 125,000

# 3) print("양식문자" %(값)) # 양식문자(p.22)
num1 = 10; num2 = 20
tot = num1 + num2
print("tot = %d" %(tot)) #tot = 30
print("%d + %d = %d" %(num1, num2, tot)) #10 + 20 = 30
print("8진수 = %o, 16진수 = %x" %(num1, num1))
# 8진수 = 12, 16진수 = a
print("%s = %8.3f" %("PI", 3.14159)) #PI =    3.142
print("%s = %.3f" %("PI", 3.14159)) #PI = 3.142

# 4) 외부 상수 받기
# "{}, {}".format(value1, value2)
print("name : {}, age : {}".format("홍길동", 35)) #name : 홍길동, age : 35

print("name : {1}, age : {0}".format("홍길동", 35)) #name : 35, age : 홍길동
# R은 1부터 세지만, python은 0부터 센다.

# format 축약형 : sql문을 만들때 많이 쓰임.
# select * from emp where name = '홍길동'
name = "홍길동"
age = 35
sql = f"select * from emp where name = '{name}' and age = {age}" #f는 포맷이란 뜻.
print(sql) #select * from emp where name = '홍길동' and age = 35

# 4. input("prompt") : 키보드 입력(문자열로 인식)
a = input("첫번째 숫자 입력 : = ") #10  # string -> int
b = input("두번째 숫자 입력 : = ") #20  # string -> int
print('a + b =', a+b) #1020

a = int(input("첫번째 숫자 입력 : = ")) #10
b = int(input("두번째 숫자 입력 : = ")) #20
print('a + b =', a+b) #30

'''
형 변환 관련 함수 
int(value) : value -> integer
float(value) : value -> float
str(value) : value -> string
bool(value) : value -> boolean
'''

a = float(input("첫번째 숫자 입력 : = ")) #string -> float
b = float(input("두번째 숫자 입력 : = ")) #string -> float
print('a + b =', a+b)
print('b=', b) # b=20.0
print(type(b)) #<class 'float'>

# boolean -> int
print(int(False)) # 0
print(int(True)) # 1

# int -> boolean
print(bool(1)) #0을 제외한 모든 값을 True로 반환한다.
print(bool(30))
print(bool(0)) #False
