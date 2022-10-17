'''
중첩함수(inner function)

형식
def outer_func(인수):
    실행문
    def inner_func(인수):
        실행문

    return inner_func
'''

# 1. 중첩함수 예
def a() : #outer
    print('a 함수')
    def b(): #inner
        print('b 함수')
    return b #inner func

b = a() #outer 호출 - a 함수 = 일급함수
b() #inner 함수 호출 - b 함수

# b랑 b()랑은 다른 것이다!

# 2. 중첩함수 응용 // outer는 inner를 여러개 가질 수 있다.
'''
inner 함수 종류
getter() : 함수 내의 data를 외부 획득자 
setter() : 함수 내의 data를 지정자 
'''
def outer_func(data) : # outer의 역할 1.데이터 저장, 2.inner 포함
    dataSet = data

    # inner의 역할 : 데이터 조작
    def tot(): # 합계
        tot_val = sum(dataSet)
        return tot_val
    def avg(tot_val): #평균 = 합계/n
        avg_val = tot_val / len(dataSet)
        return avg_val

    # 내부 데이터를 외부에서 알고 싶을 때(반환받고 싶을 때) getter를 씀.
    # getter 인수는 없지만 값을 반환해야하기 때문에 return을 갖고 있어야 함.
    def getData():
        return dataSet
    # setter : 어떤 특정 값을 지정하거나 변경해야하기 때문에 인수를 필요로 함. 반환은 굳이 필요 없다.
    def setData(newData):
        nonlocal dataSet # outer 변수
        # nonlocal을 안쓰면 지역변수로 해석이 되버림. outer를 inner에서 수정하고자 할때에 써줘야함.
        # 참조와 수정은 다른 것! 참조만 해서 길이만 구하거나 값을 구하는 것과
        # 값을 아예 수정하는 것은 완전히 다르다.
        dataSet = newData # 지역변수

    return tot, avg, getData, setData

data = list(range(1,101))
tot, avg, getData, setData = outer_func(data) #일급함수(tot, avg, get, set)

tot_val = tot() # 이것을 호출하면 tot_val이 반환될 것. # 합계 계산
avg_val = avg(tot_val) # 평균 계산
print('tot=', tot_val) #tot= 5050
print('avg =', avg_val) #avg = 50.5
print('dataset =', getData()) # dataset 리턴

newData = list(range(1,51))
setData(newData) # dataSet 변경

# getter 이용 : dataSet 확인
print('dataSet', getData())


# 3. 함수 장식자 : Tensorflow 2.0에서 적용
# - 기존 함수의 시작부분과 종료부분에 기능을 추가해서 장식 역할, 단지 꾸며주는 역할만 하는 것은 아님.
# 직접 id / pw를 입력받아 인증을 받아 접근을 할 수 있게 한다거나 하는 식으로 활용할 수 있다.
'''
@함수장식자
def 함수명():
    실행문
'''

# 함수장식자 작성
def hello_doco(func) : #outer : 함수를 인수로 받음
    def inner(name): #inner : 함수를 장식하는 역할, hello 인수 받음
        print('-'*20) # 함수 앞부분 decoration
        func(name) # 함수 실행
        print('-'*20) # 함수 뒷부분 decoration
    return inner

@hello_doco #함수 장식자 결합
def hello(name): #타겟이 되는 함수 name은 hello_codo의 inner함수로 넘어감.
    print("my name is " + name + "!!")

# 함수 호출
hello("홍길동")
'''
--------------------
my name is 홍길동!!
--------------------
'''
hello("이순신")


# 구구단 장식하기
'''
**** 2단 ****
 2 * 1 = 2
    :
 2 * 9 = 15
*************    
'''

# 함수장식자
def gugu_deco(gugu_func):
    def inner(dan):
        print('*'*4, dan, '단', '*'*4)
        gugu_func(dan)
        print('*'*13)
    return inner

# 구구단 계산
@gugu_deco
def gugu(dan):
    for i in range(1,10):
        print("%d * %d = %d"%(dan, i, dan*i))

dan = int(input("단 입력(2~9) : "))
gugu(dan)





