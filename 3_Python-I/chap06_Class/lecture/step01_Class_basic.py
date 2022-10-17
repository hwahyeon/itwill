'''
클래스(class)
    - 함수의 모임
    - 역할 : 다수의 함수와 공유 자료를 묶어서 객체(object)를 생성해주는 역할
    - 구성 요소 : 멤버(member) + 생성자
    - 유형 : 사용자 정의 클래스, 라이브러리 클래스(python)
    - 멤버(member) : 변수(자료 저장) + 함수*(자료 처리)
    - 함수*는 클래스에서 정의된 함수로써 다른 함수와 구분하기 위해 '메서드'라고 표현함.
    - 생성자 : 객체를 생성하는 역할
    형식)
    class 클래스명 :
        멤버변수 = 자료 # 자료 저장
        def 멤버메서드() : # 자료 처리
            자료 처리
        생성자 : # 객체 생성

'''

# 1. 중첩함수
def calc_func(a, b) : #outer 함수 : 실제 자료를 저장,보관하는 역할
    x = a #x에 a를 저장.
    y = b

    #inner :  자료 처리(조작)
    def plus():
        return x + y
    def minus():
        return x - y
    return plus, minus

p, m = calc_func(10, 20) #일급함수가 저장됨. plus가 p로 minus가 m으로 저장된 것임.
print('plus =', p()) # plus = 30
print('minus =', m()) # minus = -10

# 2. 클래스 정의
class calc_class:
    #멤버변수(전역변수) : 자료 저장
    x = y = 0

    # 생성자 : 객체 생성 + 멤버변수 값 초기화
    #def __init__(self): #생성자의 모양(사용자가 정의할 수 없다) 기본 self라는 인수를 가지고 있다.
    def __init__(self, a, b): #x는 a로, y는 b로 초기화한다는 의미
        # 멤버변수 초기화
        self.x = a #바깥쪽 클래스의 멤버변수를 참조하려면 self를 사용한다. 여기서 self를 빼고 그냥 x를 쓰면 지역변수로 쓰는 것이다.
        self.y = b

    # 멤버 메서드 : 클래스에서 정의한 함수
    def plus(self):
        return self.x + self.y

    def minus(self):
        return self.x - self.y

# 함수는 호출해서 사용하지만, 클래스는 생성자를 이용해 객체를 만들어주는 것.
calc_class() # 생성자 = 클래스명(), 모습만보고 함수인지 클래스인지 구분은 불가하다. Ctrl+마우스 왼쪽 클릭 -> 출처로 이동 // 생성자로 이동됨.

obj1 = calc_class(10, 20)
#object.member() : 멤버메서드 호출
print('plus = ', obj1.plus()) #plus =  30
print('minus = ', obj1.minus()) #minus =  -10
#object.member : 멤버변수 호출

print('x =', obj1.x) # x = 10
print('y =', obj1.y) # y = 20

# 하나의 클래스는 여러 개의 오브젝트를 만들 수 있다. 1:n관계이다.
# 100, 200이란 값을 더하고 빼려면 새로운 객체를 만들면 됨.

# 클래스는 1개 -> 다수의 객체 생성 가능
# 생성자 -> 객체2
obj2 = calc_class(100, 200) #객체 생성자
print('plus =', obj2.plus()) #plus = 300
print('minus = ', obj2.minus()) #minus =  -100
print('x =',obj2.x) #x = 100
print('y =',obj2.y) #y = 200

#출처는 같지만 서로 독립된 메모리 공간을 가지고 있다.
#객체의 주소 확인
print(id(obj1)) #1829532685384
print(id(obj2)) #1829532723848

# 3. 라이브러리 클래스
from datetime import date # from 모듈 import 클래스

today = date(2020, 4, 13) # 생성자 -> 년월일을 넣어 객체를 만듦.

#object.member
print('year :', today.year) #today.year는 ()가 없으므로 메소드는 아니다. 그저 값을 저장할뿐인 멤버변수이다.
print('month :', today.month)
print('day :', today.day)

#object.member() : method
week = today.weekday()
print('week :', week)


