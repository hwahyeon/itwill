'''
1. 메서드 재정의(method override)
 - 부모의 원형 메서드 -> 자식에 원형 메서드를 다시 작성하는 문법
 - 상속관계에서만 나오는 용어
 - 인수, 내용 -> 수정 대상 (집 한 채를 넘겨주면 그 안에 있는 것들을 고칠 수 있다는 의미)

2. 다형성
 - 상속관계에서만 나오는 영어
 - 기능은 한가지인데 한가지 기능으로 두 개 이상의 결과가 만들어질 수 있게 하는 것
 - ex) +는 더한다는 기능을 가지고 있는데, 들어오는 데이터가 숫자면 덧셈, 문자면 결합연산자로 작용한다.
 - 부모 객체 -> 자식 멤버 호출 (부모는 하나지만 자식은 여럿일 수 있다)
'''

# 1. 메서드 재정의(method override)

# 부모 클래스
class Super :
    data = None # 멤버 변수

    # 기본 생성자 : 객체만 생성

    # 멤버 메서드 : 원형 메서드
    def superFunc(self):
        pass # 내용 없음

# 자식1
class Sub1(Super):
    # data
    # def superFunc

    def superFunc(self, data): # 수정 -> override
        self.data = data
        print("자식1 : data = {}".format(self.data))

sub1 = Sub1() #기본 생성자
sub1.superFunc('20200414') # 자식 : data = 20200414

# 자식2
class Sub2(Super):
    # data
    # def superFunc

    def superFunc(self, data): #부모에서 넘겨준 것을 확장 -> override
        self.data = data
        print("자식2 : data = {}".format(self.data))

class Test :
    def superFunc(self, data): # 확장 -> override
        self.data = data
        print("자식3 : data = {}".format(self.data**2))

sub2 = Sub2()
sub2.superFunc(100) #자식2: data = 100

# 2. 다형성
sup = Super() # 부모 객체
sub1 = Sub1() # 자식1 객체
sub2 = Sub2() # 자식2 객체

sup = sub1
sup.superFunc(100) #자식1 : data = 100
sub = sub2
sub.superFunc(100) #자식2 : data = 10000



