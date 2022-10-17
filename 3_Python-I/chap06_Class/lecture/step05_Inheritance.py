'''
클래스 상속(Inheritance)
 - 기존 클래스(부모)를 이용하여 새로운 클래스(자식) 생성 문법
 - 부모클래스 정의 -> 자식클래스 생성
 - 상속 대상 : 멤버(o) + 생성자(x)
   -> 생성자 상속 대상 아님

형식)
class 자식클래스(부모클래스) : #class new_class(old_class)
    멤버(변수 + 메서드)
    생성자

self vs super()
    - self.member : 현재 클래스의 멤버를 호출
    - super().member : 부모 클래스의 멤버를 호출
'''


# 부모클래스 : old class
class Super : # member는 3개. (멤버변수 2개, 멤버메서드 1개)
    # 멤버변수
    name = None
    age = 0

    # 생성자 : 객체 생성 + 초기화
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # 멤버메서드 : 데이터 처리
    def display(self):
        print("이름 : {}, 나이 : {}".format(self.name, self.age))

# object 생성
#super = Super("부모", 55) # super는 키워드라서 오류가 발생
sup = Super("부모", 55)
del super
# object.member()
sup.display() #이름 : 부모, 나이 : 55

# 자식클래스 (상속을 통해 새로운 클래스를 만드는 것)
class Sub(Super): # member(3+1=4)
    # name = None # 상속받은 부모 멤버
    # age = 0 # 상속받은 부모 멤버
    gender = None #자식 멤버

    def __init__(self, name, age, gender): #자식 클래스에서 새로 정의된 자식 생성자/ 생성자는 상속이 안되기 때문에 새롭게 만든 것.
        #Super.__init__(self, name, age) # 부모클래스의 이름으로 호출, 접근하는 방법
        super().__init__(name, age) # 2차 :  부모 생성자 호출 #super()키워드로 호출
        '''
        self.name = name #위에 안보인다고 해서 없는 것이 아니다. # 1차 : 자식 생성자 초기화
        self.age = age
        '''#부모한테 있는 것을 굳이 재생성하는 것은 코드 낭비이므로 부모 생성자를 호출하는 방법을 취하는 것.
        self.gender = gender

    #def display(self): 이것 역시 눈에 보이진 않지만 상속받아서 엄연히 존재하는 것임.
    def display(self): # 상속받은 것을 확장할 수 있다. 2개 -> 3개 확장
        print("이름 : {}, 나이 : {}, 성별 : {}"
              .format(self.name, self.age, self.gender))

#object
sub = Sub('자식', 22, '남자')
# object.member()
sub.display()
# 이름 : 자식, 나이 : 22, 성별 : 남자

# 1. 부모 클래스 정의
class Parent:
    #멤버 변수
    name = job = None
    
    #생성자
    def __init__(self, name, job):
        self.name = name
        self.job = job
        
    # 멤버 메서드
    def display(self):
        print("이름 : {}, 직업 : {}".format(self.name, self.job))
        
p = Parent('홍길동', '공무원')
p.display()

# 자식 클래스1
class Children(Parent):
    # name = job
    gender = None
    
    def __init__(self, name, job, gender): #부모클래스에선 쓰지 않은 gender만 추가해주면 된다.
        Parent.__init__(self, name, job) #부모 생성자를 호출해서 부모의 멤버를 초기화
        self.gender = gender # 자식의 멤버 초기화

    def display(self):
        print("이름 : {}, 직업 : {}, 성별 : {}"
              .format(self.name, self.job, self.gender))

c1 = Children("이순신", "군인", "남자")
c1.display()
# 이름 : 이순신, 직업 : 군인, 성별 : 남자

'''
Parent -> Children2
  이름 : 유관순
  직업 : 독립열사
  성별 : 여자 
'''

class Children2(Parent):
    gender = None
    def __init__(self, name, job, gender):
        Parent.__init__(self, name, job)
        self.gender = gender

    def display(self):
        print("이름 : {}, 직업 : {}, 성별 : {}"
              .format(self.name, self.job, self.gender))

c2 = Children2("유관순", "독립열사", "여자")
c2.display()




