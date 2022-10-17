'''
동적 멤버 변수 생성
  - 필요한 경우 특정 함수 또는 생성자에서 멤버변수 생성
  self : 클래스의 멤버를 호출하는 역할
  self.멤버변수
  self.멤버메서드()
'''

class Car :
    # 멤버 변수
    door = cc = 0
    name = None #null 아직 배정되지 않음.

    # 생성자 : 객체 + 초기화
    def __init__(self, door, cc, name):
        # self.멤버변수 = 매개변수
        self.door = door #동적 멤버변수 생성
        self.cc = cc #동적 멤버변수 생성
        self.name = name #동적 멤버변수 생성

    # 멤버 메서드 : 자료 처리 역할
    def info(self):
        self.kind = "" #동적멤버 변수
        if self.cc > 3000:
            self.kind = "대형"
        else:
            self.kind = "소형"
        self.display()
    def display(self):
        print("%s는 %d cc이고(%s), 문짝은 %d개 이다."
              %(self.name, self.cc, self.kind, self.door))

#객체 생성 : 생성자() -> object
car1 = Car(4, 2000, '소나타')
# car1.member or car1.member()

print('자동차 명 : ', car1.name) #자동차 명 :  소나타
car1.info() #소나타는 2000 cc이고(소형), 문짝은 4개 이다.

# 객체2
car2 = Car(4, 3000, '그랜져')
print('자동차 명 :', car2.name) #자동차 명 : 그랜져
car2.info() #그랜져는 3000 cc이고(소형), 문짝은 4개 이다.

