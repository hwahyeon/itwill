'''
기본(default) 생성자
    - 생성자를 생략하면 기본 생성자가 만들어진다.
    - 묵시적 생성자
'''

class default_cost :
    # 생성자 생략
    #def __init__(self): # 기본 생성자
    #    pass

    def data(self, x, y):
        self.x = x
        self.y = y
    def mul(self):
        re = self.x * self.y #re는 지역변수
        return re

obj = default_cost() #기본 생성자
obj.data(10,20) # data 생성
print('mul =', obj.mul()) # mul = 200


# TV class 정의
class TV : # class = 변수(명사, 자료) + 메서드(동사, 기능(자료처리 기능))
    # 멤버변수 : 자료 저장
    channel = volume = 0 #숫자값이니 0으로 초기화
    power = False #켜고 끄는 것 뿐이니 참/거짓으로 초기화 # off(False) -> on(True)
    color = None #null 값

    # 기본 생성자
    #def __init__(self):
    #    pass #있어도 되고 없어도 된다.

    # 멤버 메서드
    def volumeUp(self):
        self.volume += 1
    def volumeDown(self):
        self.volume -= 1
    def channelUp(self):
        self.channel += 1
    def channelDown(self):
        self.channel -= 1

    def changePower(self):
        self.power = not(self.power) #현재 파워가 가지고 있는 값을 반대값으로 바꾸겠다는 뜻

    # 멤버변수 초기화 메서드
    def data(self, channel, volume, color):
        self.channel = channel
        self.volume = volume
        self.color = color

    # TV 정보 출력 메서드
    def display(self):
        print("전원 : {}, 채널 : {}, 볼륨 : {}, 색상 : {}"
              .format(self.power, self.channel, self.volume, self.color))

# 객체 생성
tv1 = TV() # 기본 생성자 -> 객체
tv1.display()
# 전원 : False, 채널 : 0, 볼륭 : 0, 색상 : None
tv1.data(5, 10, 'Black') # 멤버 변수 값 수정 #tv1 = TV(5, 10, '검정색') 한번에 객체 생성
tv1.display() # 전원 : False, 채널 : 5, 볼륨 : 10, 색상 : Black
tv1.changePower() # off -> on
tv1.channelUp() # 5->6
tv1.volumeUp() # 10->11
tv1.display() # 전원 : True, 채널 : 6, 볼륨 : 11, 색상 : Black

'''
문) tv2 객체를 다음과 같이 생성하시오.
 단계1 : 전원 : False, 채널 : 1, 볼륨 : 1, 색상 : 파랑색 - data()
 단계2 : 전원 : Ture, 채널 : 10, 볼륨 : 15 - 리모콘 메서드
 단계3 : tv2 객체 정보 출력 : display()
'''

tv2 = TV()
tv2.display()
tv2.data(1, 1, 'Blue')
#tv2 = TV(1, 1, '파랑색') 이런 식으로 한번에 객체 생성 가능
tv2.display()

tv2.changePower()
for i in range(9):
    tv2.channelUp()
for i in range(14):
    tv2.volumeUp()

tv2.display() # tv 상태 출력
# 전원 : True, 채널 : 10, 볼륨 : 15, 색상 : 파랑색





####################생성자 넣어준 코드##############################

# TV class 정의
class TV:  # class = 변수(명사, 자료) + 메서드(동사, 기능(자료처리 기능))
    # 멤버변수 : 자료 저장
    channel = volume = 0  # 숫자값이니 0으로 초기화
    power = False  # 켜고 끄는 것 뿐이니 참/거짓으로 초기화 # off(False) -> on(True)
    color = None  # null 값

    # 생성자 : 객체 + 초기화 메서드
    def __init__(self, channel, volume, color):
        self.channel = channel
        self.volume = volume
        self.color = color

    # 멤버 메서드
    def volumeUp(self):
        self.volume += 1

    def volumeDown(self):
        self.volume -= 1

    def channelUp(self):
        self.channel += 1

    def channelDown(self):
        self.channel -= 1

    def changePower(self):
        self.power = not (self.power)  # 현재 파워가 가지고 있는 값을 반대값으로 바꾸겠다는 뜻

    # 멤버변수 초기화 메서드
    def data(self, channel, volume, color):
        self.channel = channel
        self.volume = volume
        self.color = color

    # TV 정보 출력 메서드
    def display(self):
        print("전원 : {}, 채널 : {}, 볼륨 : {}, 색상 : {}"
              .format(self.power, self.channel, self.volume, self.color))

#########################################################




