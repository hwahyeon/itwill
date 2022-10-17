'''
1. private 변수 = 은닉변수 : 클래스 내에 은닉변수를 만들 때 사용.
    object.member : 객체 -> 은닉변수로 접근 불가
    getter()/setter()를 이용해서 은닉변수로 접근
2. class 포함관계(inclusion)
    - 특정 객체가 다른 객체를 포함하는 클래스 설계 기법.
    - 두 객체 간의 통신 지원
    - ex) class A(a) <-> class B(b)
'''

# 1. private 변수
class Login : # uid, pwd -> db['hong', '1234'] 저장
    # 생성자
    def __init__(self, uid, pwd):
        #self.__uid    #__를 쓰면 private변수가 됨.
        self.__dbId = uid
        self.__dbPwd = pwd

    #getter() : 획득자(return)  // 확인가능
    def getIdPwd(self): #getter를 이용해서 은닉변수를 끌어다 쓸 수 있다.
        return self.__dbId, self.__dbPwd

    #setter() : 값을 수정, 지정한다 해서 지정자라고 함. // 수정가능능
    def seIdPwd(self, uid, pwd):
        self.__dbId = uid
        self.__dbPwd = pwd

# object
login = Login('hong', '1234')
# object.member
# print(login.__dbId) #객체가지고 수정할 수 없음.

# object.getter()
uid, pwd = login.getIdPwd()
print(uid, pwd, sep = ',') # hong, 1234

# object.setter(인수)
login.setIdPwd('lee', '2345') # 변수 수정

uid, pwd = login.getIdPwd() # 변수 확인
print(uid, pwd, sep = ',') # lee,2345

# Server <-> Login
class Server :
    # 기본 생성자

    # 멤버 메서드
    def send(self, obj): # object 인수로 받음
        self.obj = obj # 멤버변수 생성

    # 인증 메서드
    def cert(self, uid, upwd): # 사용자(id/pwd)
        dbId, dbPwd = self.obj.getIdPwd() # getter 호출
        if dbId == uid and dbPwd == upwd :
            print('로그인 성공~~~')
        else:
            print('로그인 실패~~~')

server = Server()
server.send(login) # object 넘김
server.cert('hong', '1234') # 실패
server.cert('lee','2345') # 성공












