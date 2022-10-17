'''
기존 중첩함수 -> 클래스(data + function)
'''

class Account : #클래스는 인수를 받을 수 없다. #outer -> class
    #outer 변수 -> 멤버변수
    balance = 0 # 잔액(balance) 명시적일뿐 이 라인도 필요 없다.

    # 생성자 : 잔액 초기화
    def __init__(self, bal, name, no):
        self.balance = bal # 멤버변수 초기화
        self.accName = name # 동적멤버변수
        self.accNo = no # 동적멤버변수

    #inner -> 멤버메서드 (클래스에서 정의한 모든 함수는 self를 가지고 있어야 한다.)
    def getBalance(self): # 잔액확인(getter)
        return self.balance, self.accName, self.accNo

    def deposit(self, money) : # 입금하기(setter)
        if money < 0 :
            print("금액을 확인하세요.")
        else :
            #nonlocal balance 여기선 nonlocal을 self가 대신하는 격.
            self.balance += money

    def withdraw(self, money) : # 출금하기(setter)
        #nonlocal balance
        if self.balance < money :
            print("잔액이 부족합니다.")
        else :
            self.balance -= money

acc = Account(1000, '홍길동', '012-125-41520')
bal, name, no = acc.getBalance()
print('잔액 : {}, 예금주 : {}, 계좌번호 : {}'.format(bal, name, no)) # getter : 잔액 : 1000

acc.deposit(20000) # 2만원 입금
bal, name, no = acc.getBalance()
print('잔액 : {}, 예금주 : {}, 계좌번호 : {}'.format(bal, name, no)) # getter : 잔액 : 1000

acc.withdraw(5000) # 5천원 인출
bal, name, no = acc.getBalance()

'''
1. 예금자(accName), 계좌번호(accNo) 동적 멤버변수 추가하기
    -> 예금자 : 홍길동, 계좌번호 : 012-125-41520
2. getBalance() 메서드를 이용하여 잔액, 예금주, 계좌번호 출력
'''


