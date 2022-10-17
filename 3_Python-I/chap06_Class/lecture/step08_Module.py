'''
패키지(package) = 폴더 유사함
  - 유사한 모듈 파일을 저장하는 공간
모듈(module) = 파일 유사함
  - 파이썬 파일(*.py)
클래스, 함수
  - 모듈에 포함되는 단위
'''

#from 패키지명.모듈명 import 클래스 or 함수
from package_test.module01 import Sub #class
from package_test.module01 import Adder #function

sub = Sub(10, 20) # 생성자 -> 객체
print('sub =', sub.calc())

print('add =', Adder(10,20)) #함수 호출









