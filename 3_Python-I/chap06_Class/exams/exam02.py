'''
 문) 동적 멤버 변수 생성으로 다음과 같은 산포도를 구하는 클래스를 정의하시오.
 
class Scattering :         
        
        생성자 
        
        분산 함수(var_func)
        
        표준편차 함수(std_func) 
        
        
   << 출력 결과 >>
 분산 : 7.466666666666666
 표준편차 :  2.7325202042558927
'''

from statistics import mean
from math import sqrt

x = [5, 9, 1, 7, 4, 6]

from statistics import mean
from math import sqrt

class Scattering :
     #생성자 : 객체 + 동적멤버
     def __init__(self, x):
         self.x = x #동적멤버 변수 생성

     # 분산 함수(var_func)
     def var_func(self):
        mu = sum(self.x[:]) / len(self.x)
        diff = [(i-mu)**2 for i in self.x]
        self.var = sum(diff) / (len(self.x) - 1)

     # 표준편차
     def std_func(self):
        self.std = sqrt(self.var)
        

x = [5, 9, 1, 7, 4, 6]
scatter = Scattering(x)
scatter.var_func() #분산 계산
print('분산 :', scatter.var)
scatter.std_func()
print('표준 편차 :', scatter.std)