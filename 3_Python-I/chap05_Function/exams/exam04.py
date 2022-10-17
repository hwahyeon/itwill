'''
 문4) 패토리얼(Factorial) 계산 함수를 정의하시오.
     예) 5! -> 5*4*3*2*1 : 120
'''

def Factorial(n):
    pass


result_fact = Factorial(5)
print()
print('패토리얼 결과:', result_fact) #  패토리얼 결과: 120


#teacher
def Factorial(n):
    if n == 1:
        return 1
    else:
        result = n * Factorial(n - 1)
        '''
        stack = [5, 4, 3, 2]
        stack 역순 계산 : 1*[2*3*4*5] 
        '''
        print(result, end=' ')  # 2 6 24 120
        return result


result_fact = Factorial(5)
print()
print('패토리얼 결과:', result_fact)  # 패토리얼 결과: 120