'''
예외는 프로그램 실행 상태에서 예기치 않는 상황(오류)를 말한다.
코드를 작성중에는 알 수 없으니까 실행했을 때 예외를 처리하도록 만든 것.
에러가 나는 부분은 넘어가고 코드를 중단하지 않고 쭉 실행하도록 할 때 쓰임.

try :
    예외발생 코드
except:
    예외처리 코드
finally: (생략 가능한 블록)
    항상처리 코드(예외가 발생하건 하지 않건 항상 처리되는 코드)

'''

print('프로그램 시작')
x = [10, 20, 35.5, 15, 'num', 14.5]

for i in x:
    print(i)
    y = i ** 2
    print('y =', y)

print('프로그램 종료')

# 1. 간단한 예외처리
for i in x:
    try:
        print(i)
        y = i ** 2
        print('y =', y)
    except:
        print('예외발생(숫자 아님) : ', i)
print('프로그램 종료')

# 2. 유형별 예외처리
print('유형별 예외처리')

try:
    div = 1000 / 2.5 #정상
    print('div = %3.f'%div)
    div2 = 1000 / 0 #산술적 예외
    print('div2 = %3.f'%div2) #여기서 에러가 나서 '예외발생'이 뜨게 됨.
except:
    print('예외발생')
finally:
    print('프로그램 종료')

##### ZeroDivisionError / FileNotFoundError #####
print('유형별 예외처리')

try:
    #div = 1000 / 2.5 #정상
    #print('div = %3.f'%div)
    #div2 = 1000 / 0 # 1차 : 산술적 예외
    #print('div2 = %3.f'%div2) # 여기서 에러가 나서 '예외발생'이 뜨게 됨.
    #f = open('c:/text.txt') # 2차 : 파일 열기
    num = int(input("숫자입력 :")) # 3차 : 기타 예외발생
    print('num =', num)

except ZeroDivisionError as e: # 0으로 나누는 에러를 잡아서 e로 넘김.
    print('예외발생 :', e) # division by zero

except FileNotFoundError as e : # 파일을 찾을 수 없을 때. #class as object
    print('예외발생 :', e) # [Errno 2] No such file or directory: 'c:/text.txt'

except Exception as e:
    print('기타 예외발생 :', e) # 숫자를 입력 안했을 때 : invalid literal for int() with base 10: 'r'
    # 상세한 예외 정보를 줌.

finally:
    print('프로그램 종료')

















