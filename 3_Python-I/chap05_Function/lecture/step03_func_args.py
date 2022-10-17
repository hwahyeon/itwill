'''
함수의 가변인수
 - 한 개의 가인수로 여러 개의 실인수를 받는 인수 ex)2개 이상의 값을 하나의 객체로 받을 때
 형식) def 함수명(*인수)
'''

#1. tuple형으로 받는 가변인수
def Func1(name, *names):
    print(name) # 홍길동
    print(names) # ('이순신', '유관순')
    print(names[0]) #이순신 // 순서를 활용할 수 있다.

Func1('홍길동', '이순신', '유관순')

# 패키지, 모듈
import scatter.scatter_module # 방법1) import 패키지명.모듈 or 모듈
from scatter.scatter_module import Avg, var_std # 방법2) from 패키지.모듈 import 함수1, 함수2, 클래스1, 클래스2

#방법1)
scatter.scatter_module.Avg()
#방법2)
Avg()

datas = [2,3,4,5,6,7,8]
avg1 = scatter.scatter_module.Avg(datas) #방법1)
avg2 = Avg(datas) #방법2)
print(avg1)
print(avg2)

var_std(datas) #(4.666666666666667, 2.160246899469287) 이 두 값을 따로 받고 싶을 때는
var, std = var_std(datas)
print(var)
print(std)

def statis(func, *data): # * 붙인 것이 가변인수
    if func == 'sum':
        return sum(data) #함수 실행 종료(exit)
    elif func == 'avg':
        return Avg(data)
    elif func == 'var':
        return var_std(data)
    elif func == 'std':
        return var_std(data)
    else:
        return '해당 함수 없음'

print('sum =', statis('sum', 1,2,3,4,5)) #sum = 15
print('avg =', statis('avg', 1,2,3,4,5)) #avg = 3
var, _ = statis('var', 1,2,3,4,5) #받긴 받되, 바로 사용하지 않을 때 _로 받는다.
print('var=', var)

_, std = statis('std', 1,2,3,4,5)
print('std=', std)

# 2. dict형 가변인수
def person(w, h, **other) :
    print('w =', w) #w = 65
    print('h =', h) #h = 175
    print(other) #{'name': '홍길동', 'age': 35} #dict형

person(65, 175, name='홍길동', age=35)

# 3. 함수를 인수로 받기
def square(x):
    return x**2

def my_func(func, datas):
    result = [func(d) for d in datas]
    return result

datas = [1, 2, 3, 4, 5]
print(my_func(square, datas)) # (함수, 데이터셋)
# [1, 4, 9, 16, 25]



