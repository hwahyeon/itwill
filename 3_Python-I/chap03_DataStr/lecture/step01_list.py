'''
list 특징
 - 1차원 배열 구조
   형식) 변수 = [값1, 값2, ... ]
 - 다양한 자료형 저장 가능
 - index 사용, 순서 존재
   형식) 변수[index], index=0
 - 값 수정(추가, 삽입, 수정, 삭제)
'''

# 1. 단일 list
lst = [1, 2, 3, 4, 5]
print(lst, type(list), len(lst)) #[1, 2, 3, 4, 5] <class 'type'> 5

for i in lst :
    #print(i, end=' ') # 1 2 3 4 5
    print(lst[i-1: ]) # 변수[start:]

for i in lst:
    # print(i, end=' ') # 1 2 3 4 5
    print(lst[:i]) #변수[:stop-1]

'''
처음/마지막 데이터 추출
'''
x = list(range(1, 101)) # 1 ~ 100
print(x)
print(x[:5]) #[1, 2, 3, 4, 5]  / 인덱스 0부터 5개
print(x[-5:]) #[96, 97, 98, 99, 100] # 마지막 5개 원소

'''
index 형식
변수[start=0:stop-1:step=1] #step = -1은 감소
'''

print(x[:]) #전체 데이터
print(x[::2]) #[::step=2] (2씩 증가) : 홀수
print(x[1::2]) #[start::step=2] : 짝수

# 2. 중첩 list : [[], []]
a = ['a', 'b', 'c']
b = [10, 20, a, 5, True, "hong"] #서로 다른 type
print(b) # [10, 20, ['a', 'b', 'c'], 5, True, 'hong']
print(b[2]) #['a', 'b', 'c']
print(b[2][2]) #c
print(b[2][1:]) #['b', 'c']

print(type(a), type(b)) #<class 'list'> <class 'list'>
print(id(a), id(b)) #2352808179592 2352808152840

# 3. 값 수정(추가, 삽입, 수정, 삭제_
num = ['one', 'two', 'three', 'four']
print(len(num)) # 4

num.append('five') #원소 추가
print(num)  # ['one', 'two', 'three', 'four', 'five']
num.remove('five') #원소 삭제
print(num)  # ['one', 'two', 'three', 'four']
num.insert(0, 'zero') #원소 삽입
print(num) # ['zero', 'one', 'two', 'three', 'four']
num[0] = 0 # 원소 수정
print(num)  # [0, 'one', 'two', 'three', 'four']

# 4. list 연산(+, *)

# 1) list 결합
x = [1, 2, 3, 4]
y = [1.5, 2.5]
z =x + y # new object
print(z, type(z)) # [1, 2, 3, 4, 1.5, 2.5] <class 'list'>

# 2) list 확장
x.extend(y) # 기존 object
print(x) #[1, 2, 3, 4, 1.5, 2.5] - 단일 list

# 3) list 추가
x.append(y) # 기존 object
print(x) #[1, 2, 3, 4, 1.5, 2.5, [1.5, 2.5]] - 중첩 list

# 4) list 곱셈(*)
lst = [1, 2, 3, 4]
result = lst * 2 #곱셈은 반복의 의미 # 2번 반복
print(result) #[1, 2, 3, 4, 1, 2, 3, 4]

# 5) list 정렬
result.sort() #기본값 오름차순
print(result) #[1, 1, 2, 2, 3, 3, 4, 4]
result.sort(reverse=True)# True 내림차순, False 오름차순
print(result) #[4, 4, 3, 3, 2, 2, 1, 1]

# 6) scala vs vector
'''
scala 변수 : 한 개의 상수(값)를 갖는 변수(크기)
vector 변수 : 다수의 값을 갖는 변수(크기, 방향)
'''
dataset=[] # 빈 list : vector
size = int(input("vector size : ")) # scala 변수

for i in range(size) : #range(5) : 0~4
    dataset.append(i+1) #vector 변수

print(dataset) #[1, 2, 3, 4, 5]

# 7) list에서 원소 찾기
'''
if 값 in list:
   참 실행문
else 
   거짓 실행문
'''

if 5 in dataset:
    print("5가 있음")
else :
    print("5가 없음")































