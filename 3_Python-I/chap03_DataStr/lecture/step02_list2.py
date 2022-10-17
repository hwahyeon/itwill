'''
리스트 내포
    list에서 for문 사용
    형식1) 변수 = [실행문 for 변수 in 열거행객체]
    실행순서 : 1. for문 > 2. 실행문 > 3. 변수에 저장

    형식2) 변수 = [실행문 for 변수 in 열거행객체 if 조건식]
    실행순서 : 1. for문 > 2. if 문 > [3. 실행문(조건이 참인 경우만 실행) > 4. 변수에 저장]
    3번과 4번은 상황에 따라 생략이 됨. if문이 참이면 수행되지만 거짓이면 수행되지 않음.

'''
# 형식1) 변수2 = [실행문 for 변수1 in 열거행객체]
# 열거형객체에서 데이터를 하나씩 가져와서 앞의 변수1에 넣고 그 변수1를 실행문에 넣은 후 그 결과를 앞의 변수2에 넣는다.

# x각 변량에 제공
x = [2, 4, 1, 3, 7]
#x**2
data = []
for i in x :
    print(i**2)
    data.append(i**2)
print(data) #[4, 16, 1, 9, 49]

# data2 = [실행문 for i in x]

data2 = [i**2 for i in x] #가독성이 있다.
print(data2) #[4, 16, 1, 9, 49]

#형식2) 변수 = [실행문 for 변수 in 열거행객체 if 조건식]
# 1~100 -> 3의 배수
num = list(range(1,101)) # 1 ~ 100
print(num)

data3 = [ i for i in num if i % 3 == 0] #3의 배수만 저장하겠다는 의미.
print(data3)

# 내장함수 + 리스트 내포
print('sum =', sum(x)) #sum = 17
data4 = [[1,3,5], [4,5], [7,8,9]] #중첩 list

result = [sum(d) for d in data4]
print(result) # [9, 9, 24]




