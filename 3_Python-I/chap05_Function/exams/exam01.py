'''
문1) Start counter 문제  

height : 3 <- 키보드 입력 
*
**
***
star 개수 : 6
'''

# 함수 정의
def StarCount(height):
    no_star = sum(list(range(1, height + 1)))
    for i in list(range(1, height+1)):
        print('*'*i, end='\n')
    print('star 개수 :', no_star)
    return no_star

# 키보드 입력 
height = int(input('height : ')) # 층 수 입력 

# 함수 호출 및 start 개수 출력  
print('start 개수 : %d'%StarCount(height))

#teacher
# 함수 정의
def StarCount(height):
    s_cnt = 0  # 별 개수 카운트

    ## 내용 채우기 ##
    for i in range(height):  # 층수 만큼 반복
        print('*' * (i + 1))  # '*'*반복수
        s_cnt += (i + 1)

    return s_cnt