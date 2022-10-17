'''
dict 특징
 - set 구조와 유사함(순서 없음 = index 사용 불가)
 - R의 list와 유사함
 - key와 value 한 쌍으로 원소 구성
 - key -> value 참조
 - key 중복 불가, value 중복 가능
형식) 변수 = {key:value, key:value}
'''

# 1.dict 생성

# 방법 1)
dic = dict(key1 = 100, key2 = 200, key3 = 300)
print(dic, len(dic), type(dic)) #{'key1': 100, 'key2': 200, 'key3': 300} 3 <class 'dict'>

# 방법 2)
dic2 = {'name' : '홍길동', 'age': 35, 'addr' : '서울시'}
print(dic2) #{'name': '홍길동', 'age': 35, 'addr': '서울시'}

# 2. 수정, 추가, 삭제, 검색 : key 이용
dic2['age'] = 45 # 수정
dic2['pay'] = 350 # 추가
print(dic2) #{'name': '홍길동', 'age': 45, 'addr': '서울시', 'pay': 350}
del dic2['addr'] # 삭제
print(dic2) #{'name': '홍길동', 'age': 45, 'pay': 350}

# 키 검색
print('age' in dic2) #True

#3. for 이용
for k in dic2: # =for k in dic2.keys():
    print(k)
# key가 넘어오는데 그 순서는 정해진 것이 없다. 항상 랜덤하게 넘어온다.

for k in dic2.keys():
    print(k, end = '->') #key
    print(dic2[k]) #value

for v in dic2.values(): #값 넘김
    print(v)

for k, v in dic2.items(): #키, 값 묶어서 넘김 // 따로따로 순서대로 받음
    print(k, end = '->')
    print(v)

for d in dic2.items(): #키, 값 하나의 변수로 받음
    print(d)

#('name', '홍길동')
#('age', 45)
#('pay', 350) 튜플 형식으로 반환됨

# 4. key -> value
print(dic2['name']) #index 형식 : 홍길동
print(dic2.get('name')) #get() : 홍길동

# 5. {'key' : [value1, value2]} - {'이름' : [급여, 수당]}
emp = {'hong' : [250, 50], 'lee' : [350, 80], 'yoo' : [200,40]}
print(emp)

for k, v in emp.items(): #key, value
    print(k, end=' ')
    print(v)

# 급여가 250 이상인 사원 정보 출력
for k, v in emp.items(): #key, value
    if v[0] >= 250:
        print(k, end='->')
        print(v)

# 급여가 250 이상인 경우 사원명, 수당 합계
su=0
for k, v in emp.items():
    if v[0] >= 250:
        print(k)
        su += v[1]
print('수당 합계 = ',su)

# 6. 문자 빈도수 구하기 (word count)
charset = ['love', 'test', 'love', 'hello', 'test', 'love']
print(len(charset)) #6
wc = {} # 빈 set

# 방법1)
for word in charset:
    if word in wc:
        wc[word] += 1 #2회 이상 발견 : 1씩 증가
    else:
        wc[word] = 1 #최초 발견 : 1을 초기화
print('워드 카운트 : ', wc)
#워드 카운트 :  {'love': 3, 'test': 2, 'hello': 1} #순서는 실행할 때마다 랜덤하게 나옴.

print(max(wc, key=wc.get)) #love 출연 빈도 수가 가장 높은 것을 추출하겠다는 뜻.

# 방법 2)
wc2 = {} # 빈set
for word in charset:
    # key = value
    wc2[word] = wc2.get(word,0)+1 #get(key값)

print(wc2) # {'love': 3, 'test': 2, 'hello': 1}





