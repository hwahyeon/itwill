'''
문자열 처리
 - 문자열(string) : 문자들의 순서(index) 집합
 - indexing/slicing 가능
 - 문자열 = 상수 : 수정 불가
'''
# 1. 문자열 처리

# 1) 문자열 유형
lineStr = "this is one line string" # 한 줄 문자열
print(lineStr)

# 여러 줄 문자열
multiStr = '''this
is multi line
string'''
print(multiStr)

multiStr2 = 'this \nis multi line\n string' #\n
print(multiStr2)

# sql문 : 부서번호
deptno = int(input('부서번호 입력 : '))
query = f'''select * from emp
where deptno = {deptno}
order by sel desc'''
print(query)

# 2) 문자열 연산 (+, *)
print('python' + 'program') #결합연산자 : python program
#print('python' + 37)
print('python' + str(37)) #int -> str('37') : python37
print('-'*30) # 30번 반복

'''
object.member or object.member()
int.member
str.member
'''

# 3. 문자열 처리 함수
lineStr = "this is one line string"
print(lineStr, type(lineStr)) #내용, 자료형 출력
# this is one line string <class 'str'>

print('문자열 길이 : ', len(lineStr)) # 문자열 길이 :  23
print('t의 글자수 : ', lineStr.count('t')) # t의 글자수 :  2

# 접두어 : 시작문자열
print(lineStr.startswith('this')) #True 시작을 this로 하는 것이 맞는지
print(lineStr.startswith('that')) #False

# 문자열 분리 : split : 토큰 생성
# 1) 문장 -> 단어
words = lineStr.split(sep=' ') # ['this', 'is', 'one', 'line', 'string']
words
print('단어 길이 : ', len(words))

# 2) 문단 -> 문장
multiStr = '''this
is multi line
string'''

sentences = multiStr.split(sep='\n') #sep구분자를 무엇으로 쓰는지에 따라 토큰 형태가 달라짐
print(sentences) #['this', 'is multi line', 'string']
print('문장 길이 : ', len(sentences)) #문장 길이 :  3

# 결합(join) : '구분자'.join(str)
sentence = ' '.join(words) #공백을 가지고 결합한다는 의미. 1) 단어 -> 문장
print(sentence) # this is one line string

para = ','.join(sentences) # 2) 문장 -> 문단
print(para) #this,is multi line,string

print(multiStr.upper()) #소문자 -> 대문자


# 4) indexing/scling
print(lineStr)
# this is one line string
print(lineStr[0]) #첫번째 문자 - t
print(lineStr[-1]) #마지막 문자 - g

print(lineStr[:4]) #this #4 이전의 것들을 가져옴. 즉 0번 1번 2번 3번까지 가져오는 것. // [start:end-1]
print(lineStr[-6:]) #string // 오른쪽 끝 6개 슬라이싱

# 2. escape 문자 처리
'''
escape 문자 : 명령어 이외 특수문자(', ", \n, \t, \b)
'''
print("\nescape 문자")
print("\\nescape 문자") #\nescape 문자
print(r"\nescape 문자") #\nescape 문자

# c:\python\work\test
print('c:\python\work\test') #c:\python\work	est
print('c:\\python\\work\\test') #c:\python\work\test
print(r'c:\python\work\test') #c:\python\work\test


