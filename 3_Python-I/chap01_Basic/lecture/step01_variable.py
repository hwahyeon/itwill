#alt + shift + F10 : 전체 스크립트 실행
#alt + shift + e : 줄단위 실행

'''
변수(Variable)
- 형식) 변수명 = 값 or 수식 or 변수명
- 자료를 저장하는 메모리 이름을 별도로 만들어주는 것
- type 선언 없음(R과 동일)
'''

# 1. 변수와 자료
var1 = "Hello python" #string은 " , ' 둘 다 가능
var2 = 'Hello python'
print(var1) # line skip
print(var2)
# 변수 자료형 확인
print(type(var1), type(var2)) # <class 'str'>

var1=100
print(var1, type(var1)) #100 <class 'int'>

var3 = 150.25
print(var3, type(var3)) #150.25 <class 'float'>

var4 = True # False
print(var4, type(var4)) #True <class 'bool'>

# 2. 변수명 작성규칙(낙타체 : 가독성 좋음)
_num10 = 10
_NUM10 = 20
# 대소문자 구분
print(_num10, _NUM10)
print(id(_num10), id(_NUM10)) #140708137762144 140708137762464 변수가 저장된 메모리주소

# 키워드 확인 (키워드는 변수로 사용할 수 없음)
import keyword #모듈 임포트
py_keyword = keyword.kwlist #키워드 반환
print('python keyword', py_keyword)
print(len(py_keyword)) # 키워드 개수 35개를 보여줌. 이것들은 변수명으로 쓰면 안된다.

#낙타체
korScore = 90 #변수 = 상수
matScore = 85 #변수 = 상수
engScore = 75 #변수 = 상수

tot = korScore + matScore + engScore #변수 = 수식
print("tot =", tot) # tot = 250

# 3. 참조변수 : 메모리 객체를 참조하는 주소를 저장하는 변수
x = 150 # 150 객체의 주소를 가지고 있다.
y = 45.23
y2 = y #변수 복제(주소 복제)
x2 = 150# 기존 객체 있으면, 주소 반환

print(x, y, y2, x2) #150 45.23 45.23 150 #변수 내용 출력
print(id(x), id(y), id(y2), id(x2)) #변수 주소 출력
#140708137766624 2889923752208 2889923752208 140708137766624
# x와 x2의 주소가 같고 y와 y2의 주소가 같다.
# y가 가진 주소가 y2에 복제가 된 것. 내용이 복제가 된 것이 아니다. 객체의 주소가 복제된 것.
# 기존의 메모리에 똑같은 객체가 있으면, 매번 똑같은 객체를 만드는 것이 아니다.
# 메모리의 효율을 위해 똑같은 기존 객체의 주소를 가져오는 셈이다. 참조변수의 특징.



