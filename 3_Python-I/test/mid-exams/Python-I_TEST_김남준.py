'''
Python-I TEST
 파일명 : Python-I_TEST_홍길동.py
 메일 전송 : kpjiju@naver.com
'''

'''
chap02_Control ~ chap03_DataSet 관련문제

[문1] 교차검정 dataset 생성하기
  - 교차검정 : train과 test 셋을 cross check 하여 모델을 검정하는 방법

<< 출력 화면 예시>>
검정 데이터 : 1
훈련 데이터 : [2, 3, 4, 5]
검정 데이터 : 2
훈련 데이터 : [1, 3, 4, 5]
검정 데이터 : 3
훈련 데이터 : [1, 2, 4, 5]
검정 데이터 : 4
훈련 데이터 : [1, 2, 3, 5]
검정 데이터 : 5
훈련 데이터 : [1, 2, 3, 4]
'''
dataset = [1,2,3,4,5] # 교차검정 dataset

for idx, val in enumerate(dataset):
    test = val
    train = dataset[:idx] + dataset[idx+1:]
    print(f'검정 데이터 : {test}\n훈련 데이터 : {train}')

print('\n\t*** 1번 문제 2번 문제 경계선 ***\n')


'''
- chap04_regExText ~ chap05_Function 관련 문제
[2문] 다음 벡터(pay)는 '입사년도사원명급여'순으로 사원의 정보가 기록된 데이터 있다.
      이 벡터 데이터를 이용하여 아래와 같은 출력결과가 나타나도록 함수를 정의하시오. 

   <출력 결과>
 전체 급여 평균 : 260
 평균 이상 급여 수령자
 이순신 => 300 
 유관순 => 260 
'''

pay = ["2014홍길동220", "2002이순신300", "2010유관순260"]

# 함수 정의
def pay_pro(x): 
    from statistics import mean # 평균 
    import re # 정규표현식

    name = list()
    sal = list()
    for i in x:
        name.append(re.findall('[가-힣]+', i)[0])
        sal.append(int(re.findall(r'\d+$', i)[0]))

    print(f'전체 급여 평균 : {mean(sal)}\n평균 이상 급여 수령자')

    for idx, val in enumerate(sal):
        if val >= mean(sal):
            print(f'{name[idx]} => {sal[idx]}')

# 함수 호출 
pay_pro(pay)

print('\n\t*** 2번 문제 3번 문제 경계선 ***')

'''
 chap05_Function 관련 문제 
 [문3] student(3명의 학생 점수)를 이용하여 다음 조건에 맞게 학생관리 프로그램의
       함수로 완성하시오.
  <조건1> outer : students() -> 제목(title) 출력 , inner 함수 포함  
  <조건2> inner : tot_age_calc()  -> 총점과 평균 계산 반환
          inner : score_display() -> 학생 이름과 과목점수, 총점, 평균 출력 
  <조건3> 기타 나머지는 출력 예시 참조           

            <<출력 예시>>
    *** 2018년도 2학기 성적처리 결과 ***
-----------------------------------------    
 번호  국어   영어  수학   총점    평균
-----------------------------------------
  1.   90    85    70    245    81.67
  2.   99    90    95    284    94.67
  3.   70    80    100   250    83.33
------------------------------------------
'''
#  [국어,영어,수학]
hong = [90, 85, 70]
lee = [99, 90, 95]
yoo = [70, 80, 100]
student = [hong, lee, yoo]

def students():
    print('\n\t*** 2018년도 2학기 성적처리 결과 ***')
    print('-' * 50)
    print(" 번호\t 국어\t 영어\t 수학\t 총점\t 평균")
    print('-' * 50)

    # 평균, 총점 계산
    def tot_age_calc(score):
        return sum(score), sum(score)/len(score)

    # 점수 출력
    def score_display(student):
        for idx, val in enumerate(student):
            # print(f"{i+1}. {i[0]} {i[1]} {i[2]} {sum(i)} {format(sum(i) / 3, '4.2f')}")
            print(f" {format(idx + 1, '3d')}.    {format(val[0], '3d')}     {format(val[1], '3d')}     "\
                  f"{format(val[2], '3d')}      {format(tot_age_calc(val)[0], '3d')}    "\
                  f"{format(tot_age_calc(val)[1], '4.2f')}")
        print('-' * 50)

    return score_display

students()(student)

print('\n\t*** 3번 문제 4번 문제 경계선 ***')

'''
 chap06_Class 관련 문제 
 [문4] 문3의 내용을 클래스로 구현하시오.
'''

class Student :
    def __init__(self, students):
        self.students = students

    # Method 'tot_age_calc' may be 'static'
    def tot_age_calc(self, score):
        return sum(score), sum(score)/len(score)

    def score_display(self):
        print('\n\t*** 2018년도 2학기 성적처리 결과 ***')
        print('-' * 50)
        print(" 번호\t 국어\t 영어\t 수학\t 총점\t 평균")
        print('-' * 50)
        for idx, val in enumerate(self.students):
            # print(f"{i+1}. {i[0]} {i[1]} {i[2]} {sum(i)} {format(sum(i) / 3, '4.2f')}")
            print(f" {format(idx + 1, '3d')}.    {format(val[0], '3d')}     {format(val[1], '3d')}     "\
                  f"{format(val[2], '3d')}      {format(self.tot_age_calc(val)[0], '3d')}    "\
                  f"{format(self.tot_age_calc(val)[1], '4.2f')}")
        print('-' * 50)

Student(student).score_display()

