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

test = 0 # 검정 데이터
train = [] # 훈련 데이터
for i in range(1, 6):
    dataset = [1, 2, 3, 4, 5]
    test = i
    dataset.remove(i)
    train = dataset
    print("검정 데이터 :", test)
    print("훈련 데이터 :", train)



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
### 내용 채우기 ###
# 함수 정의
def pay_pro(x): 
    from statistics import mean # 평균 
    import re # 정규표현식     
    
    ### 내용 채우기 ###
    ind_pay = [int(re.sub("[1-9][0-9]{3}[가-힣]{3}", "", i)) for i in x]
    name = [re.findall("[가-힣]{3}", i)[0] for i in x]
    pr_pay = []
    pr_name = []
    for i in range(len(x)):
        if ind_pay[i] >= mean(ind_pay):
            pr_pay.append(ind_pay[i])
            pr_name.append(name[i])
    print("전체 급여 평균 :", mean(ind_pay))
    print("평균 이상 급여 수령자")
    for i in range(len(pr_pay)):
        print("%s => %d" % (pr_name[i], pr_pay[i]))


# 함수 호출 
pay_pro(pay)


'''
 chap05_Function 관련 문제 
 [문3] student(3명의 학생 점수)를 이용하여 다음 조건에 맞게 학생관리 프로로그램의
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
    from statistics import mean
    print('\n\t*** 2018년도 2학기 성적처리 결과 ***')
    print('-' * 50)
    print(" 번호\t 국어\t 영어\t 수학\t 총점\t 평균")
    print('-' * 50)
    tot = [0, 0, 0]
    avg = []

    # 평균, 총점 계산
    def tot_age_calc(score):
        nonlocal tot
        nonlocal avg
        for i in range(len(score)):
            for r in range(len(score)):
                tot[i] += score[i][r]
        for i in range(len(score)):
            avg.append((tot[i]) / len(score))


    # 점수 출력
    def score_display(student):
        nonlocal tot, avg
        for i in range(len(student)):
            print(" %d. \t %d \t %d \t %d \t %d \t %d"
                  % (i+1, student[i][0], student[i][1], student[i][2], tot[i], avg[i]))

    return score_display, tot_age_calc
score_display, tot_age_calc = students()
tot_age_calc(student)
score_display(student)
'''
 chap06_Class 관련 문제 
 [문4] 문3의 내용을 클래스로 구현하시오.
'''
hong = [90, 85, 70]
lee = [99, 90, 95]
yoo = [70, 80, 100]
student = [hong, lee, yoo]

class Student :
    def __init__(self, student):
        self.student = student

    def tot_age_calc(self):
        self.tot = [0, 0, 0]
        self.avg = []
        for i in range(len(self.student)):
            for r in range(len(self.student)):
                self.tot[i] += self.student[i][r]
        for i in range(len(self.student)):
            self.avg.append((self.tot[i]) / len(self.student))

    def score_display(self):
        print('\n\t*** 2018년도 2학기 성적처리 결과 ***')
        print('-' * 50)
        print(" 번호\t 국어\t 영어\t 수학\t 총점\t 평균")
        print('-' * 50)
        for i in range(len(self.student)):
            print(" %d. \t %d \t %d \t %d \t %d \t %d"
                  % (i + 1, self.student[i][0], self.student[i][1], self.student[i][2], self.tot[i], self.avg[i]))

a = Student(student)
a.tot_age_calc()
a.score_display()
