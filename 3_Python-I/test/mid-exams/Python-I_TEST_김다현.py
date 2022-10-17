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

for i in range(5) :
    test = dataset[i]
    train = dataset[:i] + dataset[i+1:]
    print('검정 데이터 :',test)
    print('훈련 데이터 :',train)


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
    from re import findall
    p = []
    for e in x :
        pro = findall("[0-9]{3}", e)
        p.append(int(pro[1]))
    mu = mean(p)
    print('전체 급여 평균 :',mu)
    print('평균 이상 급여 수령자')
    for e in x :
        name = findall("[가-힣]{3}", e)
        pro = findall("[0-9]{3}", e)
        if int(pro[1]) >= mu :
            print(name[0],'=>',pro[1])

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
    print('\n\t*** 2018년도 2학기 성적처리 결과 ***')
    print('-' * 50)
    print(" 번호\t 국어\t 영어\t 수학\t 총점\t 평균")
    print('-' * 50)

    # 평균, 총점 계산
    def tot_age_calc(score):
        from statistics import mean
        s = sum(score)
        m = mean(score)
        return s, round(m,2)

    # 점수 출력
    def score_display(student):
        cnt = 1
        for i in student :
            s,m = tot_age_calc(i)
            print('\t',str(cnt)+'.\t',i[0],'\t',i[1],'\t',i[2],'\t',s,'\t',m)
            cnt += 1

    return score_display

stu = students()
ans = stu(student)
print('-' * 50)

'''
 chap06_Class 관련 문제 
 [문4] 문3의 내용을 클래스로 구현하시오.
'''

class Student :

    print('\n\t*** 2018년도 2학기 성적처리 결과 ***')
    print('-' * 50)
    print(" 번호\t 국어\t 영어\t 수학\t 총점\t 평균")
    print('-' * 50)

    def tot_age_calc(self, score):
        self.score = score
        from statistics import mean
        s = sum(self.score)
        m = mean(self.score)
        return s, round(m,2)

    # 점수 출력
    def score_display(self, student):
        self.student = student
        cnt = 1
        for self.i in self.student:
            s, m = self.tot_age_calc(self.i)
            print('\t', str(cnt) + '.\t', self.i[0], '\t', self.i[1],
                  '\t', self.i[2], '\t', s, '\t', m)
            cnt += 1
        print('-' * 50)

stu = Student()
print(stu.score_display(student))



