'''
문) emp.csv 파일을 읽어와서 다음 출력 예시와 같이 처리하시오. 
 
       <<출력 예시>>
관측치 길이 :  5
전체 평균 급여 : 370.0
==============================
최저 급여 : 150, 이름 : 홍길동
최고 급여 : 500, 이름 : 강감찬
==============================
'''
import pandas as pd

# 1. file read
emp = pd.read_csv('../data/emp.csv', encoding='utf-8')
print(emp.info())

pay = emp['Pay']
name = emp['Name']
print('관측치 길이 : ', len(emp))
pay = emp.Pay
print('전체 사원 평균 급여 :', pay.mean())
min_pay = pay.min() # 150
max_pay = pay.max() # 500
name = emp.Name # 이름 벡터

# for문 형식
print('='*30)
for i, p in enumerate(pay) : # index, pay
    if p == min_pay :
        print(f"최저 급여 : {p}, 이름 : {name[i]}")
    if p == max_pay :
        print(f"최고 급여 : {p}, 이름 : {name[i]}")
print('='*30)

# index 형식
print("\nindex 형식")
print('='*30)
min_name = name[pay==min_pay] # ['홍길동']
print(f"최저 급여 : {min_pay}, 이름 : {min_name.values[0]}")
max_name = name[pay==max_pay]#[0]
print(f"최고 급여 : {max_pay}, 이름 : {max_name.values[0]}")
print('='*30)



