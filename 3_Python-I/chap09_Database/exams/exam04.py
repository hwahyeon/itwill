'''
문제4) emp.csv 파일을 읽어서 다음과 같이 db 테이블에 저장하시오.
 <조건1> 테이블명 : emp_table
 <조건2> 사원 이름으로 레코드 조회(sql문 작성)
 
 <작업순서>
 1. table 생성 : emp_table(sql 폴더)
 2. python code : 레코드 추가 
 3. python code : 레코드 조회(사원이름)  
'''

import pandas as pd

# 칼럼 단위 읽기 
emp = pd.read_csv("../data/emp.csv", encoding='utf-8')
print(emp)
print(emp.info())
# No Name  Pay
no = emp.No
name = emp.Name
pay = emp.Pay
print(no, name, pay)

import pymysql

config = {
    'host': '127.0.0.1',
    'user': 'scott',
    'password': 'tiger',
    'database': 'work',
    'port': 3306,
    'charset': 'utf8',
    'use_unicode': True}
try:
    # 1. db 연동 객체
    conn = pymysql.connect(**config)
    # 2. cursor 객체 : sql문
    cursor = conn.cursor()

    # 3. table 생성
    '''
    sql = """create or replace table emp_table(
    eno int primary key,
    name varchar(20) not null,
    pay int not null
    )"""
    cursor.execute(sql)
    print('1. table 작성 완료~~~')
    '''

    # 4. 레코드 조회




    if data:  # True : 검색
        for row in data:
            print("%d    %s   %d" % row)

        print('전체 레코드 수 : ', len(data))

        ### 사원 검색
        name = input("사원 이름 :")
        sql = f"select * from emp_table where name like '%{name}%'"
        cursor.execute(sql)
        data2 = cursor.fetchall()
        if data2:
            for row in data2:
                print("%d    %s   %d" % row)

            print('검색된 레코드 수:', len(data2))
        else:
            print('검색된 레코드 없음')

    else:  # False : 레코드 추가
        for i in range(len(no)):
            eno = no[i];
            ename = name[i];
            epay = pay[i]
            sql = f"""insert into emp_table 
                        values({eno},'{ename}',{epay})"""
            cursor.execute(sql)
            conn.commit()
        print('2. 레코드 추가 성공~~')
except Exception as e:
    print('db error :', e)
finally:
    cursor.close()
    conn.close()





    





