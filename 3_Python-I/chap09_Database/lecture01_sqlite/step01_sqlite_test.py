'''
sqlite3
    - 내장형 DBMS : 기기 내부에서만 사용
    - 외부 접근 허용 안됨
'''
import os
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap09_Database')


import sqlite3

print(sqlite3.version_info) #(2, 6, 0)
print(sqlite3.sqlite_version_info) #(3, 31, 1) 버젼에 관한 정보

try:
    # 1. database 생성 & db 연동 객체
    conn = sqlite3.connect("./data/sqlite.db") #sqlite.db
    # sql문 실행 객체
    cursor = conn.cursor()

    # 2. table 생성 # if not exists, sqlite의 특징.
    sql = """create table if not exists test_tab(name text(10),
    phone text(15), addr text(50) )"""
    cursor.execute(sql) #table 생성

    # 3. 레코드 추가
    '''
    cursor.execute("insert into test_tab values('홍길동', '010-111-1111', '서울시')")
    cursor.execute("insert into test_tab values('이순신', '010-111-1111', '해남시')")
    cursor.execute("insert into test_tab values('유관순', '010-111-1111', '충남시')")
    conn.commit() # db 반영 (commit은 데이터베이스 구조를 변경시킬 때만 사용)
    '''

    # 4. 레코드 조회
    cursor.execute("select * from test_tab")
    dataset = cursor.fetchall() # 객체 레코드 -> 레코드 가져오기
    for row in dataset:
        print(row) # tuple

    print("이름 \t\t 전화번호 \t\t 주소")
    for row in dataset:
        print(row[0]+'\t'+row[1]+'\t'+row[2])

except Exception as e:
    print('db 연동 오류 :', e)
    conn.rollback() # 이전 상태로 되돌림
finally:
    cursor.close()
    conn.close()



