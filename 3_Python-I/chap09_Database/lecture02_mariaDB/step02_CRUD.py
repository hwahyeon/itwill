'''
CRUD
    Create, Read, Updata, Delete
'''

import pymysql

config = {
    'host' : '127.0.0.1',
    'user' : 'scott',
    'password' : 'tiger',
    'database' : 'work',
    'port' : 3306,
    'charset':'utf8',
    'use_unicode' : True}

# Read 부분

try:
    # 1. db 연동 객체
    conn = pymysql.connect(**config)
    # 2. cursor 객체 : sql문
    cursor = conn.cursor()

    # 3. Read(Select)
    sql = "select * from goods"
    cursor.execute(sql)
    data = cursor.fetchall()

    for row in data:
        print(row[0], row[1], row[2], row[3])

    print('전체 레코드 수 :', len(data))


    # 상품명 조회
    name = input("조회 상품명 입력 :")
    sql = f"select * from goods where name like '%{name}%'" #name은 글자로 표현되어야 해서
    cursor.execute(sql)
    data2 = cursor.fetchall() #레코드가 한 개인 검색 결과를 받아올 때.

    if data2:
        for row in data2:
            print(row) # tuple 타입
    else:
        print('조회 상품 없음')


    # 상품 코드
    code = int(input("조회 코드 입력 :"))
    sql = f"select * from goods where code = {code}"
    cursor.execute(sql)
    row = cursor.fetchone() # 검색된 레코드 1개만 반환
    if row:
        print(row)
    else:
        print('조회 상품 코드 없음')

except Exception as e:
    print("db 연동 error :", e)
    conn.rollback()
finally:
    cursor.close(); conn.close()

##################update
try:
    # 1. db 연동 객체
    conn = pymysql.connect(**config)
    # 2. cursor 객체 : sql문
    cursor = conn.cursor()

    '''
    # 4. Insert
    code = int(input("code :"))
    name = input("name :")
    su = int(input("su :"))
    dan = int(input("dan :"))
    sql = f"insert into goods values({code}, '{name}', {su}, {dan})"
    cursor.execute(sql) #레코드 추가
    conn.commit()
    '''

    # 5. Update : code -> su, dan 수정
    code = int(input("수정 code :"))
    su = int(input("수정 su :"))
    dan = int(input("수정 dan :"))
    sql = f"update goods set su = {su}, dan = {dan} where code = {code}"
    cursor.execute(sql)
    conn.commit()


    # 3. Read(Select)
    sql = "select * from goods"
    cursor.execute(sql)
    data = cursor.fetchall()

    for row in data:
        print(row[0], row[1], row[2], row[3])

    print('전체 레코드 수 :', len(data))

except Exception as e:
    print("db 연동 error :", e)
finally:
    cursor.close(); conn.close()

##################Delete
try:
    # 1. db 연동 객체
    conn = pymysql.connect(**config)
    # 2. cursor 객체 : sql문
    cursor = conn.cursor()

    # 6. Delete : code -> 유뮤 -> 삭제 or '코드 없음'
    code = int(input('삭제 code :'))
    cursor.execute(f"select * from goods where code = {code}")
    row = cursor.fetchone()

    if row:
        cursor.execute(f"delete from goods where code = {code}")
        conn.commit()
    else:
        print('해당 코드 없음')


    # 3. Read(Select)
    sql = "select * from goods"
    cursor.execute(sql)
    data = cursor.fetchall()

    for row in data:
        print(row[0], row[1], row[2], row[3])

    print('전체 레코드 수 :', len(data))

except Exception as e:
    print("db 연동 error :", e)
    conn.rollback()
finally:
    cursor.close(); conn.close()




