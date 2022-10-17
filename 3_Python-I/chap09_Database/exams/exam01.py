'''
문제1) goods 테이블을 이용하여 다음과 같은 형식으로 출력하시오.  
 <조건1> 전자레인지 수량, 단가 수정 
 <조건2> HDTV 수량 수정 

    [ goods 테이블 현황 ]
1 냉장고 2 850000
2 세탁기 3 550000
3 전자레인지 5  600000 <- 수량, 단가 수정
4 HDTV 2 1500000  <- 수량 수정
전체 레코드 수 : 4
'''

import sqlite3

try :
    # 1. db 연동 객체
    conn = sqlite3.connect("../data/sqlite.db")  # sqlite.db
    # sql문 실행 객체
    cursor = conn.cursor()

    sql = "update goods set su=5, dan=600000 where code=3"
    cursor.execute(sql)
    sql = "update goods set su=2 where code=4"
    cursor.execute(sql)
    conn.commit()  # db 반영

    # 레코드 조회 : 전체 레코드 조회
    cursor.execute("select * from goods")
    dataset = cursor.fetchall()
    for row in dataset:
        print("%d    %s    %d    %d" % (row))

    print('전체 레코드 수 :', len(dataset))

except Exception as e :
    print('db 연동 error :',e)
    conn.rollback()

finally:
    cursor.close()
    conn.close()




