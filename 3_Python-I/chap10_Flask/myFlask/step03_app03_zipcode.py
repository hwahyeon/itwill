'''
<작업 순서>
1. index 페이지 작성 -> 동 입력
2. flask server 에서 동(파라미터) 받기
3. db 연동 -> 주소 조회
4. 조회 결과 -> result 페이지 출력
'''

from flask import Flask, render_template, request #request : parameter를 받을 수 있는 멤버  # app 생성, html 호출
app = Flask(__name__) # object -> app object

#함수 장식자
@app.route('/') # 기본 url 요청 -> 함수 요청
def index():
    return render_template("/app03/index.html") # '/search'

@app.route('/search', methods=['GET', 'POST'] )
def search() :
    if request.method == 'POST' :
        dong = request.form['dong'] # 동
        #print('dong=', dong) # dong= 제기동

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

            # 4. 레코드 조회
            sql = "select * from zipcode_tab"
            cursor.execute(sql)
            data = cursor.fetchall()

            if data:  # True : 검색
                '''
                for row in data:
                    print("[%s]    %s   %s   %s  %s" % row)

                print('전체 레코드 수 : ', len(data))
                '''
                ### 1. 동 검색
                #dong = input("검색할 동 입력 :")
                sql = f"select * from zipcode_tab where dong like '%{dong}%'"
                cursor.execute(sql)
                data2 = cursor.fetchall()
                if data2:
                    for row in data2:
                        print("[%s]    %s   %s   %s  %s" % row)

                    print('검색된 레코드 수:', len(data2))
                    size = len(data2)
                else:
                    print('검색된 레코드 없음')
                    size = 0

        except Exception as e:
            print('db 연동 오류 :', e)
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

        return render_template("/app03/result.html",
                               dong = dong, data = data2, size=size)

#프로그램 시작점
if __name__ == "__main__" :
    app.run(host='192.168.12.31', port=80) #자신 컴퓨터의 url을 넣고 적용하고 싶은 port번호를 넣어서 활용
    #192.168.12.31는 선생님 컴퓨터 ID address (127.0.0.1 port=5000 이것이 기본값이다.)
    # host, port 변경