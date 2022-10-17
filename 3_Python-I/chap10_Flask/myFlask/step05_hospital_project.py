# db 연결 객체 생성 함수
def db_conn() :
    import pymysql
    config = {
        'host': '127.0.0.1',
        'user': 'scott',
        'password': 'tiger',
        'database': 'work',
        'port': 3306,
        'charset': 'utf8',
        'use_unicode': True}

    # db 연결 객체 생성
    conn = pymysql.connect(**config)
    # SQL 실행 객체 생성
    cursor = conn.cursor()
    return conn, cursor


from flask import Flask, render_template, request # app 생성, html 호출

app = Flask(__name__) # object -> app object

# 함수 장식자
@app.route('/') # 기본 url 요청 -> 함수 호출
def index() :
    return  render_template("/app05/main.html")

@app.route('/docForm')
def docForm() :
    return render_template("/app05/docForm.html")

@app.route("/docPro", methods =['GET', 'POST'])
def docPro():
    if request.method == 'POST' :
        doc_id = int(request.form['id'])
        major = request.form['major']

        conn, cursor = db_conn()
        sql = f"""select * from doctors where doc_id = {doc_id}
                  and major_treat = '{major}' """
        cursor.execute(sql)
        row = cursor.fetchone()

        if row : # login 성공 -> 진로정보
            #print('login 성공') : 601, 내과
            sql = f"""select d.doc_id, t.pat_id, 
            TREAT_CONTENTS, TREAD_DATE  
            from doctors d inner join treatments t
            on d.doc_id = t.doc_id and d.doc_id = {doc_id}"""

            cursor.execute(sql)
            data = cursor.fetchall()

            if data :
                for row in data :
                    print(row)
                size = len(data)
            else :
                size = 0

            return render_template("/app05/docPro.html", dataset=data, size=size)

        else : # login 실패
            return render_template("/app05/error.html", info="id 또는 진뢰과목 확인")

@app.route('/nurseForm')
def nurseForm() :
    return render_template("/app05/nurseForm.html")

@app.route("/nursePro", methods = ['GET', 'POST'])
def nursePro() :
    if request.method == 'POST':
        nur_id = int(request.form['id'])

        conn, cursor = db_conn()
        sql = f"""select * from NURSES where nur_id = {nur_id}"""
        cursor.execute(sql)
        row = cursor.fetchone()

        if row:  # login 성공 -> 진로정보
            # print('login 성공') : 40089
            sql = f"""select n.nur_id, p.doc_id, 
            PAT_NAME, PAT_PHONE  
            from NURSES n inner join PATIENTS p
            on n.nur_id = p.nur_id and n.nur_id = {nur_id}"""

            cursor.execute(sql)
            data = cursor.fetchall()

            if data:
                for row in data:
                    print(row)
                size = len(data)
            else:
                size = 0

            return render_template("/app05/nursePro.html", dataset=data, size=size)

        else:  # login 실패
            return render_template("/app05/error.html", info="id 확인")


#프로그램 시작점
if __name__ == "__main__" :
    app.run() # application 실행