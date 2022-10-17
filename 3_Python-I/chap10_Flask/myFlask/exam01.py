'''
문) emp 테이블을 대상으로 사원명을 조회하는 application 을 구현하시오.
  조건1> index 페이지에서 사원명을 입력받아서 post 방식 전송
  조건2> 해당 사원이 있으면 result 페이지에 사번, 이름, 직책, 부서번호 칼럼 출력
  조건3> 해당 사원이 없으면 result 페이지에 '해당 사원 없음' 이라고 출력
'''
import pymysql

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/') # 시작 페이지
def index() :
    return render_template('/exam01/index.html') # templates/temp 폴더에서 작성

@app.route('/search', methods=['GET', 'POST']) #검색된 사원 이름을 search로 보내줌.
def search() :
    if request.method=='POST' : # login form 제공
        name = request.form['name']
        print(name)

    config = {
        'host': '127.0.0.1',
        'user': 'scott',
        'password': 'tiger',
        'database': 'work',
        'port': 3306,
        'charset': 'utf8',
        'use_unicode': True}

    try:
        conn = pymysql.connect(**config)
        cursor = conn.cursor()

        # 2. 레코드 조회
        sql = "select * from emp"

        cursor.execute(sql)
        dataset = cursor.fetchall()

        if dataset:  # 레코드 있는 경우(검색) (즉 검색되면 조건이 참이게 됨.)
            #dong = input("검색 동 입력 : ")
            sql = f"select * from emp where ename like '%{name}%'"
            cursor.execute(sql)
            dataset2 = cursor.fetchall() #예를 들어 홍으로 검색하면 홍으로 시작하는 이름이 모두 나와야하니까.

            if dataset2:
                print('검색된 레코드 : ', len(dataset2))
                size = len(dataset2)
            else:
                print('검색된 레코드 없음')
                size = 0

    except Exception as e:
        print('db error :', e)
    finally:
        cursor.close(); conn.close()

    return render_template('/exam01/result.html', name=name, dataset=dataset2, size=size) #해당 템플릿으로 보내 출력.
if __name__ == '__main__':
    app.run()


