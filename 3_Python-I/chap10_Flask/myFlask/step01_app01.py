'''
1. templates file 작성
    - 사용자 요청과 서버의 응답을 작성하는 html file
2. static file 작성
    - 정적 파일 : image file, js(동적기능), css 등
'''

from flask import Flask, render_template #html 페이지 호출

#flask application
app = Flask(__name__) # 생성자 -> object(app)

# 함수 장식자 : 사용자 요청 url -> 함수 호출
@app.route('/') # 기본 url : http://127.0.0.1:5000/
def index(): #함수 장식자에 의해 호출되는 함수
    return render_template('/app01/index.html') # 호출 html 페이지

@app.route('/info') # http://127.0.0.1:5000/info
def info():
    return render_template('/app01/info.html') # 호출 html 페이지

# 프로그램 시작점 (서버)
if __name__ == "__main__":
    app.run() # application 실행
