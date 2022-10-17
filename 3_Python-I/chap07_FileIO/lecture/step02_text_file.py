'''
텍스트 파일 입출력
형식)
    open(file, mode='r', 'w', 'a')
'''

import os #파일 경로 확인
#파일 입출력에서 예외처리를 많이 씀.

try :
    # 1. 파일 읽기
    print('현재 경로 :', os.getcwd()) #현재 파일 경로 확인하기 R에서의 getwd와 비슷하다.
    os.chdir('C:\ITWILL\\3_Python-I\workspace')
    '''
    #file open : 절대경로
        file = open(os.getcwd() + '/chap07_FileIO/data/ftest.txt', mode = 'r')
    print(file.read()) #file 사용
    file.close() #file close
    '''
    #file open : 상대경로(. : 현재 디렉터리, .. : 상위 디렉터리)
    file = open('./chap07_FileIO/data/ftest.txt', mode = 'r') #.은 현재 위치(경로)를 말함.
    print(file.read())
    file.close()

    # 2. 파일 쓰기
    file2 = open('./chap07_FileIO/data/ftest2.txt', mode = 'w')
    file2.write("my first text~~") #덮어쓰기 형식으로 된다.
    file2.close()

    # 3. 파일 쓰기(덮어쓰기가 아닌 추가 개념)
    file3 = open('./chap07_FileIO/data/ftest2.txt', mode='a') #append
    file3.write("\nmy second text")  # 덮어쓰기 형식으로 된다.
    file3.close()

    '''
    file.read() : 전체 문서 한 번에 읽기
    file.readline() : 전체 문서에서 한 줄 읽기
    file.readlines() : 전체 문서를 줄 단위로 읽어옴.
    '''

    # 4. readline()
    file4 = open('./chap07_FileIO/data/ftest2.txt')
    '''
    row = file4.readline() #줄단위로 한줄씩만 읽어옴. 이 문장을 또 한번 실행하면 다음 문장을 읽는다.
    print('row :' ,row)
    file4.close()
    '''

    for i in range(2) : #readline으로 다 읽으려고 할 떄 씀. i 대신 _를 쓰면 받긴 받나 사용하지는 않겠다는 뜻이다.
        row = file4.readline()
        print('row :', str(i+1), row.strip())
    file4.close()

    # 5. readlines() : 전체 문장을 줄단위로 읽어온다.
    file5 = open('./chap07_FileIO/data/ftest2.txt')
    rows = file5.readlines()
    print('rows =', rows) # rows = ['my first text~~\n', 'my second text'] list객체로 반환된다.

    for row in rows: #'my first text~~\n'
        for sent in row.split('\n'): #'my first text~~' ''
            if sent:
                print(sent)

    # string.strip() : 문장 끝 불용어(공백, \n, \t 등등) 제거
    print('strip 함수')
    for row in rows:
        print(row.strip()) #위의 이중for문 코드와 결과는 같다. 즉 이것이 더 간결한 코드.

    str_text = "asdfsdafqwsadf234\n \t \r"
    print('str_text :', str_text.strip()) #문장 끝에 오는 불용어들만 잡아주는 것임.
    file5.close()

    # 6. with
    with open('./chap07_FileIO/data/ftest3.txt', mode = 'w', encoding='utf-8') as file6: #블록 내에서만 file6객체를 쓸 수 있다.
        file6.write("파이썬 파일 작성 연습")
        file6.write("\n파이썬 파일 작성 연습2")
        #close를 하지 않아도 된다.

    with open('./chap07_FileIO/data/ftest3.txt', mode = 'r', encoding='utf-8') as file7:
        print(file7.read())

except FileNotFoundError as e:
    print('예외 정보 : ', e)
finally:
    print('~~종료~~')











