'''
우편번호 검색
[우편번호]tab[도/시]tab[구]tab[동]tab[세부주소]

135-806	서울	강남구	개포1동 경남아파트		1
[우편번호]tab[도/시]tab[구]tab[동]
135-807	서울	강남구	개포1동 우성3차아파트	(1∼6동)	2
[우편번호]tab[도/시]tab[구]tab[동]tab[세부주소]

'''

import os
print(os.getcwd()) #C:\ITWILL\3_Python-I\workspace\chap07_FileIO\exams
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap07_FileIO\lecture')

import os
print(os.getcwd())
# C:\ITWILL\3_Python-I\workspace\chap07_FileIO\lecture

try :
    dong = input("동을 입력하세요 :") # 개포
    file = open("../data/zipcode.txt", mode='r', encoding='utf-8')
    line = file.readline() # 첫줄 주소 읽음

    while line : # null == False
        addr = line.split(sep='\t')

        if addr[3].startswith(dong) :
            print('['+addr[0]+']',addr[1], addr[2], addr[3], addr[4]) # [135-806]

        line = file.readline() # 두번째 ~ n 줄 주소 읽음

except Exception as e :
    print("예외발생 :", e)
finally:
    file.close()
    print("~~ 종료 ~~")








