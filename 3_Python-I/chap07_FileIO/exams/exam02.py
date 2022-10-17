'''
문제2) 서울 지역을 대상으로 '동' 이름만 추출하여 다음과 같이 출력하시오.
  단계1 : 'ooo동' 문자열 추출 : 예) '개포1동 경남아파트' -> '개포1동'
  단계2 : 중복되지 않은 전체 '동' 개수 출력 : list -> set -> list
  
  <출력 예시>  
서울시 전체 동 개수 =  797
'''

import os
print(os.getcwd()) #C:\ITWILL\3_Python-I\workspace\chap07_FileIO\exams
os.chdir('C:\ITWILL\\3_Python-I\workspace\chap07_FileIO\exams')

try:
    file = open("../data/zipcode.txt", mode='r', encoding='utf-8')
    lines = file.readline()  # 첫줄 읽기
    print(lines)
    dongs = []  # 서울시 동 저장 list

    # 내용채우기
    cnt = 0
    while lines:
        addr = lines.split(sep='\t')
        if addr[1] == '서울':  # 주소 서울인 경우
            # print(lines)
            dong = addr[3].split()  # 공백으로 split
            dongs.append(dong[0])  # '[개포1동]' '경남아파트'
            cnt += 1

        lines = file.readline()  # 두번째 줄 ~ n

    dongs = list(set(dongs))  # list -> set -> list
    print('주소 개수 :', cnt)  # 주소 개수 : 8080
    print('서울시 전체 동 개수 =', len(dongs))
    # 서울시 전체 동 개수 = 797
    print('-' * 50)
    print(dongs)
    print('-' * 50)

except Exception as e:
    print('예외발생 :', e)







