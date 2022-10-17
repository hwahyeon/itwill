# -*- coding: utf-8 -*-
"""
xlsxwriter
"""


import xlsxwriter

# 1. 생성

workbook = xlsxwriter.Workbook('2_1.xlsx') # 워크북 생성
worksheet = workbook.add_worksheet('Fisrt_sheet') # 워크 시트 생성

# 셀에 값 입력
worksheet.write('A1', 'Hello')
worksheet.write('B1', 'World')

# 좌표로 입력
worksheet.write(0, 2, "It's me") # A1 = (0,0)

worksheet.write(1,0,2)
worksheet.write(1,1,4)
worksheet.write(2,0, '=A2+B2')

workbook.close()

# 2. 평균 점수 계산

workbook = xlsxwriter.Workbook('2_2.xlsx')
worksheet = workbook.add_worksheet('First_sheet')

student_math_score =(
    ['hojun', 95],
    ['eunjung', 75],
    ['subin', 98],
    ['eunbin', 67]
    )
row = 0
col = 0

for student, score in student_math_score:
    worksheet.write(row, col, student)
    worksheet.write(row, col+1, score)
    row += 1
    
bold = workbook.add_format({'bold':True})

worksheet.write(row, 0, 'Average', bold)
worksheet.write(row, 1, '=AVERAGE(B1:B4)', bold)

workbook.close()

cell_format = workbook.add_format({'bold':True, 'italic':True})
worksheet.write(0, 0, 'Hello', cell_format)

# 3. write_formula(row, col, fromula)

worksheet.write_formula(0, 0, '=A2+A4')
worksheet.write_formula(1, 0, '=SIN(PI()/4)')
worksheet.write_formula('A4', 'IF(A3>1, "Yes","No")')
worksheet.write_formula('A6', '=DATEVALUE("1-Jan-2013")')

# 1) datetime
date_time = datetime.datetime.strptime('2013-01-23', '%Y-%m-%d')

'''
date_format = add_format({'num_format':'yy/mm/dd'}) # 16.12.01
date_format = add_format({'num_format':'yyyy-mm-dd'}) # 2016-12-01
date_format = add_format({'num_format':'yy/mm/dd hh:mm'}) # 16.12.01 12:00
date_format = add_format({'num_format':'yyyy mmm dd'}) # 2016 Dec 01
date_format = add_format({'num_format':'yyyy mmm dd hh:mm AM/PM'}) # 2016 Dec 01 12:00 PM
'''

worksheet.write_datetime(0, 0, date_time, date_format)


# 3. Text 읽고 쓰기, Excel 읽고 쓰기

# 1) 쓰기
f = open("test.txt", 'w') # 쓰기 모드
data = 'text 입출력을 test하고 있습니다.'
f.write(data)
f.close()

# 2) 읽기
f = open("test.txt", 'r') # 읽기 모드
data = f.read()
print(data)
f.close()






















