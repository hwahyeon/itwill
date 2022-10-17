# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:16:30 2020

@author: user
"""

import olefile

f = olefile.OleFileIO('C:\\ITWILL\\4_Python-II\\workspace/test.hwp') #olefile로 한글파일 열기
encoded_text = f.openstream('PrvText').read() #PrvText 스트림 안의 내용 꺼내기 (유니코드 인코딩 되어 있음)
decoded_text = encoded_text.decode('UTF-16') #유니코드이므로 UTF-16으로 디코딩
print(decoded_text)












