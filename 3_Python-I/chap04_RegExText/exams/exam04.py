'''
문4) 다음 texts 객체를 대상으로 단계별로 텍스트를 전처리하시오. 

 <텍스트 전처리 후 결과> 
['우리나라 대한민국 우리나라 만세', '비아그라 정력 최고', '나는 대한민국 사람', '보험료 원에 평생 보장 마감 임박', '나는 홍길동']
'''

# 전처리 전 텍스트
texts = [' 우리나라    대한민국, 우리나라%$ 만세', '비아그&라 500GRAM 정력 최고!', '나는 대한민국 사람', '보험료 15000원에 평생 보장 마감 임박', '나는 홍길동']

from re import sub

print('전처리 전 : ', texts)

# 1. 소문자 변경 
text_re1 = [text.lower() for text in texts]
print('text_re1', text_re1)

# 2. 숫자 제거 
text_re2 = [sub('[0-9]', '',text) for text in text_re1]
print('text_re2', text_re2)

# 3. 문장부호 제거 
punt = '[.,;:!?]'
text_re3 = [sub(punt, '', text) for text in text_re2]
print('text_re3', text_re3)
# texts_re3 = [ sub('[.,;:!?]', '', text) for text in texts_re2]
# 이렇게 해도 됨.

# 4. 영문자 제거
text_re4 = [sub('[a-z]','',text) for text in text_re3]
print('text_re4', text_re4)

# 5. 특수문자 제거 
spe = '[@#$%^&*()]' #이것도 sub뒤에 바로 넣어버려도 된다.
text_re5 = [sub(spe,'',text) for text in text_re4]
print('text_re5', text_re5)

# 6. 공백 제거(2칸 이상 공백 -> 1칸 공백)
text_re6 = [' '.join(text.split()) for text in text_re5]
print('texts_re5 : ', text_re6)

