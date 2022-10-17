'''
step3 관련 문제

문) word counter
   - 여러줄의 문장을 단어로 분류하고, 단어 수 출력하기
'''

multiline="""안녕하세요. Python 세계로 오신걸
환영합니다.
파이션은 비단뱀 처럼 매력적인 언어입니다."""

lst1=[]

lst1 = multiline.split()
print("\n".join(lst1))


# <<출력 결과>> 
'''
안녕하세요.
Python
세계로
오신걸
환영합니다.
파이션은
비단뱀
처럼
매력적인
언어입니다.
단어수 : 10
'''


#teacher
words = []
for sent in multiline.split('\n') :
    for word in sent.split() :
        print(word)
        words.append(word) # 단어 저장

print('단어수 :', len(words))

