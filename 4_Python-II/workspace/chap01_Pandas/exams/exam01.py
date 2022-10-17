''' 
step02 관련문제 
문1) score.csv 파일을 읽어와서 다음과 같이 처리하시오.
   조건1> tv 칼럼이 0인 관측치 2개 삭제 (for, if문 이용)
   조건2> score, academy 칼럼만 추출하여 DataFrame 생성
   조건3> score, academy 칼럼의 평균 계산 
   - <<출력 결과 >> 참고    
   
<<출력 결과 >>
   score  academy
1     75        1
2     77        1
3     83        2
4     65        0
5     80        3
6     83        3
7     70        1
9     79        2
score      76.500
academy     1.625   
'''

import pandas as pd

score = pd.read_csv("C:\\ITWILL\\4_Python-II\\data\\score.csv")
print(score.head())


# 조건1
score_df0 = score[score.tv != 0]
score_df0

# 조건2
score_df0 = score_df0[['score', 'academy']]
score_df0

# 조건3
score_df0.mean(axis=0) #한번에 모든 열 평균 구해줌.

score_df0.score.mean()
score_df0.academy.mean()



# for와 if문 이용
import pandas as pd

score = pd.read_csv("C:/ITWILL/4_Python-II/data/score.csv")
print(score.info())
score

score_c = score
tv = score.tv
for i, c in enumerate(tv):
    if c == 0:
        score_c = score_c.drop(i)

score_c

# 조건1> tv 칼럼이 0인 관측치 2개 삭제
score_df = score[score.tv != 0]
score_df

# 조건2> score, academy 칼럼만 추출하여 DataFrame 생성
score_df = score_df[['score', 'academy']]
score_df

# 조건3> score, academy 칼럼의 평균 계산
score_df.mean(axis=0)  # 열단위 평균

# 각 칼럼 단위 평균
score_df.score.mean()
score_df.academy.mean()



