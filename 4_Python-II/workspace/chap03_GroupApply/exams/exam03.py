import pandas as pd

election = pd.read_csv('../data/election_2012.csv', encoding='ms949')
print(election.info())
'''
cand_id : 대선 후보자 id
cand_nm : 대선 후보자 이름
contbr_nm : 후원자 이름 
contbr_occupation : 후원자 직업군 
contb_receipt_amt : 후원금 
'''

# DF 객체 생성 
name = ['cand_nm', 'contbr_occupation', 'contb_receipt_amt'] 
# subset 생성 
election_df = pd.DataFrame(election, columns= name)
print(election_df.info())
print(election_df.head())
print(election_df.tail())


# 중복되지 않은 대선 후보자 추출 
unique_name = election_df['cand_nm'].unique()
print(len(unique_name)) 
print(unique_name) 

# 중복되지 않은 후원자 직업군 추출 
unique_occ =  election_df['contbr_occupation'].unique()
print(len(unique_occ)) 
print(unique_occ)

#############################################
#  Obama, Barack vs Romney, Mitt 후보자 분석 
#############################################

# 1. 두 후보자 관측치만 추출 : isin()
two_cand_nm=election_df[election_df.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]
print(two_cand_nm.head())
print(two_cand_nm.tail())
print(len(two_cand_nm)) # 700975
'''
문1) two_cand_nm 변수를 대상으로 피벗테이블 생성하기 
    <조건1> 교차셀 칼럼 : 후원금, 열 칼럼 : 대선 후보자,
             행 칼럼 : 후원자 직업군, 적용함수 : sum
    <조건2> 피벗테이블 앞부분 5줄 확인   
문2) 피벗테이블 대상 필터링 : 2백만달러 이상 후원금 대상     
'''

two_cand_nm.info()
'''
Int64Index: 700975 entries, 411 to 701385
Data columns (total 3 columns):
'''

ptable = pd.pivot_table(two_cand_nm,
                        values = 'contb_receipt_amt',
                        index = 'contbr_occupation',
                        columns = 'cand_nm',
                        aggfunc = 'sum')
ptable.head()
ptable.shape # (33605, 2) -> (직업군, 후보자)

# 문2) 행 단위 합계 -> 행 합계 2백만달러 이상 필터링
over_2m = ptable[ptable.sum(axis = 1) >= 2000000]
over_2m.shape #(13, 2)

over_2m
'''
cand_nm                                 Obama, Barack  Romney, Mitt
contbr_occupation                                                  
ATTORNEY                                  11126932.97    5302578.82
CEO                                        2069784.79     353310.92
CONSULTANT                                 2459812.71    1404576.94
EXECUTIVE                                  1355161.05    2230653.79
HOMEMAKER                                  4243394.30    8037250.86
INFORMATION REQUESTED                      4849801.96           NaN
INFORMATION REQUESTED PER BEST EFFORTS            NaN   11173374.84
INVESTOR                                    884133.00    1494725.12
LAWYER                                     3159391.87       7705.20
PHYSICIAN                                  3732387.44    1332996.34
PRESIDENT                                  1878009.95    2403439.77
PROFESSOR                                  2163571.08     160362.12
RETIRED                                   25270507.23   11266949.23
[해설]
오바마 후보자 : ATTORNEY, RETIRED 직업군 우세
롬니 후보자 : HOMEMAKER, PRESIDENT 직업군 우세
'''

# 문) 두 후보자 모두 200백만달러 이상의 후원금을 지불한 직업군 필터링
ptable[ptable['Obama, Barack'] >= 2000000 and
       ptable['Romney, Mitt'] >= 2000000]
# 이런 식은 오류가 난다.

import numpy as np

over_2mn_each = ptable[np.logical_and(ptable['Obama, Barack'] >= 2000000,
                                      ptable['Romney, Mitt'] >= 2000000)]
over_2mn_each.shape #(3, 2)
over_2mn_each
'''
cand_nm            Obama, Barack  Romney, Mitt
contbr_occupation                             
ATTORNEY             11126932.97    5302578.82
HOMEMAKER             4243394.30    8037250.86
RETIRED              25270507.23   11266949.23
'''

# 두 후보자 중 한 명이 200백만달러 이상 직업군
over_2mn_each2 = ptable[np.logical_or(ptable['Obama, Barack'] >= 2000000,
                                      ptable['Romney, Mitt'] >= 2000000)]
over_2mn_each2.shape #(12, 2)
over_2mn_each2





