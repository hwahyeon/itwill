# -*- coding: utf-8 -*-
"""
DataFrame 모양 변경
"""

import pandas as pd

buy = pd.read_csv('C:\\ITWILL\\4_Python-II\\data\\buy_data.csv')
buy.info()
type(buy) #pandas.core.frame.DataFrame

buy.shape #(22, 3) : 2차원
buy

# 1. row -> column(wide -> long)
buy_long = buy.stack() # 2차원을 1차원으로 축소할 때 stack을 사용.
buy_long.shape #(66,) : 1차원
buy_long

# 2. column -> row(long -> wide)
buy_wide = buy_long.unstack()
buy_wide.shape #(22, 3)
buy_wide

# 3. 전치행렬 : t() -> .T
wide_t = buy_wide.T
wide_t.shape #(3, 22)

# 4. 중복 행 제거
buy.duplicated() #중복 확인(있는지 없는지만 확인하는 것.)
buy_df = buy.drop_duplicates() #똑같은 행이 있으면 다 제거해주는 역할
buy_df.shape # (20, 3)
buy.drop_duplicates()










