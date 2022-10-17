# -*- coding: utf-8 -*-
"""
indexing/slicing
  - 1차원 indexing : list 동일함
  - 2, 3차 indexing
  - boolean indexing
"""

import numpy as np


# 1. indexing
'''
1차원 : obj[index]
2차원 : obj[행index, 열index]
1차원 : obj[면index, 행index, 열index]
'''

# 1) list 객체
ldata = [0,1,2,3,4,5]
ldata
ldata[:] #전체 원소
# ldata[:] = 10
ldata[2:] # [n:~]
ldata[:3] # [~:n]
ldata[-1]

# 2) numpy 객체
arr1d = np.array(ldata)
arr1d.shape #(6,)
arr1d[2:]
arr1d[:3]
arr1d[-1] # 5

# 2. slicing
arr = np.array(range(1,11))
arr #array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr_s1 = arr[4:8]
arr_s1 # 사본

# 블럭 수정
arr_s1[:] = 50
arr_s1 #array([50, 50, 50, 50])

arr # 원본 수정(사본을 수정해도 원본이 수정되는 것이 특징)

# 3. 2, 3차 indexing
arr2d = np.array([[1,2,3],[2,3,4],[3,4,5]]) # 3x3의 구조
arr2d.shape #(3, 3)
arr2d
'''
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]])
'''
# 행 index : default
arr2d[1] #array([2, 3, 4])
#arr2d[1] = arr2d[1,:]
arr2d[1:] # 2~3행
arr2d[:,1:] # 2~3열 (행 자리를 생략할 수는 없다.)
arr2d[2,1] # 3행 2열
arr2d[:2,1:] #box 선택

# [면,행,열] : 면 index : default
arr3d = np.array([   [[1,2,3], [2,3,4], [3,4,5]],
                   [[2,3,4], [3,4,5], [6,7,8]]   ])
# 면 2, 행 3, 열 3
arr3d
'''
array([[[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]],

       [[2, 3, 4],
        [3, 4, 5],
        [6, 7, 8]]])
'''
arr3d.shape #(2, 3, 3)

arr3d[0] # 면 index = 1면
arr3d[1]

# 면 -> 행 index
arr3d[0, 2] #첫번째 면에서 3번째 행을 꺼낸다는 뜻 [3, 4, 5]

# 면 -> 행 -> 열 index
arr3d[1,2,2] # 8

# 2번째 면에서 우측 하단 상자영역(2x2) 선택하기
arr3d[1,1:,1:]

# 4. boolean indexing (조건을 만족하는 특정 DF 필터링)
dataset = np.random.randint(1, 10, size=100) # 1~10
len(dataset) #100

dataset

# 데이터에서 5이상인 경우를 선별
dataset2 = dataset[dataset >= 5]
len(dataset2) # 58
dataset2

# 5 ~ 8 자료 선택
dataset[dataset >= 5 and dataset <= 8]
# index에선 관계식 1개엔 문제가 없지만
# 2개 이상의 관계식을 연결하는 논리식은 오류가 난다.

np.logical_and
np.logical_or
np.logical_not

dataset2 = dataset[np.logical_and(dataset >= 5, dataset <= 8)]
dataset2 # 5 ~ 8







