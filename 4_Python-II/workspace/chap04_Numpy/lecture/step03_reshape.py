# -*- coding: utf-8 -*-
"""
1. image shape : 3차원(세로, 가로, 칼럼)
2. reshape : size 변경 안됨
    ex) [2,5] -> [5,2] 이런 식의 reshape는 가능하다.
        [3,4] -> [4,2] 이런 식의 reshape는 size가 달라 불가하다.
"""

import numpy as np
from matplotlib.image import imread # 이미지 읽어오는 함수
import matplotlib.pylab as plt


# 1. image shape
file_path = 'C:/ITWILL/4_Python-II/workspace/chap04_Numpy/images/test1.jpg'
image = imread(file_path)
print(type(image)) #<class 'numpy.ndarray'>

image.shape #(360, 540, 3) -> (세로, 가로, 컬러(컬러가 3이란 것은 3원색이란 뜻))
print(image) # 0~255 값 사이를 가진다. -> 0~1로 변경

plt.imshow(image) # 이미지 보기

# RGB 색상 분류
r = image[:,:,0] # R
g = image[:,:,1] # G
b = image[:,:,2] # B

r.shape # (360, 540)
g.shape
b.shape

# 2. image data reshape
from sklearn.datasets import load_digits #아나콘다에서 제공하는 데이터셋

digit = load_digits() # dataset loading
digit.DESCR # 설명보기

x = digit.data # x변수(입력변수) : image
y = digit.target # y변수(정답 = 정수)

x.shape #(1797, 64) 64 = 8x8
y.shape #(1797,)

img_0 = x[0].reshape(8,8) # 행 index
img_0
plt.imshow(img_0) #0
y[0] # 0

x_3d = x.reshape(-1, 8, 8) # 모든 것을 3차원으로 리셰이프하겠다.
x_3d.shape #(1797, 8, 8) #컬러이미지라면 맨 뒤에 (1797, 8, 8, 3)처럼 들어가야한다.
#즉 이것은 흑백이미지라고 볼 수 있다.
# -> (전체 이미지, 세로, 가로, [컬러])
# [컬러] 부분이 1이거나 생략되었을 때는 흑백이다.

x_4d = x_3d[:,:,:,np.newaxis] # 4번째 축 추가
# newaxis 특정한 위치에 새로운 축을 추가하고자 할때 사용
# 이미지분석을 할 때 2차원을 3차원으로 바꾸거나 할 때 사용한다.
x_4d.shape #(1797, 8, 8, 1)

# 3. reshape
'''
전치행렬 : T
swapaxis = 전치행렬
transpose() : 3차원 이상 모양 변경
'''

# 1) 전치행렬
data = np.arange(10).reshape(2, 5)
data
'''
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
'''

data.T
'''
array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
'''

# 2) transpose()
'''
1차원 : 효과 없음
2차원 : 전치행렬과 동일함
3차원 : (0,1,2) -> (2,1,0)
'''

arr3d = np.arange(1,25)
arr3d # 1차원의 데이터를 만듦
'''
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24])
'''
arr3d = np.arange(1,25).reshape(4, 2, 3)
arr3d.shape #(4, 2, 3)
'''
array([[[ 1,  2,  3],
        [ 4,  5,  6]],

       [[ 7,  8,  9],
        [10, 11, 12]],

       [[13, 14, 15],
        [16, 17, 18]],

       [[19, 20, 21],
        [22, 23, 24]]])
'''

# (0,1,2) -> (2,1,0)
arr3d_tran = arr3d.transpose(2,1,0) # 직접 축의 번호를 통해서 스와핑이 가능하다.
arr3d_tran.shape #(3, 2, 4)

# (0,1,2) -> (1,2,0)
arr3d_tran = arr3d.transpose(1,2,0)
arr3d_tran.shape #(2, 3, 4)










