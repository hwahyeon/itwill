# -*- coding: utf-8 -*-
"""
reshape vs resize
  - reshape : 모양 변경
  - resize : 크기 변경
    ex) images -> 120x150 규격화 -> model

image 규격화 : 실습
"""

from glob import glob # file 검색 패턴 사용
#(문자열 경로, *.jpg 이런 패턴을 활용하여 특정 파일 선정 가능)
from PIL import Image # image file read
#PIL(python image library)은 아나콘다에서 제공하는 패키지라 설치 안해도 된다.
import numpy as np
import matplotlib.pyplot as plt # 이미지 시각화

# 1개 image file open

#Spyder경로를 C:\ITWILL\4_Python-II\workspace로 해놓은 상태에서
path = "./chap04_Numpy"
file = path + '/images/test1.jpg'

img = Image.open(file) #image file read
type(img) #PIL.JpegImagePlugin.JpegImageFile
np.shape(img) #(360, 540, 3) -> (120, 150, 3)
plt.imshow(img)

img_re = img.resize(  (150, 120)  ) # w,h 세로사이즈랑 가로 사이즈 순서를 변경해줘야한다.
np.shape(img_re) #(120, 150, 3)
plt.imshow(img_re) # 해상도가 떨어진 것을 볼 수 있다.

#PIL -> numpy
type(img_re) #PIL.Image.Image
img_arr = np.asarray(img)
img_arr.shape #(360, 540, 3)
type(img_arr) #numpy.ndarray

# 여러 장의 image resize 함수
def imageResize() :
    img_h = 120 # 세로 픽셀
    img_w = 150 # 가로 픽셀
    
    image_resize = [] # 규격화된 image 저장

    # glob: file 패턴
    for file in glob(path + '/images/' + '*.jpg'):
        # test1.jpg -> test2.jpg, ...
        img = Image.open(file) # image file read
        print(np.shape(img)) # image shape
        
        # PIL -> resize
        img = img.resize( (img_w, img_h) ) # w, h
        # PIL -> numpy
        img_data = np.asarray(img)
        
        # resize image save
        image_resize.append(img_data)
        print(file, ':', img_data.shape) # image shape
    
    return np.array(image_resize) #list -> numpy


image_resize = imageResize()
'''
(360, 540, 3) : test1.jpg
./chap04_Numpy/images\test1.jpg : (120, 150, 3)
(332, 250, 3) : test2.jpg
./chap04_Numpy/images\test2.jpg : (120, 150, 3)
'''

image_resize.shape
#(2, 120, 150, 3) (이미지 수, 세로 픽셀, 가로 픽셀, 컬러)

image_resize[0].shape #첫번째 이미지를 보겠다는 뜻 (120, 150, 3)
image_resize[1].shape #두번째 이미지를 보겠다는 뜻 (120, 150, 3)
#안의 픽셀값은 다르지만 사이즈는 같은 것을 알 수 있다.

# image 보기
plt.imshow(image_resize[0])
plt.imshow(image_resize[1])













