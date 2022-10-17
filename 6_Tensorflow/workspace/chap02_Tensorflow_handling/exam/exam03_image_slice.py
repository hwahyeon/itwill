'''
문) image.jpg 이미지 파일을 대상으로 파랑색 우산 부분만 slice 하시오.
'''

import tensorflow as tf
import matplotlib.image as mp_image
import matplotlib.pyplot as plt

filename = "C:/ITWILL/6_Tensorflow/data/image.jpg"
input_image = mp_image.imread(filename)

#dimension
print('input dim = {}'.format(input_image.ndim))
#shape
print('input shape = {}'.format(input_image.shape))
# input shape = (512, 768, 3) - (높이 pixel, 폭 pixel, 3color)

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

img_slice = tf.slice(input_image, [105,30,0],[300,520,-1])# [높이,폭,색상]
# 전체 높이 512에서 100을 기준으로 높이 0~16, 폭,색상 전체 

print(img_slice.shape) # (300, 520, 3)
     
plt.imshow(img_slice)
plt.show()

