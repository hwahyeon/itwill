# -*- coding: utf-8 -*-
"""
model.py
"""




##
# =============================================================================
# 1차 모델 불러오기
# =============================================================================
from tensorflow.keras.models import load_model
import os
import shutil # 폴더 복사
import pandas as pd
import numpy as np
import pickle
# 1) model load
paint_model = load_model("/content/drive/My Drive/dataset/paint_model.h5")
print("model has been loaded")


##
# =============================================================================
# RGB 이미지 1장 불러오기
# =============================================================================
from PIL import Image # read/save
import numpy as np
img_h = img_w = 224

pic = "/content/drive/My Drive/dataset/test_input/Raphael_90.jpg"
img = Image.open(pic) # image read
img = img.convert("RGB")
img = img.resize( (img_h, img_w) )
img_data = np.array(img)
test_img = []
test_img.append(img_data)
test_img = np.array(test_img)
print(test_img.shape)

##

# =============================================================================
# single image 예측 (1차) -> 장르 도출
# =============================================================================

import numpy as np
from sys import argv
from PIL import Image # read/save
from glob import glob # *.jpg
from tensorflow.keras.models import load_model
import os # dir 

preds = paint_model.predict(test_img)

top_1 = np.argpartition(preds[0], -1)[-1:][0]
genre_name = genre[top_1]
style = genres[top_1]
print(genre_name)
print(style)
top_3 = np.argpartition(preds[0], -3)[-3:]
top_3 = reversed(top_3[np.argsort(preds[0][top_3])])
print('Top 3 Predictions:')
print('------------------')
for i in top_3:
    print('{0}: {1:.2f}%'.format(genres[i], 100 * preds[0][i]))
    
    
##
# =============================================================================
# single image 예측 (2차) -> 작가 도출
# =============================================================================
from tensorflow.keras.models import load_model
import numpy as np
# 1) model load
artist_model = load_model(f"/content/drive/My Drive/dataset/second_models/{style}.h5")
print(f"{genre_name} model has been loaded")

pred_artist = artist_model.predict(test_img)
# =============================================================================
# 작가 이름 도출
# =============================================================================
artist_idx = data[data['genre2'] == f'{genre_name}'].index
name = list(data['name'][artist_idx])
names_sort = sorted(name)

top_1 = np.argpartition(pred_artist[0], -1)[-1:][0]
print(top_1)
artist = names_sort[top_1]
print(artist)
top_3 = np.argpartition(pred_artist[0], -3)[-3:]
top_3 = reversed(top_3[np.argsort(pred_artist[0][top_3])])
print('Top 3 Predictions:')
print('------------------')
for i in top_3:
    print('{0}: {1:.2f}%'.format(names_sort[i], 100 * pred_artist[0][i]))
 






