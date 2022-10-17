from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image # read/save
from sys import argv
from glob import glob # *.jpg
import os # dir
import json
from tensorflow.keras.models import model_from_json



def modeling_fir(path): # 빠른 실행을 위한 전체 모델 라벨값  불러오기
    # pickle load
    file = open("C:/Users/user/Desktop/artists_dataset.pickle", mode='rb')  # rb -> read binary
    data = pickle.load(file)

    # 장르 불러오기
    genre = np.unique(data['genre'])
    genres = np.unique(data['genre2'])

    start = time.time()
    # 1차 모델 1) model load
    paint_model = load_model("C:/Users/user/Desktop/paint_model.h5")
    #print("model has been loaded")

    # RGB 이미지 1장 불러오기
    img_h = img_w = 224
    pic = path
    img = Image.open(pic) # image read
    img = img.convert("RGB")
    img = img.resize( (img_h, img_w) )
    img_data = np.array(img)
    test_img = []
    test_img.append(img_data)
    test_img = np.array(test_img)
    #print(test_img.shape)

    # single image 예측 (1차) -> 장르 도출
    preds = paint_model.predict(test_img)

    top_1 = np.argpartition(preds[0], -1)[-1:][0]
    genre_name = genre[top_1]
    style = genres[top_1]
    print(genre_name)
    print(style)
    top_3 = np.argpartition(preds[0], -3)[-3:]
    top_3 = reversed(top_3[np.argsort(preds[0][top_3])])
    print('Top 3 Predictions:')


    art_list = []
    for i in top_3:
        a = 100 * preds[0][i]
        art_list.append(a)

    return style, art_list[0]



a, b = modeling_fir("C:/Users/user/Desktop/new/raw_images/Cubism/Pablo_Picasso/Pablo_Picasso_129.jpg")
print(b)
print(a)

preview = Image.open("C:/Users/user/Desktop/new/raw_images/Cubism/Pablo_Picasso/Pablo_Picasso_129.jpg")
preview


