from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import time
from PIL import Image  # read/save
from sys import argv
from glob import glob  # *.jpg
import os  # dir
import json
from tensorflow.keras.models import model_from_json

print("모델 불러오기")
# =============================================================================
# 빠른 실행을 위한 전체 모델 라벨값  불러오기
# =============================================================================
file = open("C:/Users/user/Desktop/artists_dataset.pickle", mode='rb')  # rb -> read binary
data = pickle.load(file)

# 장르 불러오기
genres = np.unique(data['genre2'])

start = time.time()
'''# 1차 모델 1) model load
paint_model = load_model("C:/Users/user/Desktop/paint_model.h5")
print("model has been loaded")'''

json_model = json.load(open("C:/Users/user/Desktop/paint_model/paint_model.json", 'r'))
loaded_model = model_from_json(json_model)
loaded_model.load_weights("C:/Users/user/Desktop/paint_model/paint_model_weight")
paint_model = loaded_model

second_models = []

# 2차 모델
for i in range(len(genres)):
    json_model = json.load(open(f"C:/Users/user/Desktop/weight_save/{genres[i]}/{genres[i]}.json", 'r'))
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights(f"C:/Users/user/Desktop/weight_save/{genres[i]}/{genres[i]}_weight")

    second_models.append(loaded_model)

print("모델 불러오기완료")



def process_img(pic):
    img_h = img_w = 224
    img = Image.open(pic)  # image read
    img = img.convert("RGB")
    img = img.resize((img_h, img_w))
    img_data = np.array(img)
    test_img = []
    test_img.append(img_data)
    test_img = np.array(test_img)
    test_img = test_img.astype('float32')
    print(test_img.shape)
    return test_img


# predict 통합 함수
def predict(model, test_img):
    idx_top3 = []
    preds = model.predict(test_img)
    try:
      idx = np.argpartition(preds[0], -3)[-3:] # top3
    except:
      idx = np.argpartition(preds[0], -1)[-1:] # top1
    idx = reversed(idx[np.argsort(preds[0][idx])])
    for i in idx:
      idx_top3.append(i)
    return idx_top3, preds*100


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

# 파일 업로드 처리
@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        f = request.files['file']
        global pic
        pic = 'C:/Users/user/Desktop/' + secure_filename(f.filename)
        f.save(pic)  # 저장 위치
        test_img = process_img(pic)
        idx_g, preds_g = predict(paint_model, test_img)  # top3 장르 [인덱스, 확률]
        top3_g = [genres[i] for i in idx_g]  # top3 장르
        top3_g_per = [preds_g[0][i] for i in idx_g]  # top3 장르 확률

        names_p = sorted(data['name'][data['genre2'] == genres[idx_g[0]]])  # 예상 장르에 따른 name list 가져오기
        # 2차 모델 결과값
        idx_p, preds_p = predict(second_models[idx_g[0]], test_img)  # top3 화가 [인덱스, 확률]
        top3_n = [names_p[i] for i in idx_p]  # top3 화가
        top3_n_per = [preds_p[0][i] for i in idx_p]  # top3 화가 확률

    return render_template("index.html", style= top3_g[0] ,percent_art=top3_g_per[0],
                           top1= top3_n[0], top2= top3_n[1], top3= top3_n[2],
                           prob_1= top3_n_per[0], prob_2= top3_n_per[1], prob_3= top3_n_per[2])




# ip
if __name__ == "__main__":
    app.debug = True
    app.run()


