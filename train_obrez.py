import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import json
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model
from tredprocess import show_img

import tqdm

import time

def round_point(lts_l):
    return list(map(lambda x: round(x, 1), lts_l))


def parser_files():
    onlyfiles = [f for f in listdir("pasport2.2")]
    labels = [f for f in onlyfiles if ("json" in f and not("_" in f))]
    images = [cv2.imread(f"pasport2.2/{label.replace('json', 'png')}") for label in labels]
    lab = []
    for i, label in enumerate(labels):
        data = json.load(open(f"pasport2.2/{label}"))
        lst = data["shapes"][0]["points"]
        l_l = []
        for i, l in enumerate(lst):
            k = round_point(l)
            l_l.extend(k)
        lab.append(l_l)
    #print(lab)
    labels_np = np.array(lab)
    images = np.array(images)

    images = images / 255.0

    #show_img(images[0], labels_np[0])
    X_train, X_test, y_train, y_test = train_test_split(images, labels_np, test_size=0.1, random_state=42)

    print(f"[PARSER INFO] Files ready for trin\nX_train = {len(X_train)}\ny_train = {len(y_train)}\nX_test = {len(X_test)}\ny_test = {len(y_test)}")

    return X_train, X_test, y_train, y_test 

def train(X_train, y_train, n1=1, n2=1, epochs=200, train_or_load="load" ,safe_model_flag=True):
   
    if train_or_load == "train":
        # Создание модели нейронной сети
        model = tf.keras.Sequential([
          Conv2D(32, (3,3), activation='relu', input_shape=(500, 500, 3)),
          MaxPooling2D((2,2)),
          Conv2D(64, (3,3), activation='relu'),
          MaxPooling2D((2,2)),
          Conv2D(128, (3,3), activation='relu'),
          MaxPooling2D((2,2)),
          Flatten(),
          Dense(256, activation='relu'),
          Dense(8) # выходной слой с 8 нейронами для координат
        ])

        # Компиляция модели
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])

    elif train_or_load == "load":
        model=load_model(f'Model3_{n1}.h5')

    else:
        assert "Нет такого значения модели"


    # обучаем модель на обучающей выборке
    model.fit(X_train, y_train, epochs=epochs)
    
    if safe_model_flag:
        print(f"[TRAIN INFO] Model safe in file: Model3_{n2}.h5")
        model.save(f"Model3_{n2}.h5")
    
    return model

def test_model(model, X_test, y_test):
    # оцениваем точность на тестовой выборке
    test_loss, test_mae = model.evaluate(X_test, y_test)
    
    print('[TEST INFO] Test loss:', test_loss)
    print('[TEST INFO] Test mae:', test_mae)
    
    return test_loss, test_mae

def main(X_train, X_test, y_train, y_test, epochs, n1, n2):
    model = None
    model = train(X_train, y_train, n1, n2, epochs)
    test_loss, test_mae = test_model(model, X_test, y_test)
    return test_loss < 1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = parser_files()
    epochs = 50
    n1, n2 = 33, 34
    while True:
        if main(X_train, X_test, y_train, y_test, epochs, n1, n2):
            break
        epochs += 20
        n1 += 1
        n2 += 1
