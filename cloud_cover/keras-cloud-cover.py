import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

import cv2
import tifffile as tiff
from tqdm import tqdm

x_train = []
y_train = []

df_train = pd.read_csv('/input/train.csv')

AVAILABLE_LABELS = [
    'clear', 
    'cloudy', 
    'haze', 
    'partly_cloudy']

tags = pd.DataFrame()

for label in AVAILABLE_LABELS:
    tags[label] = df_train.tags.apply(lambda x: np.where(label in x, 1, 0))
    
df_train = pd.concat([df_train, tags], axis=1)

df_train = pd.concat([df_train[df_train.clear == 1].sample(n=7251),
    df_train[df_train.cloudy == 1].sample(n=7251),
    df_train[df_train.haze == 1],
    df_train[df_train.partly_cloudy == 1].sample(n=7251)], axis=0, ignore_index=True)

for f, tags, clear, cloudy, haze, partly_cloudy in tqdm(df_train.values, miniters=1000):
    img = tiff.imread('/input/train-tif-v2/{}.tif'.format(f))
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append([clear, cloudy, haze, partly_cloudy])
    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

print(x_train.shape)
print(y_train.shape)

split = 35000

x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 4)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(x_train, y_train,
          batch_size=128,
          epochs=200,
          verbose=1,
          callbacks=[ModelCheckpoint('/output/keras-simple.model', monitor='val_loss', verbose=0, mode='auto', period=1)],
          validation_data=(x_valid, y_valid))
          
p_valid = model.predict(x_valid, batch_size=128)
print(y_valid)
print(p_valid)
                
# from sklearn.metrics import fbeta_score
# def f2_score(y_true, y_pred):
#     # fbeta_score throws a confusing error if inputs are not numpy arrays
#     y_true, y_pred, = np.array(y_true), np.array(y_pred)
#     # We need to use average='samples' here, any other average method will generate bogus results
#     return fbeta_score(y_true, y_pred, beta=2, average='samples')
                     
# print(f2_score(y_valid, p_valid))

