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
x_test = []
y_train = []

df_train = pd.read_csv('../input/train.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = tiff.imread('../input/train-tif-v2/{}.tif'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (32, 32)))
    y_train.append(targets)
    
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
model.add(Dense(17, activation='sigmoid'))

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          callbacks=[ModelCheckpoint('/output/keras-simple.model', monitor='val_loss', verbose=0, mode='auto', period=1)],
          validation_data=(x_valid, y_valid))
          
p_valid = model.predict(x_valid, batch_size=128)
print(y_valid)
print(p_valid)
                
from sklearn.metrics import fbeta_score
def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')
                     
print(f2_score(y_valid, np.array(p_valid) > 0.2))


# load test data
TEST_DIR = '../input/test-tif-v2/'
test_tif = os.listdir(TEST_DIR)

for file in tqdm(test_tif):
    img = tiff.imread(TEST_DIR + file)
    x_test.append(cv2.resize(img, (32, 32)))

x_test = np.array(x_test, np.float16) / 255.
y_pred = (model.predict(x_test) > 0.2)

output = open('/output/results.csv', 'w')
output.write('image_name,tags\n')

for i, row in enumerate(y_pred):
    string = test_tif[i][:-4] + ','
    for idx, x in enumerate(row):
        if (x):
            string += (inv_label_map[idx] + ' ')
    output.write(string + '\n')
    
output.close()