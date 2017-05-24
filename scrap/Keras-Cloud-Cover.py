
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import os
from random import shuffle
from tqdm import tqdm
from skimage import io
from scipy.misc import imresize
import cv2
import tifffile as tiff

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# In[ ]:

DATA_DIR = '../input/'
TRAIN_TIF_DIR = DATA_DIR + 'train-tif-v2/'
TRAIN_CSV = DATA_DIR + 'train.csv'
TEST_TIF_DIR = DATA_DIR + 'test-tif/'

IMG_SIZE = 227
LR = 1e-3

MODEL_NAME = 'amazon-{}-{}.model'.format(LR, 'alexnet')

CLOUD_COVER_LABELS = [
    'clear', 
    'cloudy', 
    'haze', 
    'partly_cloudy']

# read our data and take a look at what we are dealing with
train_csv = pd.read_csv(TRAIN_CSV)
train_csv.head()

tags = pd.DataFrame()

for label in CLOUD_COVER_LABELS:
    tags[label] = train_csv.tags.apply(lambda x: np.where(label in x, 1, 0))
    
train_csv = pd.concat([train_csv, tags], axis=1)


# In[17]:

train = pd.concat([train_csv[train_csv.clear == 1].sample(n=7251),
    train_csv[train_csv.cloudy == 1].sample(n=7251),
    train_csv[train_csv.haze == 1],
    train_csv[train_csv.partly_cloudy == 1].sample(n=7251)], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

del train_csv
del tags

for f, tags in tqdm(train.values, miniters=1000):
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

model.compile(loss='categorical_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
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
                     
# In[ ]:

model.save('/output/' + MODEL_NAME)


# In[ ]:

# need to measure F2 score instead of accuracy
y_pred = model.predict(X_test)
score = f2_score(y_test, y_pred)