
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


# In[ ]:

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

# convert cloud cover labels to array [clear, cloudy, haze, partly_cloudy]
def get_cloud_cover_labels(row):
    labels = np.array([row.clear, row.cloudy, row.haze, row.partly_cloudy])
    return labels

# load image
# reduce image from 255,255,4 to 100,100,4
# flatten out to 1-D array in order R,G,B,NIR (should we use greyscale instead, ignore NIR?)
def load_image(filename):
    path = os.path.abspath(os.path.join(TRAIN_TIF_DIR, filename))
    if os.path.exists(path):
        img = tiff.imread(path)[:,:,:3]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))
    
# create training data from train.csv DataFrame
def load_training_data():
    train_images = []

    for index, row in tqdm(train.iterrows()):
        grey_image = load_image(row.image_name + '.tif')
        train_images.append([grey_image, 
                             get_cloud_cover_labels(row),
                             row.image_name])
    return train_images


# In[ ]:

# 256 x 256
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, name='targets')

model = tflearn.DNN(network)


# In[ ]:

train_images = load_training_data()

train_data = train_images[:-8]
# need a cross validation set
cv_data = train_images[-8:-4]
test_data = train_images[-4:]

X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = [i[1] for i in train_data]

X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = [i[1] for i in test_data]

X_cv = np.array([i[0] for i in cv_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_cv = [i[1] for i in cv_data]

model.fit({'input': X}, {'targets': y}, n_epoch=1000, validation_set=({'input': X_cv}, {'targets': y_cv}), 
  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


# In[ ]:

model.save('/output/' + MODEL_NAME)


# In[ ]:

# need to measure F2 score instead of accuracy
y_pred = model.predict(X_test)
score = f2_score(y_test, y_pred)

