
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import os
from random import shuffle
from tqdm import tqdm

DATA_DIR = '../input/'
TRAIN_TIF_DIR = DATA_DIR + 'train-tif-v2/'
TRAIN_CSV = DATA_DIR + 'train.csv'

IMG_SIZE = 256
LR = 1e-3

MODEL_NAME = 'amazon-{}-{}.model'.format(LR, '6conv-large')


# In[4]:

from skimage import io
from scipy.misc import imresize
import cv2
import tifffile as tiff

# convert cloud cover labels to array [clear, cloudy, haze, partly_cloudy]
def get_labels(row):
    labels = np.array([row.clear, 
                       row.cloudy,
                       row.haze,
                       row.partly_cloudy,
                       row.agriculture,
                       row.cultivation,
                       row.habitation,
                       row.primary,
                       row.water,
                       row.road])
    return labels

# load image
# reduce image from 255,255,4 to 100,100,4
# flatten out to 1-D array in order R,G,B,NIR (should we use greyscale instead, ignore NIR?)
def load_image(filename):
    path = os.path.abspath(os.path.join(TRAIN_TIF_DIR, filename))
    if os.path.exists(path):
        img = tiff.imread(path)
        return img
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))
    
# create training data from train.csv DataFrame
def load_training_data():
    train_images = []

    for index, row in tqdm(train.iterrows()):
        grey_image = load_image(row.image_name + '.tif')
        train_images.append([grey_image, 
                             get_labels(row),
                             row.image_name])

    return train_images

# load test data from test data folder
# reduce image to 100,100,4, flatten etc as above
def create_test_data():
    test_images = []
    
    for image_name in os.listdir(TRAIN_TIF_DIR):
        grey_image = load_image(row.image_name + '.tif')
        test_images.append([grey_image, image_name.split('.')[0]])
        
    return test_images


# In[5]:

AVAILABLE_LABELS = [
    'agriculture', 
    'clear', 
    'cloudy', 
    'cultivation', 
    'habitation', 
    'haze', 
    'partly_cloudy', 
    'primary', 
    'road', 
    'water']

# read our data and take a look at what we are dealing with
train_csv = pd.read_csv(TRAIN_CSV)
train_csv.head()

tags = pd.DataFrame()

for label in AVAILABLE_LABELS:
    tags[label] = train_csv.tags.apply(lambda x: np.where(label in x, 1, 0))
    
train_csv = pd.concat([train_csv, tags], axis=1)
train_csv.head(n=2)

# In[6]:

import tensorflow as tf
tf.reset_default_graph()


# In[7]:

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 4], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='sigmoid')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='binary_crossentropy', name='targets')

model = tflearn.DNN(convnet)

# In[ ]:

for batch in range(0,3):
    train = train_csv[batch*8000:batch*8000 + 8000]
    
    train_images = load_training_data()
    
    train_data = train_images[:-1600]
    test_data = train_images[-1600:]

    X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 4)
    y = [i[1] for i in train_data]

    X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 4)
    y_test = [i[1] for i in test_data]

    model.fit({'input': X}, {'targets': y}, n_epoch=50, validation_set=({'input': X_test}, {'targets': y_test}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


# In[ ]:

model.save('/output/' + MODEL_NAME)

