
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import os
from random import shuffle
from tqdm import tqdm

DATA_DIR = '../input/amazon/'
TRAIN_TIF_DIR = DATA_DIR + 'train-tif/'
TRAIN_CSV = DATA_DIR + 'train.csv'
TEST_TIF_DIR = DATA_DIR + 'test-tif/'

IMG_SIZE = 100
LR = 1e-3

MODEL_NAME = 'amazon=-{}-{}.model'.format(LR, '2conv-basic')

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

pd.concat([train_csv[train_csv.clear == 1].sample(n=7251),
train_csv[train_csv.cloudy == 1].sample(n=7251),
train_csv[train_csv.haze == 1],
train_csv[train_csv.partly_cloudy == 1].sample(n=7251)], axis=0, ignore_index=True)

