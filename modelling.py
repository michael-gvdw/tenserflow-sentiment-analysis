import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

# load data from tensorflow
dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

# split into train and test data
train_dataset, test_dataset = dataset['train'], dataset['test']


BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).b