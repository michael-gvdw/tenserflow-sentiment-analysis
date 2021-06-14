import numpy as np
import pandas as pd

from sklearn import model_selection

# load in dataset
df = pd.read_csv("./assets/original_data/IMDB Dataset.csv")

# view dataset
print(df.shape)
print(df.head())


# convert label from string to numerical 0: negative, 1: positive
df['sentiment'] = df['sentiment'].apply(lambda sentiment: 1 if sentiment == "positive" else 0)
print(df.head())

# add padding to all sentences in order for all sentences to be of same length


# split data into train and test data
y = df.pop('sentiment')
X = df

X_train, y_train, X_test, y_test = model_selection.train_test_split(X, y, random_state=42)