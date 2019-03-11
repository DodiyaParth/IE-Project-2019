import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

data_df = pd.read_csv(os.path.join('C:/Users/admin/ML/car-behavioral-cloning-master', 'driving_log.csv'))

print(data_df)

X = data_df[['centre', 'left', 'right']].values
#X = data_df['centre'].values
y = data_df['sa'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

print(y_train)
