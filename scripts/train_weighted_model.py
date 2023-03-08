"""Trains weighted CNN model with optimal hyper-parameters."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling1D, Dropout, Activation
from tensorflow.keras.layers import Conv1D, Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from skopt import gp_minimize, callbacks
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
from sklearn.metrics import f1_score, recall_score, precision_score
from train_model import train_model


if __name__ == '__main__':
    # Read data.
    with open('../processed_dataset/processed_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)

    # Delete last three columns and combine training set and validation set for final training.
    X_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['X_val_1']).iloc[:,:-2]])).astype('float32')
    y_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['y_val_1']).iloc[:,:-2]])).astype('float32')

    # Train weighted model.
    train_model(X_train_val, y_train_val, 37, 'w', 0, 1)