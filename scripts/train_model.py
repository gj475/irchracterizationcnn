"""Trains CNN model with optimal hyper-parameters."""

import numpy as np
import pickle
import pandas as pd
import plaidml.keras
import os
import random

plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import backend as K
from keras.models import Model
from keras.layers import Input, MaxPooling1D, Dropout, Activation
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.callbacks import Callback, LearningRateScheduler
from skopt import gp_minimize, callbacks
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver
from sklearn.metrics import f1_score, recall_score, precision_score


def train_model(X_train_val, y_train_val, num_fgs, aug, num, weighted):
    """Trains final model with the best hyper-parameters."""
    # Input
    X_train_val = X_train_val.reshape(X_train_val.shape[0], 600, 1)

    # Shape of input data.
    input_shape = X_train_val.shape[1:]
    input_tensor = Input(shape=input_shape)

    # 1st CNN layer.
    x = Conv1D(filters=31,
               kernel_size=(11), 
               strides=1,
               padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # 2nd CNN layer.
    x = Conv1D(filters=62,
       kernel_size=(11),
       strides=1,
       padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Flatten layer.
    x = Flatten()(x)

    # 1st dense layer.
    x = Dense(4927, activation='relu')(x)
    x = Dropout(0.48599073736368)(x)

    # 2nd dense layer.
    x = Dense(2785, activation='relu')(x)
    x = Dropout(0.48599073736368)(x)

    # 3rd dense layer.
    x = Dense(1574, activation='relu')(x)
    x = Dropout(0.48599073736368)(x)

    output_tensor = Dense(num_fgs, activation='sigmoid')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()
    optimizer = Adam()

    if weighted == 1:

        def calculate_class_weights(y_true):
            number_dim = np.shape(y_true)[1]
            weights = np.zeros((2, number_dim))
            # Calculates weights for each label in a for loop.
            for i in range(number_dim):
                weights_n, weights_p = (y_train_val.shape[0]/(2 * (y_train_val[:,i] == 0).sum())), (y_train_val.shape[0]/(2 * (y_train_val[:,i] == 1).sum()))
                # Weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
                weights[1, i], weights[0, i] = weights_p, weights_n

            return weights.T

        def get_weighted_loss(weights):
            def weighted_loss(y_true, y_pred):
                return K.mean((weights[:,0]**(1.0-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
            return weighted_loss

        model.compile(optimizer=optimizer, loss=get_weighted_loss(calculate_class_weights(y_train_val)))

    else:

        model.compile(optimizer=optimizer, loss='binary_crossentropy')

    def custom_learning_rate_schedular(epoch):
        if epoch < 31:
            return 2.5e-4
        elif 31 <= epoch < 37:
            return 2.5000001187436283e-05
        elif 37 <= epoch < 42:
            return 2.5000001187436284e-06

    callback = [LearningRateScheduler(custom_learning_rate_schedular, verbose=1)]

    # Start training.
    history = model.fit(x=X_train_val, y=y_train_val, epochs=41, batch_size=41, callbacks=callback)

    # Check if path exists.
    if not os.path.exists('./models/'):
        os.makedirs('./models/')

    # Save trained models.
    if aug == 'o':
        model.save('./models/' + str(num) + '_model_original.h5')
    elif aug == 'e':
        model.save('./models/' + str(num) + '_model_extended.h5')
    elif aug == 'c':
        model.save('./models/' + str(num) + '_model_control.h5')
    elif aug == 'h':
        model.save('./models/' + str(num) + '_model_horizontal.h5')
    elif aug == 'v':
        model.save('./models/' + str(num) + '_model_vertical.h5')
    elif aug == 'lc':
        model.save('./models/' + str(num) + '_model_linearcomb.h5')
    elif aug == 'w':
        model.save('./models/' + str(num) + '_model_weighted.h5')


if __name__ == '__main__':
    # Read data.
    with open('../processed_dataset/processed_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)

    # Delete last three columns and combine training set and validation set for final training.
    X_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['X_val_1']).iloc[:,:-2]])).astype('float32')
    y_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['y_val_1']).iloc[:,:-2]])).astype('float32')

    # Train extended model.
    train_model(X_train_val, y_train_val, 37, 'e', 0, 0)

    # Convert input to use for original model training.
    y_train_val = np.asarray(pd.DataFrame(y_train_val).drop([21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], axis=1))

    # Train original model.
    train_model(X_train_val, y_train_val, 22, 'o', 0, 0)

    # Read data for augmented and control models.
    with open('../augmented_dataset/augmented_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)

    # Train augmented and control models.
    for num in range(25, 101, 25):
        # Shuffle.
        z = list(zip(dict_data['X_train_val_' + str(num)], dict_data['y_train_val_' + str(num)]))
        random.shuffle(z)
        X, y = zip(*z)
        print(np.asarray(X))
        # Train.
        train_model(np.asarray(X, dtype=float), np.asarray(y, dtype=float), 37, 'c', num, 0)

        z = list(zip(dict_data['X_train_val_h_' + str(num)], dict_data['y_train_val_h_' + str(num)]))
        random.shuffle(z)
        X, y = zip(*z)
        train_model(np.asarray(X, dtype=float), np.asarray(y, dtype=float), 37, 'h', num, 0)

        z = list(zip(dict_data['X_train_val_v_' + str(num)], dict_data['y_train_val_v_' + str(num)]))
        random.shuffle(z)
        X, y = zip(*z)
        train_model(np.asarray(X, dtype=float), np.asarray(y, dtype=float), 37, 'v', num, 0)

        z = list(zip(dict_data['X_train_val_lc_' + str(num)], dict_data['y_train_val_lc_' + str(num)]))
        random.shuffle(z)
        X, y = zip(*z)       
        train_model(np.asarray(X, dtype=float), np.asarray(y, dtype=float), 37, 'lc', num, 0)