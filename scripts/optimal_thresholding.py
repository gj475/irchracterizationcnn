"""Calculates probability thresholds for the classification of each functional group."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import plaidml.keras
import os

plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


def optimal_threshold(X_train_val, y_train_val):
    """Calculates the optimal threshold for each functional group classification."""
    # Reshape input.
    X_train_val = X_train_val.reshape(X_train_val.shape[0], 600, 1)

    # Load model.
    current_model = load_model('../models/0_model_extended.h5')

    # Get probabilities.
    y_predict = current_model.predict(X_train_val)

    # Plot precision-recall curve for each class.
    precision = dict()
    recall = dict()
    thresholds = dict()
    average_precision = dict()

    for i in range(37):

        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_train_val[:, i], y_predict[:, i])
        average_precision[i] = average_precision_score(y_train_val[:, i], y_predict[:, i])

    optimal_thresholds = dict()

    for i in range(37):

        # Calculate f score.
        f_score = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

        # Locate the index of the largest f score.
        j = np.argmax(f_score)

        optimal_thresholds[i] = thresholds[i][j]

        # plt.plot(recall[i][j], precision[i][j], 'ro', linewidth=1)
        # print('optimal_threshold = %f, f_score = %.3f' % (thresholds[i][j], f_score[j]))
        # plt.step(recall[i], precision[i], where='post', linewidth=1)
        # plt.show()

    return optimal_thresholds


if __name__ == '__main__':
    # Read data.
    with open('../processed_dataset/processed_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)

    # Delete last two columns and combine training set and validation set for final training.
    X_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['X_val_1']).iloc[:,:-2]])).astype('float32')
    y_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['y_val_1']).iloc[:,:-2]])).astype('float32')

    optimal_threshold(X_train_val, y_train_val)