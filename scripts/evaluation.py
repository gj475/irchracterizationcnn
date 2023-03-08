"""Demonstrates evaluation of the model."""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import plaidml.keras
import os

plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import load_model
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from optimal_thresholding import optimal_threshold
from smarts import fg_list_extended, label_names_extended


def model_predict(X_test, y_test, loaded_model, optimal_thresh, optimal):
    """Predicts probabilties for all classes for a given model."""
    # Prediction probabilities.
    y_probabilities = loaded_model.predict(X_test)

    # Classify probabilities.
    y_pred = []
    # Apply a threshold of 0.5.
    if optimal == 0: 
        for prob in y_probabilities:
            y_pred.append([1 if k>=0.5 else 0 for k in prob])
    # Apply optimal thresholds.
    elif optimal == 1: 
        for prob in y_probabilities: 
            single = []
            for i in range(37):
                if prob[i]>=optimal_thresh[0]:
                    single.append(1)
                else:
                    single.append(0)
            y_pred.append(single)

    return y_pred


def f_score(y_test, y_pred, y_train_val, label_names):
    """Calculates the F1-score, precision, and recall."""
    fs = []
    pr = []
    re = []
    for i in range(len(label_names)):
        temp_test = []
        temp_pred = []
        for test_sample, pred_sample in zip(y_test, y_pred):
            temp_test.append(test_sample[i])
            temp_pred.append(pred_sample[i])
        # F1-score.
        fs.append(f1_score(temp_test, temp_pred))
        # Precision.
        pr.append(precision_score(temp_test, temp_pred))
        # Recall.
        re.append(recall_score(temp_test, temp_pred))
    
    # Count number of samples per functional group.
    data = pd.DataFrame(y_train_val)
    count = []
    for i in range(len(label_names)):
        count.append(int(sum(data[i])))
    
    # Order functional groups in order of highest to lowest F1-score.
    names = [label for _, label in sorted(zip(fs, label_names))]
    
    # Reorder precision, recall, and sample count.
    re_fs = sorted(fs)
    re_pr = []
    re_rc = []
    re_count = []
    for name in names:
        idx = label_names.index(name)
        # Precision.
        re_pr.append(pr[idx])
        # Recall.
        re_rc.append(re[idx])
        # Sample count.
        re_count.append(count[idx])

    # Create DataFrame.
    result = pd.DataFrame(
        {'F-score': re_fs,
         'Precision': re_pr,
         'Recall': re_rc,
         'Frequency': re_count
        }, index=names)
    # Index title.
    result.index.name = 'FGs'
    print(result)

    return re_fs, re_pr, re_rc, re_count, names


def emr_fgs(y_test, y_pred, y_train_val):
    """Calculates the exact match rate per number of functional groups in a single molecule."""
    # Calculate max sum.
    sum_ls = []
    for sample in y_train_val:
        fg_sum = int(sum(sample))
        if fg_sum not in sum_ls:
            sum_ls.append(fg_sum)
    max_sum = max(sum_ls)
    num_fgs = list(np.arange(1, max_sum + 1))
    # Count number of samples per number of functional groups.
    train = dict()
    # Create empty lists.
    for i in range(1, max_sum + 1): 
        if i in train:
            continue
        else:
            train[i] = []
    # Append to lists.
    for train_sample in y_train_val:
        fg_sum = sum(train_sample)
        for i in range(1, max_sum + 1):
            if fg_sum == i:
                train[i].append(train_sample)
    # Count.
    count = []
    for i in range(1, max_sum + 1):
        count.append(len(train[i]))
    # Exact match ratio calculation.
    test = dict()
    pred = dict()
    # Create empty lists.
    for i in range(1, max_sum + 1):
        if i in test:
            continue
        else:
            test[i] = []
            pred[i] = []
    # Append to lists.
    for test_sample, pred_sample in zip(y_test, y_pred):
        fg_sum = sum(test_sample)
        for i in range(1, max_sum + 1):
            if fg_sum == i:
                test[i].append(test_sample)
                pred[i].append(pred_sample)
    emr = []
    acc_emr = []
    for i in range(1, max_sum + 1):
        # Non-accumulative.
        emr.append(accuracy_score(test[i], pred[i]))
        if i == 1:
            acc_test = test[i]
            acc_pred = pred[i]
        else:
            acc_test = acc_test + test[i]
            acc_pred = acc_pred + pred[i]
        # Accumulative.
        acc_emr.append(accuracy_score(acc_test, acc_pred))

    # Create DataFrame.
    result = pd.DataFrame(
        {'Individual EMR': emr,
         'Accumulative EMR': acc_emr,
         'Frequency': count
        }, index=num_fgs)
    # Index title.
    result.index.name = 'No. of FGs'
    print(result)

    return emr, acc_emr, count, num_fgs


def emr_class(y_test, y_pred, y_train_val):
    """Calculates the exact match rate per number of classes in a single molecule."""
    y_test = pd.DataFrame(y_test)
    y_pred = pd.DataFrame(y_pred)
    
    # Count number of samples.
    count = []
    data = pd.DataFrame(y_train_val)
    for i in range(37):
        count.append(int(sum(data[i])))
    # Sort count.
    re_count = sorted(count, reverse=True)
    
    # Get index order of sorted count.
    idxs = []
    for i in re_count:
        idxs.append(count.index(i))
    
    # Calculate exact match ratio per number of classes considered.
    cols = []
    emr = []
    for idx in idxs:
        # Considered columns.
        cols.append(idx)
        emr.append(accuracy_score(y_test[cols], y_pred[cols]))
    num_class = list(np.arange(1, 38))

    # Create DataFrame.
    result = pd.DataFrame(
        {'Accumulative EMR': emr,
         'Frequency': re_count
        }, index=num_class)
    # Index title.
    result.index.name = 'No. of classes'

    print(result)

    return emr, re_count, num_class


def accuracy(y_test, y_pred, y_train_val):
    """Finds the accuracies for the presence and absence of functional groups as well as the overall accuracy."""
    # accuracies = accuracy(y_test, y_pred)
    # Calculate accuracies.
    acc_pr = []
    acc_abs = []
    for i in range(37):
        temp_test = []
        temp_pred = []
        temp_test_pr = []
        temp_pred_pr = []
        temp_test_abs = []
        temp_pred_abs = []
        for test_sample, pred_sample in zip(y_test, y_pred):
            # Presence.
            if test_sample[i] == 1:
                temp_test_pr.append(test_sample[i])
                temp_pred_pr.append(pred_sample[i])
            # Absence.
            elif test_sample[i] == 0:
                temp_test_abs.append(test_sample[i])
                temp_pred_abs.append(pred_sample[i])
        acc_pr.append(accuracy_score(temp_test_pr, temp_pred_pr))
        acc_abs.append(accuracy_score(temp_test_abs, temp_pred_abs))
    # Count number of samples
    count = []
    data = pd.DataFrame(y_train_val)
    for i in range(37):
        count.append(int(sum(data[i])))
    # Order in terms of highest accuracy for presence of functional groups.
    names = [label for _, label in sorted(zip(acc_pr, label_names_extended))]
    re_count = []
    re_acc_abs = []
    re_acc_pr = sorted(acc_pr)
    for name in names:
        idx = label_names_extended.index(name)
        re_acc_abs.append(acc_abs[idx])
        re_count.append(count[idx])

    pos_per = [x / y_train_val.shape[0] * 100 for x in re_count] # Postive class percentage.
    neg_per = [(100 - x) for x in pos_per] # Negative class percentage.

    # Create DataFrame.
    result = pd.DataFrame(
        {'Presence': re_acc_pr,
         'Absence': re_acc_abs,
         '+ve %': pos_per,
         '-ve %': neg_per,
        }, index=names)
    # Index title.
    result.index.name = 'FGs'

    return re_acc_pr, re_acc_abs, pos_per, neg_per, names, result


def avg_precision(X_test, y_test, loaded_model):
    """Summarises precision-recall curves buy calculating the average precision."""
    # Prediction probabilities.
    y_probabilities = loaded_model.predict(X_test)
    # Calculate average precision.
    y_test = np.array(y_test)
    y_probabilities = np.array(y_probabilities)
    precisions = []
    for i in range(37):
        average_precision_score(y_test[:, i], y_probabilities[:, i])
        precisions.append(average_precision_score(y_test[:, i], y_probabilities[:, i]))

    return precisions


if __name__ == '__main__':
    # Read data.
    with open('../processed_dataset/processed_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)

    # Load model.
    loaded_model = load_model('..Calculating/models/0_model_extended.h5')

    # Combined training and validation sets.
    X_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['X_val_1']).iloc[:,:-2]])).astype('float32')
    y_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['y_val_1']).iloc[:,:-2]])).astype('float32')
    # Define test set.
    X_test = np.asarray(pd.DataFrame(dict_data['X_test']).iloc[:,:-2]).astype('float32')
    y_test = np.asarray(pd.DataFrame(dict_data['y_test']).iloc[:,:-2]).astype('float32')
    # Shape input data into three dimensions.
    X_test = X_test.reshape(X_test.shape[0], 600, 1)

    # Calculate optimal thresholds for each functional groups.
    optimal_thresh = optimal_threshold(X_train_val, y_train_val)

    # Make predictions.
    y_pred = model_predict(X_test, y_test, loaded_model, optimal_thresh, 0)

    # Display evaluation.
    # F-score, precision, recall, and sample frequency.
    print('Calculating F-score, precision, recall, and sample frequency of functional groups (FGs) in Figure 2.\n')
    f_score(y_test, y_pred, y_train_val, label_names_extended)
    print('\n')
   
    # Exact match ratio based on functional groups.
    print('Calculating exact match rate (EMR) based on number of functional groups (FGs) in Figure 3.\n')
    emr_fgs(y_test, y_pred, y_train_val)
    print('\n')
    # Exact match ratio based on classes.
    print('Calculating exact match rate (EMR) based on number of classes in Figure 3.\n')
    emr_class(y_test, y_pred, y_train_val)
    print('\n')

    # Average precision based on functional groups.
    print('Calculating average precision (AP) for functional groups in Figure 4.\n')
    # Plot average precision.
    names = [label for _, label in sorted(zip(avg_precision(X_test, y_test, loaded_model), label_names_extended))]
    ap = sorted(avg_precision(X_test, y_test, loaded_model))
    # Create DataFrame.
    result = pd.DataFrame(
        {'AP': ap,
         'Random classifier': accuracy(y_test, y_pred, y_train_val)[2],
        }, index=names)
    # Index title.
    result.index.name = 'FGs'
    print(result)
    print('\n')

    # Accuracies for presence and absence of functional groups.
    print('Calculating accuracies for presence and absence of functional groups in Figure 5.\n')
    print(accuracy(y_test, y_pred, y_train_val)[5])
    print('\n')