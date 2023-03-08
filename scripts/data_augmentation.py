"""Creates augmented data of different sizes."""

from operator import add
from scipy import interpolate
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import random
import pickle
import pandas as pd
import numpy as np
import collections


def horizontal_aug(y, points):
    """Shift spectrum left or right by a few wavenumbers."""
    shift = [i for i in range(-10, 11)]
    shift.remove(0)
    x = np.linspace(4000, 400, points)
    x_new = np.linspace(4000, 400, 3600)
    f = interpolate.interp1d(x, y, kind='slinear')
    y = f(x_new)
    select = random.choice(shift)
    if select > 0:
        # Delete last few.
        y_r = y[:-select]
        for i in range(select):
            y_r = np.insert(y_r, 1, y[0])
        y = y_r
    elif select < 0:
        y_l = y
        for i in range(select*-1):
            y_l = np.delete(y_l, 0)
        last = y_l[len(y_l)-1]
        for i in range(select*-1):
            y_l = np.append(y_l, last)
        y = y_l
    f = interpolate.interp1d(x_new, y, kind='slinear')
    y = f(x)
    return y


def vertical_aug(y):
    """Add vertical noise."""
    for i in range(len(y)):
        constant = round(1 - y[i], 3)
        multiplier = round(random.uniform(-0.05, 0.05), 2)
        y[i] = y[i] + (multiplier * constant)
        if 0 < y[i] < 1:
            continue
        else:
            while 0 < y[i] < 1:
                y[i] = y[i] + (multiplier * constant1)
    return y


def linear_comb(X_train_val, y_train_val, num, repeats):
    """Creates new spetra by linearly combing multiple entries of the same compound."""
    ids = y_train_val[:, 39]
    # Functionals groups with multiple samples.
    fgs = [item for item, count in collections.Counter(ids).items() if count > 1]
    # Empty DataFrame.
    y_sample_total = pd.DataFrame(columns=[i for i in range(36)])
    X_sample_total = pd.DataFrame(columns=[i for i in range(600)])
    # Calculate remainder.
    remainder = -(len(fgs) - num)
    # Select and combine two spectra.
    for i in range(repeats):
        random.shuffle(fgs)
        for fg in fgs[:remainder]:
            # Convert to DataFrame.
            y_train_val = pd.DataFrame(y_train_val)
            X_train_val = pd.DataFrame(X_train_val)
            # Select duplicates.
            y_selection = y_train_val[y_train_val[39] == fg]
            X_selection = X_train_val.iloc[y_selection.index.values]
            # Calculate number of duplicates.
            dups = y_train_val[y_train_val[39] == fg].shape[0]
            sample_idx = random.sample([i for i in range(dups)], 2)
            # Select two samples.
            one = X_selection.iloc[sample_idx[0]].to_numpy()
            two = X_selection.iloc[sample_idx[1]].to_numpy()
            # Select classification info of first selection as both are same.
            y_sample_temp = y_selection.iloc[[0]]
            # Remove last three columns which contain extra info.
            one = one[:-3]
            two = two[:-3]
            # Generate two random coefficients that add to 1.
            constant1 = round(random.uniform(0.01, 0.99), 2)
            constant2 = round(1 - constant1, 2)
            # Multiply intensity of spectra with generated coefficients and combine.
            y1 = [x * constant1 for x in one]
            y2 = [x * constant2 for x in two]
            y3 = list(map(add, y1, y2))
            X_sample_temp = pd.DataFrame(y3).T
            y_sample_total = pd.concat([y_sample_total, y_sample_temp])
            X_sample_total = pd.concat([X_sample_total, X_sample_temp])
    # Remove last three columns with extra info.
    y_sample_total = y_sample_total.iloc[:,:-2]
    return X_sample_total, y_sample_total


def sampling(X_train_val, y_train_val, size):
    """Samples a specified proportion of the training data."""
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=size, random_state=0)
    for train_val_index, sample_index in msss.split(X_train_val, y_train_val):
        X_remainder, X_sample = X_train_val[train_val_index], X_train_val[sample_index]
        y_remainder, y_sample = y_train_val[train_val_index], y_train_val[sample_index]
    return X_sample, y_sample


if __name__ == '__main__':
    # Read data.
    with open('../processed_dataset/processed_dataset.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)
    # Combined training and validation sets.
    X_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['X_val_1']).iloc[:,:-2]]))
    y_train_val = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']).iloc[:,:-2], pd.DataFrame(dict_data['y_val_1']).iloc[:,:-2]]))
    # With ids.
    X_train_val_ids = np.asarray(pd.concat([pd.DataFrame(dict_data['X_train_1']), pd.DataFrame(dict_data['X_val_1'])]))
    y_train_val_ids = np.asarray(pd.concat([pd.DataFrame(dict_data['y_train_1']), pd.DataFrame(dict_data['y_val_1'])]))
    # Define test set.
    X_test = np.asarray(pd.DataFrame(dict_data['X_test']).iloc[:,:-2])#.astype('float32')
    y_test = np.asarray(pd.DataFrame(dict_data['y_test']).iloc[:,:-2])

    # Control oversampling.
    # 25 %
    sample = sampling(X_train_val, y_train_val, 0.25)
    X_sample_25 = sample[0]
    y_sample_25 = sample[1]
    # 50 %
    sample = sampling(X_train_val, y_train_val, 0.50)
    X_sample_50 = sample[0]
    y_sample_50 = sample[1]
    # 75 %
    sample = sampling(X_train_val, y_train_val, 0.75)
    X_sample_75 = sample[0]
    y_sample_75 = sample[1]
    # 100 %
    X_sample_100 = X_train_val
    y_sample_100 = y_train_val
    # Combine.
    X_train_val_25 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(X_sample_25)])
    y_train_val_25 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_25)])
    X_train_val_50 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(X_sample_50)])
    y_train_val_50 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_50)])
    X_train_val_75 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(X_sample_75)])
    y_train_val_75 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_75)])
    X_train_val_100 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(X_sample_100)])
    y_train_val_100 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_100)])

    # Horizontal shift.
    temp_h_25 = []
    temp_h_50 = []
    temp_h_75 = []
    temp_h_100 = []
    # 25 %
    for x in X_sample_25:
        temp_h_25.append(horizontal_aug(x, 600))
    # 50 %
    for x in X_sample_50:
        temp_h_50.append(horizontal_aug(x, 600))
    # 75 %
    for x in X_sample_75:
        temp_h_75.append(horizontal_aug(x, 600))
    # 100 %
    for x in X_sample_100:
        temp_h_100.append(horizontal_aug(x, 600))
    # Combine.
    X_train_val_h_100 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_h_100)])
    y_train_val_h_100 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_100)])
    X_train_val_h_75 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_h_75)])
    y_train_val_h_75 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_75)])
    X_train_val_h_50 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_h_50)])
    y_train_val_h_50 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_50)])
    X_train_val_h_25 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_h_25)])
    y_train_val_h_25 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_25)])

    # Vertical noise.
    temp_v_25 = []
    temp_v_50 = []
    temp_v_75 = []
    temp_v_100 = []
    # 25 %
    for x in X_sample_25:
        temp_v_25.append(vertical_aug(x))
    # 50 %
    for x in X_sample_50:
        temp_v_50.append(vertical_aug(x))
    # 75 %
    for x in X_sample_75:
        temp_v_75.append(vertical_aug(x))
    # 100 %
    for x in X_sample_100:
        temp_v_100.append(vertical_aug(x))
    # Combine.
    X_train_val_v_100 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_v_100)])
    y_train_val_v_100 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_100)])
    X_train_val_v_75 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_v_75)])
    y_train_val_v_75 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_75)])
    X_train_val_v_50 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_v_50)])
    y_train_val_v_50 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_50)])
    X_train_val_v_25 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_v_25)])
    y_train_val_v_25 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(y_sample_25)])

    # Linear combination.
    temp_lc_25 = linear_comb(X_train_val_ids, y_train_val_ids, len(X_sample_25), 1)
    temp_lc_50 = linear_comb(X_train_val_ids, y_train_val_ids, len(X_sample_50), 2)
    temp_lc_75 = linear_comb(X_train_val_ids, y_train_val_ids, len(X_sample_75), 3)
    temp_lc_100 = linear_comb(X_train_val_ids, y_train_val_ids, len(X_sample_100), 4)
    # Combine.
    X_train_val_lc_100 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_lc_100[0])])
    y_train_val_lc_100 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(temp_lc_100[1])])
    X_train_val_lc_75 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_lc_75[0])])
    y_train_val_lc_75 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(temp_lc_75[1])])
    X_train_val_lc_50 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_lc_50[0])])
    y_train_val_lc_50 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(temp_lc_50[1])])
    X_train_val_lc_25 = pd.concat([pd.DataFrame(X_train_val), pd.DataFrame(temp_lc_25[0])])
    y_train_val_lc_25 = pd.concat([pd.DataFrame(y_train_val), pd.DataFrame(temp_lc_25[1])])

    # Add all linearly combined data to dictionary.
    # Define empty dictionary.
    data_dictionary = {}
    # Add control/oversampled data to dictionary.
    data_dictionary['X_train_val_25'] = np.asarray(X_train_val_25)
    data_dictionary['y_train_val_25'] = np.asarray(y_train_val_25)
    data_dictionary['X_train_val_50'] = np.asarray(X_train_val_50)
    data_dictionary['y_train_val_50'] = np.asarray(y_train_val_50)
    data_dictionary['X_train_val_75'] = np.asarray(X_train_val_75)
    data_dictionary['y_train_val_75'] = np.asarray(y_train_val_75)
    data_dictionary['X_train_val_100'] = np.asarray(X_train_val_100)
    data_dictionary['y_train_val_100'] = np.asarray(y_train_val_100)
    # Add data with horizonal shift to dictionary.
    data_dictionary['X_train_val_h_25'] = np.asarray(X_train_val_h_25)
    data_dictionary['y_train_val_h_25'] = np.asarray(y_train_val_h_25)
    data_dictionary['X_train_val_h_50'] = np.asarray(X_train_val_h_50)
    data_dictionary['y_train_val_h_50'] = np.asarray(y_train_val_h_50)
    data_dictionary['X_train_val_h_75'] = np.asarray(X_train_val_h_75)
    data_dictionary['y_train_val_h_75'] = np.asarray(y_train_val_h_75)
    data_dictionary['X_train_val_h_100'] = np.asarray(X_train_val_h_100)
    data_dictionary['y_train_val_h_100'] = np.asarray(y_train_val_h_100)
    # Add data with vertical noise to dictionary.
    data_dictionary['X_train_val_v_25'] = np.asarray(X_train_val_v_25)
    data_dictionary['y_train_val_v_25'] = np.asarray(y_train_val_v_25)
    data_dictionary['X_train_val_v_50'] = np.asarray(X_train_val_v_50)
    data_dictionary['y_train_val_v_50'] = np.asarray(y_train_val_v_50)
    data_dictionary['X_train_val_v_75'] = np.asarray(X_train_val_v_75)
    data_dictionary['y_train_val_v_75'] = np.asarray(y_train_val_v_75)
    data_dictionary['X_train_val_v_100'] = np.asarray(X_train_val_v_100)
    data_dictionary['y_train_val_v_100'] = np.asarray(y_train_val_v_100)
    # Add linearly combined data to dictionary.
    data_dictionary['X_train_val_lc_25'] = np.asarray(X_train_val_lc_25)
    data_dictionary['y_train_val_lc_25'] = np.asarray(y_train_val_lc_25)
    data_dictionary['X_train_val_lc_50'] = np.asarray(X_train_val_lc_50)
    data_dictionary['y_train_val_lc_50'] = np.asarray(y_train_val_lc_50)
    data_dictionary['X_train_val_lc_75'] = np.asarray(X_train_val_lc_75)
    data_dictionary['y_train_val_lc_75'] = np.asarray(y_train_val_lc_75)
    data_dictionary['X_train_val_lc_100'] = np.asarray(X_train_val_lc_100)
    data_dictionary['y_train_val_lc_100'] = np.asarray(y_train_val_lc_100)
    # Add test data to dictionary.
    data_dictionary['X_test'] = X_test
    data_dictionary['y_test'] = y_test

    # Save dictionary as a pickle file.
    with open('../augmented_dataset/augmented_dataset.pickle', 'wb') as handle:
        pickle.dump(data_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)