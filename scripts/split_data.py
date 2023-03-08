"""Creates training, validation, and test sets."""

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd
import pickle


# Define paths.
data_dir = '../processed_dataset'


def split_data(data_dir, fgs):
    """Splits data into n-fold train, validation, test sets for hyperparameter optimisation."""
    data_dictionary = {}
    # Load X and y data.
    y_df = pd.read_csv(data_dir + '/label_dataset.csv')
    y_df.columns = [i for i in range(fgs + 2)]
    x_df = pd.read_csv(data_dir + '/input_dataset.csv')
    x_df.columns = [i for i in range(602)]

    # Identify unique InChIs and add new column to use for linear combination method of augmenation.
    inchis = y_df[fgs].unique()
    num = 1
    y_df[39] = y_df[fgs]
    x_df[602] = x_df[600]
    for inchi in inchis:
        y_df[fgs + 2] = y_df[fgs + 2].replace(inchi, num)
        x_df[602] = x_df[602].replace(inchi, num)
        num += 1

    # Convert dataframes to numpy arrays.
    X = x_df.to_numpy()
    y = y_df.to_numpy()

    # Split data into training, validation, and test datasets.
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
    for train_val_index, test_index in msss.split(X, y):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]

    # Add test set to dictionary.
    data_dictionary['X_test'] = X_test
    data_dictionary['y_test'] = y_test
    
    # Create four-fold split of training and validation sets.
    mskf = MultilabelStratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    num = 1
    for train_index, val_index in mskf.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        data_dictionary['X_train_' + str(num)] = X_train
        data_dictionary['y_train_' + str(num)] = y_train
        data_dictionary['X_val_' + str(num)] = X_val
        data_dictionary['y_val_' + str(num)] = y_val
        num += 1

    # Save the dictionary as a pickle file.
    with open(data_dir + '/processed_dataset.pickle', 'wb') as handle:
        pickle.dump(data_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    split_data(data_dir, 37)