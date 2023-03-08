"""Performs optimization of the hyper-parameters of the CNN."""

import numpy as np
import pickle
import pandas as pd
import plaidml.keras
import os

plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import backend as K
from keras.models import Model
from keras.layers import Input, MaxPooling1D, Dropout, Activation
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score
from skopt import gp_minimize
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from skopt import callbacks
from skopt.callbacks import CheckpointSaver


# Define paths.
data_dir = '../processed_dataset/'
params_dir = '../searched_parameters/'
results_dir = '../checkpoints/'

# Make directories if does not exist.
if not os.path.exists(params_dir):
    os.makedirs(params_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Define search space.
dim_num_dense_layers = Integer(low=1, high=4, name='num_dense_layers')
dim_num_filters = Integer(low=4, high=32, name='num_filters')
dim_dense_divisor = Real(low=0.25, high=0.8, name='dense_divisor')
dim_num_cnn_layers = Integer(low=1, high=5, name='num_cnn_layers')
dim_dropout = Real(low=0, high=0.5, name='dropout')
dim_batch_size = Integer(low=8, high=512, name='batch_size')
dim_kernel_size = Integer(low=2, high=12, name='kernel_size', dtype='int')
dim_num_dense_nodes = Integer(low=1000, high=5000, name='num_dense_nodes')

dimensions = [dim_num_dense_layers,
              dim_num_filters,
              dim_dense_divisor,
              dim_num_cnn_layers,
              dim_dropout,
              dim_batch_size,
              dim_kernel_size,
              dim_num_dense_nodes]


# Load training data.
with open(data_dir + 'processed_dataset.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)

# Define training, validation, and test sets.
X_train = pd.DataFrame(dict_data['X_train_1'])
X_val = pd.DataFrame(dict_data['X_val_1'])
y_train = pd.DataFrame(dict_data['y_train_1'])
y_val = pd.DataFrame(dict_data['y_val_1'])
X_test = pd.DataFrame(dict_data['X_test'])
y_test = pd.DataFrame(dict_data['y_test'])

# Remove extra info.
X_train = X_train.drop(X_train.columns[[600, 601]], axis=1)
X_test = X_test.drop(X_test.columns[[600, 601]], axis=1)
X_val = X_val.drop(X_val.columns[[600, 601]], axis=1)
y_train = y_train.drop(y_train.columns[[37, 38]], axis=1)
y_test = y_test.drop(y_test.columns[[37, 38]], axis=1)
y_val = y_val.drop(y_val.columns[[37, 38]], axis=1)

# Convert to array.
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')
X_val = np.asarray(X_val).astype('float32')
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_val = np.asarray(y_val).astype('float32')

# Shape data for input to CNN.
X_train = X_train.reshape(X_train.shape[0], 600, 1)
X_val = X_val.reshape(X_val.shape[0], 600, 1)

# Define variables.
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]

# Parameters for CNN that do not have to be saved.
maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5

# Parameters that change for each iteration that must be saved.
list_early_stop_epochs = []
list_validation_loss = []
list_saved_model_name = []


class Metrics(Callback):
    """Define loss function."""
    def __init__(self, validation):   
        super(Metrics, self).__init__()
        self.validation = validation    
        
    def on_train_begin(self, logs={}):        
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()
        val_targ = self.validation[1]

        val_f1 = f1_score(val_targ, val_predict, average='micro')
        val_recall = recall_score(val_targ, val_predict, average='micro')         
        val_precision = precision_score(val_targ, val_predict, average='micro')
        
        self.val_f1s.append(round(val_f1, 6))
        self.val_recalls.append(round(val_recall, 6))
        self.val_precisions.append(round(val_precision, 6))
        
        global f1score
        f1score = val_f1
         
        print(f' — val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}')
        return


def create_model(num_dense_layers,
                 num_filters,
                 dense_divisor,
                 num_cnn_layers,
                 dropout,
                 kernel_size,
                 num_dense_nodes):
    """Creates a CNN model with given parameters."""
    # Start construction of a Keras Functional model.
    input_tensor = Input(shape=input_shape)
    
    # Convolutional layers.
    x = Conv1D(filters=num_filters, 
               kernel_size=(kernel_size), 
               strides=1, 
               padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    num_filters = num_filters * 2   
    for layer in range(num_cnn_layers - 1):
        x = Conv1D(filters=num_filters, 
                   kernel_size=(kernel_size), 
                   strides=1, 
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        num_filters = num_filters * 2

    # Flatten the output of the convolutional layers
    x = Flatten()(x)

    # Dense layers.
    x = Dense(num_dense_nodes, activation='relu')(x)
    num_dense_nodes = int(num_dense_nodes * dense_divisor)
    x = Dropout(dropout)(x)
    for i in range(num_dense_layers - 1):
        x = Dense(num_dense_nodes, activation='relu')(x)
        x = Dropout(dropout)(x)
        num_dense_nodes = int(num_dense_nodes * dense_divisor)

    # Define output tensor.
    output_tensor = Dense(num_classes, activation='sigmoid')(x)
    
    # Instantiate model.
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # Print model summary.
    model.summary()
    
    # Use the Adam method for training the network.
    optimizer = Adam(lr=2.5e-4)
    
    # Compile the model so it can be trained.
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=[])
      
    return model


@use_named_args(dimensions=dimensions)
def fitness(num_dense_layers,
            num_filters,
            dense_divisor,
            num_cnn_layers,
            dropout,
            batch_size,
            kernel_size,
            num_dense_nodes):
    """Defines the settings for the optimization of hyper-parameters."""
    # Print the chosen hyper-parameters for the epoch.
    print('num_dense_layers:', num_dense_layers)
    print('num_filters:', num_filters)
    print('dense_divisor:', dense_divisor)
    print('num_cnn_layers:', num_cnn_layers)
    print('dropout:', dropout)
    print('batch_size:', batch_size)
    print('kernel_size', kernel_size)
    print('num_dense_nodes', num_dense_nodes)

    # Create model name and print.
    model_name = 'cnn_' + str(np.random.uniform(0, 1, ))[2:9]
    print('model_name:', model_name)

    # Create the neural network with these hyper-parameters.
    model = create_model(num_dense_layers=num_dense_layers,
                         num_filters=num_filters,
                         dense_divisor=dense_divisor,
                         num_cnn_layers=num_cnn_layers,
                         dropout=dropout,
                         kernel_size=kernel_size,
                         num_dense_nodes=num_dense_nodes)
    
    # Create a callback-function for Keras which will be run after each epoch has ended during training.
    callback_list = [EarlyStopping(monitor='val_loss', patience=early_stop_epochs),                     
                     ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.1, 
                                       patience=learning_rate_epochs, 
                                       verbose=1, 
                                       mode='auto', 
                                       min_lr=1e-6),
                     ModelCheckpoint(filepath=results_dir + model_name + '.h5',
                                     monitor='val_loss', 
                                     save_best_only=True),
                     Metrics(validation=(X_val, y_val))]
    
    # Use Keras to train the model.
    history = model.fit(x=X_train,
                        y=y_train,
                        epochs=maximum_epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=callback_list)
    
    # Define validation loss.
    val_loss = history.history['val_loss'][-1]
    
    # Record actual best epochs and validation loss here, added to bayes opt parameter df below.
    list_early_stop_epochs.append(len(history.history['val_loss']) - early_stop_epochs)
    list_validation_loss.append(np.min(history.history['val_loss']))
    list_saved_model_name.append(model_name)

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session.
    K.clear_session()
    
    return val_loss


# Define checkpoints.
checkpoint_saver = CheckpointSaver(results_dir + 'checkpoint.pkl', compress=9)
# Run optimization.
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=50,
                            callback=[checkpoint_saver],
                            n_jobs=-1)


# Store search results in a dataframe. 
results_list = []
for result in zip(search_result.func_vals, 
                  search_result.x_iters, 
                  list_early_stop_epochs, 
                  list_validation_loss,
                  list_saved_model_name):
    temp_list = []
    temp_list.append(result[0])
    temp_list.append(result[2])
    temp_list.append(result[3])
    temp_list.append(result[4])
    temp_list = temp_list + result[1]
    results_list.append(temp_list)

# Define columns of the dataframe.
df_results = pd.DataFrame(results_list, columns=['last_val_loss', 
                                                 'epochs', 
                                                 'lowest_val_loss', 
                                                 'model_name',
                                                 'num_dense_layers',
                                                 'num_filters', 
                                                 'dense_divisor', 
                                                 'num_cnn_layers', 
                                                 'dropout',
                                                 'batch_size',
                                                 'kernel_size',
                                                 'num_dense_nodes'])

# Save as pickle and csv files.
df_results.to_pickle(params_dir + 'searched_parameters.pkl')
df_results.to_csv(params_dir + 'searched_parameters.csv')