"""
Deep Sloop Hyperopt Script for hyperparameter tuning of models
"""

#
# Import Dependencies
#

import time

start_time = time.time()

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL

from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D

import numpy as np
from numpy import array

import DeepSloop_Utils as DSU

import os

#
# Definitions
#

def data():
    """
    Read in the fasta file full of sloops and output the data in tensor foramt.
    """
    repository_path = r"../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_0"
    Tr_data_type = 'NS_RNS'

    fasta_file_name = repository_path.split('Data/')[-1].split('split_datasets_')[0]
    Tr_fasta = os.path.join(repository_path, "{}Tr_{}.fasta".format(fasta_file_name, Tr_data_type))
    Val_fasta = os.path.join(repository_path, '{}Va_NS_RNS.fasta'.format(fasta_file_name))

    # find the maximum sloop length for purposes of padding out your sloops and establishing your Model
    max_sloop_len = max(DSU.ck_sloops_in_fasta(Tr_fasta), DSU.ck_sloops_in_fasta(Val_fasta))

    X_Tr_tens, y_Tr_tens, num_Tr_sloops, max_Tr_sloop_len = DSU.fasta_to_tens(Tr_fasta, max_sloop_len)
    X_Val_tens, y_Val_tens, num_Val_sloops, max_Val_sloop_len = DSU.fasta_to_tens(Val_fasta, max_sloop_len)

    assert X_Tr_tens.shape[1:] == X_Val_tens.shape[1:]

    return X_Tr_tens, y_Tr_tens, X_Val_tens, y_Val_tens, max_sloop_len


X_train, Y_train, X_test, Y_test, max_sloop_len = data()

# Uncomment, but do not delete parameters as try out tuning for different features
param_values = {
    # 'filters': [64, 128],
    # 'kernel_size': [2, 3, 4],
    # 'strides': [1],
    # 'dropout0': [0.3, 0.4, 0.5, 0.6],
    'bilstm1': [128],
    'bilstm2': [128],
    'bilstm3': [32, 48, 64, 80, 96, 112, 128],
    # 'dense1': [32, 48, 64, 80, 96, 112, 128],
    # 'dropout1': [0.3, 0.4, 0.5, 0.6],
    # 'dense2': [16, 32, 64],
}

space = {
    # 'filters': hp.choice('filters', param_values['filters']),
    # 'kernel_size': hp.choice('kernel_size', param_values['kernel_size']),
    # 'strides': hp.choice('strides', param_values['strides']),
    # 'dropout0': hp.choice('dropout0', param_values['dropout0']),
    'bilstm1': hp.choice('bilstm1', param_values['bilstm1']),
    'bilstm2': hp.choice('bilstm2', param_values['bilstm2']),
    'bilstm3': hp.choice('bilstm3', param_values['bilstm3']),
    # 'dense1': hp.choice('dense1', param_values['dense1']),
    # 'dropout1': hp.choice('dropout1', param_values['dropout1']),
    # 'dense2': hp.choice('dense2', param_values['dense2']),
}


def ds_model(params):
    """
    Define model, compile and fit
    """

    print('>>> Params: {}'.format(params))
    # if params['strides'] > params['kernel_size']:
    #     print('    >>>> strides ({}) > kernel_size ({})'.format(params['strides'], params['kernel_size']))
    #     return {'loss': -0.0, 'status': STATUS_FAIL}

    # Build the model
    # ===============
    model = Sequential()

    # model.add(Conv1D(
    #     activation='relu',
    #     input_shape=(max_sloop_len, 4),
    #     filters=params['filters'],
    #     kernel_size=params['kernel_size'],
    #     strides=params['strides']
    # ))
    #
    # model.add(MaxPooling1D(pool_size=2,
    #                        strides=2))
    #
    # model.add(Dropout(params['dropout0']))

    model.add(Bidirectional(CuDNNLSTM(params['bilstm1'],
                                      return_sequences=True),
                            input_shape=(max_sloop_len, 4)))

    model.add(Bidirectional(CuDNNLSTM(params['bilstm2'],
                                      return_sequences=True)))

    model.add(Bidirectional(CuDNNLSTM(params['bilstm3'])))

    # model.add(Dense(params['dense1'], activation='relu'))
    # model.add(Dropout(params['dropout1']))
    # model.add(Dense(params['dense2'], activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # print model details
    # ===================
    model.summary()
    for layer in model.layers:
        print('layer: {:>15}   input shape: {:<15}   output shape: {}'
              .format(layer.name, str(layer.input_shape), layer.output_shape))

    # Compile the model
    # =================
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    # check_pointer = ModelCheckpoint(filepath='keras_weights.hdf5',
    #                                verbose=1,
    #                                save_best_only=True)

    # Fit the data
    # ============
    history = model.fit(X_train, Y_train,
                        epochs=16,
                        batch_size=16,
                        validation_data=(X_test, Y_test),
                        # callbacks=[early_stopping, check_pointer],
                        verbose=2)

    history_dict = history.history

    # Extract accuracy and loss data to report your results
    # tr_acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    # tr_loss = history_dict['loss']
    # val_loss = history_dict['val_loss']

    print('Test accuracy:', val_acc[-1])
    return {'loss': -val_acc[-1], 'status': STATUS_OK}


#
# Main
#

trials = Trials()
best = fmin(fn=ds_model,
            space=space,
            algo=tpe.suggest,
            max_evals=16,
            trials=trials)

print()
print('Best run:')
for key in best:
    print('    {} = {}'.format(key, param_values[key][best[key]]))

result_list = []
for trial in trials:
    result_list.append([trial['result']['loss'], trial['misc']['vals']])

result_list.sort(key=lambda x: x[0])
print(result_list)

print()
for res in result_list:
    print('val_acc: {:.4f} for: '.format((res[0] * (-1))), end='')
    for p in res[1]:
        index = res[1][p][0]
        print('{} = {:>3}, '.format(p, param_values[p][index]), end='')
    print('')

print()
end_time = time.time() - start_time
print('run time = {:.3f} sec, {:.3f} min, {:.3f} hrs'.format(end_time, end_time/60.0, end_time/3600.0))
