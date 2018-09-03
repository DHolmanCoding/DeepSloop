"""
This is the DeepSloop Script for model Training and Validation.
"""

#
# Dependencies
#

from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Bidirectional, CuDNNLSTM
from keras.layers import Dropout, Dense
import keras as keras
from keras import callbacks

from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint


import numpy as np
from numpy import array

import Deep_Sloop_Utils as DSU

import matplotlib.pyplot as plt

import os


# Set up a dictionary with the standard base-pairing rules for RNA, and set forth your set of bases
basepairing = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
bases = "ACUG"


def Deep_Sloop_Model(repository_path=r"C:/Users/Douglas/PycharmProjects/Deep_Sloop/Datasets/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_010_100mult",
                     Tr_data_type='R_NS_RNS',
                     batch_size=16, epochs=32,
                     Convolve=False,
                     cv_filters=128, cv_kernel_size=3, cv_strides=1,
                     mp_pool_size=2, mp_strides=2,
                     mode1_size=128,
                     custom_expt_name="",
                     custom_expt_var="",
                     custom_expt_value=""
                     ):
    """
    This is a Convolutional, BiLSTM designed to train on sloops.

    Arguments:
    repository_path -- A string representing the FULL path to the directory of the data repository you wish to use
                       Example: C:/......../filtered_sloops_split_datasets_0
    Tr_data_type -- A string representing the type of data you would like to train on for your deep sloop model
                    types of data:
                    R_NS - reversed, negated and scrambled
                    M_R_NS - mutated, reversed, negated and scrambled


    # Convolutional Layers
    filters -- An integer representing the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size -- An integer specifying the length of the 1D convolution window.
    CV_strides: Convolutional Strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.

    # Max Pooling
    pool_size -- An integer representing the size of the max pooling windows.
    MP_strides -- Max Pooling Strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.

    # Dropout
    dropout1 -- A floating point number between 0 and 1 representing the amount of dropout

    # LSTM Layers
    model_size1 -- An integer representation the size of the bidirectional LSTM Layer

    batch_size -- An integer representing the number of sloops you will run through before back propogation
    epochs -- An integer representing the number of times you will pass over your entire dataset
    """
    fasta_file_name = repository_path.split('Datasets/')[-1].split('split_datasets_')[0]
    Tr_fasta = os.path.join(repository_path, "{}Tr_{}.fasta".format(fasta_file_name, Tr_data_type))
    Val_fasta = os.path.join(repository_path, '{}Va_R_NS_RNS.fasta'.format(fasta_file_name))

    # find the maximum sloop length for purposes of padding out your sloops and establishing your Model
    max_sloop_len = max(DSU.ck_sloops_in_fasta(Tr_fasta), DSU.ck_sloops_in_fasta(Val_fasta))

    X_Tr_tens, y_Tr_tens, num_Tr_sloops, max_Tr_sloop_len = DSU.fasta_to_tens(Tr_fasta, max_sloop_len)
    X_Val_tens, y_Val_tens, num_Val_sloops, max_Val_sloop_len = DSU.fasta_to_tens(Val_fasta, max_sloop_len)

    assert X_Tr_tens.shape[1:] == X_Val_tens.shape[1:]

    # Setting up your model
    model = Sequential()

    if Convolve:
        print("Adding Convolutional Layers...\n")
        model.add(Conv1D(activation="relu",
                         input_shape=(max_sloop_len, 4),
                         filters=cv_filters,
                         kernel_size=cv_kernel_size,
                         strides=cv_strides,
                         dilation_rate=1))

        model.add(MaxPooling1D(pool_size=mp_pool_size, strides=mp_strides))

    print("Adding BiLSTM Layers...\n")
    model.add(Bidirectional(CuDNNLSTM(mode1_size, return_sequences=True),
                            input_shape=(max_sloop_len, 4)))

    model.add(Bidirectional(CuDNNLSTM(mode1_size, return_sequences=False),
                            input_shape=(max_sloop_len, 4)))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    # Print out data pertinent to the I/O of each layer
    print("Model Architecture and I/O:")
    for layer in model.layers:
        print('Layer: {:>15}   Input Shape: {:<15}   Output Shape: {}'
              .format(layer.name, str(layer.input_shape), layer.output_shape))
    print("")

    # Set up your callbacks
    wts_ext_num = 0
    wts_returned = False
    while not wts_returned:
        weights_name = "{}{}wts_{}_".format(fasta_file_name, Tr_data_type, wts_ext_num) + "{epoch:02d}-{val_acc:.3f}.hdf5"
        weights_path = repository_path.split('/Datasets')[0] + '/Results/{}'.format(weights_name)
        if os.path.exists(weights_path):
            wts_ext_num += 1
        else:
            wts_returned = True

    checkpointing = keras.callbacks.ModelCheckpoint(weights_path,
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_best_only=True)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=5,
                                                   verbose=1,
                                                   mode='auto',
                                                   baseline=None)

    # Fit your model and collect data from the training and validation
    history = model.fit(X_Tr_tens, y_Tr_tens,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(X_Val_tens, y_Val_tens),
                        verbose=2,
                        callbacks=[early_stopping, checkpointing])

    history_dict = history.history

    # Extract accuracy and loss data for your training and validation
    train_acc = history_dict['acc']
    train_loss = history_dict['loss']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epoch_range = range(1, len(train_acc) + 1)
    final_val_acc = str(round(float(val_acc[-1]), 3))[2:]

    # Report how your model improved over the course of the training session (Looks like data is end of epoch)
    print('\n  acc change: {:.3f} -> {:.3f}'.format(train_acc[0], train_acc[-1]))
    print(' val_acc change: {:.3f} -> {:.3f}'.format(val_acc[0], val_acc[-1]))
    print('    loss change: {:.3f} -> {:.3f}'.format(train_loss[0], train_loss[-1]))
    print('val_loss change: {:.3f} -> {:.3f}'.format(val_loss[0], val_loss[-1]))

    # Accuracy versus epochs for training and validation
    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epoch_range, train_acc, 'bo', label='Training Acc')
    plt.plot(epoch_range, val_acc, 'b', label='Validation Acc')
    plt.title('Training and Validation Acc')
    plt.legend()

    # Loss versus epochs for training and validation loss
    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epoch_range, train_loss, 'bo', label='Training Loss')
    plt.plot(epoch_range, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save your image file
    plt_ext_num = 0
    plt_made = False
    while not plt_made:
        image_name = '{}_{}_{}_fig{}'.format(fasta_file_name, Tr_data_type, final_val_acc, plt_ext_num)
        images_path = repository_path.split('/Datasets')[0] + '/Results/{}'.format(image_name)
        if os.path.exists(images_path):
            plt_ext_num += 1
        else:
            plt.savefig(images_path)
            plt_made = True

    # Output your validation data into a CSV for figure generation
    csv_name = '{}_{}_{}_{}.csv'.format(fasta_file_name, Tr_data_type, final_val_acc, custom_expt_name)
    csv_path = repository_path.split('/Datasets')[0] + '/Results/{}'.format(csv_name)
    if not os.path.exists(csv_path):
        with open(csv_path, 'a+') as f:
            f.write('Trial,{},Epoch,Validation_Loss,Validation_Accuracy\n'.format(custom_expt_var))
            for epoch in range(len(epoch_range)):
                f.write('{},{},{},{},{}\n'.format(1, custom_expt_value, epoch, val_loss[epoch], val_acc[epoch]))


Deep_Sloop_Model()
