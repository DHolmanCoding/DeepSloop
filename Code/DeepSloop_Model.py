"""
This is the Deep Sloop Model, containing a BiLSTM model, with an optional 1D convolution that was used for testing, but was
ultimately not included in the final model(s). This routine is responsible for testing and validation the model, as well
as producing visualizations and information regarding the model's performance.
"""

#
# Dependencies
#

import time
from datetime import datetime

from keras.models import Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Bidirectional, CuDNNLSTM
from keras.layers import Dropout, Dense, Flatten
import keras as keras
from keras import callbacks

from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint

import DeepSloop_Utils as DSU

import matplotlib.pyplot as plt

import os


#
# Main
#

# Set up a dictionary with the standard base-pairing rules for RNA, and set forth your set of bases
basepairing = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
bases = "ACUG"


def Deep_Sloop_Model(repository_path=r"../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_0",
                     Tr_data_type='NS_RNS',
                     Convolve=False,
                     cv_filters=128, cv_kernel_size=3, cv_strides=1,
                     mp_pool_size=2, mp_strides=2,
                     lstm_size=128,
                     batch_size=16, epochs=32,
                     custom_expt_name="",
                     custom_expt_var="",
                     custom_expt_value=""
                     ):
    """
    This is a BiLSTM(with optional convolution) designed to train on sloops.

    Arguments:
    repository_path -- A string representing the FULL path to the directory of the data repository you wish to use
                       Example: C:/..../filtered_sloops_split_datasets_0
    Tr_data_type -- A string representing the type of data you would like to train on for your deep sloop model
                    types of data:
                    NS - negated and scrambled
                    M_NS - mutated, negated and scrambled

    # Convolutional Layers
    filters -- An integer representing the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size -- An integer specifying the length of the 1D convolution window.
    CV_strides: Convolutional Strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution.
                Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.

    # Max Pooling
    pool_size -- An integer representing the size of the max pooling windows.
    MP_strides -- Max Pooling Strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input.
                  If None, it will default to pool_size.

    # LSTM Layers
    model_size1 -- An integer representation the size of the bidirectional LSTM Layer

    batch_size -- An integer representing the number of sloops you will run through before back propogation
    epochs -- An integer representing the number of times you will pass over your entire dataset
    """
    start_time = time.time()

    fasta_file_name = repository_path.split('Data/')[-1].split('split_datasets_')[0]
    Tr_fasta = os.path.join(repository_path, "{}Tr_{}.fasta".format(fasta_file_name, Tr_data_type))
    Val_fasta = os.path.join(repository_path, '{}Va_NS_RNS.fasta'.format(fasta_file_name))

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
    model.add(Bidirectional(CuDNNLSTM(lstm_size, return_sequences=True),
                            input_shape=(max_sloop_len, 4)))

    model.add(Bidirectional(CuDNNLSTM(lstm_size, return_sequences=False),
                            input_shape=(max_sloop_len, 4)))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_name = 'Adam'
    optimizer_fn = optimizers.Adam()
    loss_fn = 'binary_crossentropy'


    model.compile(optimizer=optimizer_fn,
                  loss=loss_fn,
                  metrics=['accuracy'])

    # Print out data pertinent to the I/O of each layer
    print("Model Architecture and I/O:")
    for layer in model.layers:
        print('Layer: {:>15}   Input Shape: {:<15}   Output Shape: {}\n'
              .format(layer.name, str(layer.input_shape), layer.output_shape))

    # Set the results directory
    results_dir = os.path.join(repository_path.split('/Data')[0], 'Model_Results')

    # Set up your callbacks
    weights_path = os.path.join(results_dir, 'weights.hdf5')

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

    # Display runtime information
    end_time = time.time() - start_time
    print('\nrun time = {:.3f} sec, {:.3f} min, {:.3f} hrs'.format(end_time, end_time / 60.0, end_time / 3600.0))

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    history_dict = history.history

    # Extract accuracy and loss data for your training and validation
    train_acc = history_dict['acc']
    train_loss = history_dict['loss']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epoch_range = range(1, len(train_acc) + 1)

    # Report how your model improved over the course of the training session (Looks like data is end of epoch)
    print('\n  acc change: {:.3f} -> {:.3f}'.format(train_acc[0], train_acc[-1]))
    print(' val_acc change: {:.3f} -> {:.3f}'.format(val_acc[0], val_acc[-1]))
    print('    loss change: {:.3f} -> {:.3f}'.format(train_loss[0], train_loss[-1]))
    print('val_loss change: {:.3f} -> {:.3f}'.format(val_loss[0], val_loss[-1]))

    # Compute core model metrics for naming file and model assessment purposes
    min_loss = min(train_loss)
    min_loss_idx = train_loss.index(min_loss)

    max_acc = max(train_acc)
    max_acc_idx = train_acc.index(max_acc)

    min_val_loss = min(val_loss)
    min_val_loss_idx = val_loss.index(min_val_loss)

    max_val_acc = max(val_acc)
    max_val_acc_idx = val_acc.index(max_val_acc)

    base_output_filename = 'run_{}_res_{}_{}_{:.3f}_{:.3f}_{:.3f}' \
        .format(timestamp, custom_expt_name, min_val_loss_idx, min_val_loss, val_acc[min_val_loss_idx], max_val_acc)

    # Select the weights file from your checkpoints that has the best performance, as measured by validation loss
    new_weights_filename = os.path.join(results_dir, base_output_filename + '.hdf5')
    os.rename(weights_filename, new_weights_filename)

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
    image_filename = os.path.join(results_dir, base_output_filename + '.pdf')
    plt.savefig(image_filename)

    # Output the validation data into a CSV for figure generation
    csv_filename = os.path.join(results_dir, base_output_filename + '.csv')
    with open(csv_filename, 'a+') as f:
        f.write('Trial,{},Epoch,Validation_Loss,Validation_Accuracy\n'.format(custom_expt_var))
        for epoch in range(len(epoch_range)):
            f.write('{},{},{},{},{}\n'.format(1, custom_expt_value, epoch, val_loss[epoch], val_acc[epoch]))

    # Log run details into a txt summary file
    log_filename = os.path.join(results_dir, base_output_filename + '.txt')
    with open(log_filename, 'w') as f:
        f.write('{}\n'.format(base_output_filename))
        f.write('\n')
        f.write('acc     : {:.4f} -> [{}]{:.4f} -> {:.4f}\n'.format(train_acc[0], max_acc_idx, max_acc, train_acc[-1]))
        f.write(
            'loss    : {:.4f} -> [{}]{:.4f} -> {:.4f}\n'.format(train_loss[0], min_loss_idx, min_loss, train_loss[-1]))
        f.write(
            'val acc : {:.4f} -> [{}]{:.4f} -> {:.4f}\n'.format(val_acc[0], max_val_acc_idx, max_val_acc, val_acc[-1]))
        f.write('val loss: {:.4f} -> [{}]{:.4f} -> {:.4f}\n'.format(val_loss[0], min_val_loss_idx, min_val_loss,
                                                                    val_loss[-1]))
        f.write('\n')
        f.write('val_acc at min_val_loss_idx = {:.4f}\n'.format(val_acc[min_val_loss_idx]))
        f.write('val_loss at max_val_acc_idx = {:.4f}\n'.format(val_loss[max_val_acc_idx]))
        f.write('\n')
        for layer in model.layers:
            f.write('Layer: {:>18}   Input Shape: {:<18}   Output Shape: {}\n'
                    .format(layer.name, str(layer.input_shape), layer.output_shape))
        f.write('\n')
        f.write('repository path = {}\n'.format(repository_path))
        f.write('Tr_data_type = {}\n'.format(Tr_data_type))
        f.write('Convolve = {}\n'.format(Convolve))
        if Convolve:
            f.write('cv_filters = {}\n'.format(cv_filters))
            f.write('cv_kernel_size = {}\n'.format(cv_kernel_size))
            f.write('cv_strides = {}\n'.format(cv_strides))
            f.write('mp_pool_size = {}\n'.format(mp_pool_size))
            f.write('mp_strides = {}\n'.format(mp_strides))

        f.write('lstm1 size = {}\n'.format(lstm_size))
        f.write('lstm2 size = {}\n'.format(lstm_size))
        f.write('epochs = {}\n'.format(epochs))
        f.write('batch size = {}\n'.format(batch_size))

        f.write('\n')
        f.write('optimizer fn = {}\n'.format(optimizer_name))
        f.write('loss fn = {}\n'.format(loss_fn))


#
# Main
#

if __name__ == '__main__':
    import argparse

    # define arguments for the command line
    parser = argparse.ArgumentParser()

    # declaring your arguments

    parser.add_argument('-rpath', default=r"C:/Users/Douglas/PycharmProjects/Deep_Sloop/Datasets/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_010_100mult",
                        type=str, help='A string representing the FULL path to the directory of the data repository you wish to use' )
    parser.add_argument('-Trdt', default='R_NS_RNS', type=str,
                        help='Tr_data_type -- A string representing the type of data you would like to train on for your deep sloop model'
                             ' | types of data:'
                             'R_NS - reversed, negated and scrambled'
                             'M_R_NS - mutated, reversed, negated and scrambled')
    parser.add_argument('-bs', default=16, type=int,
                        help='batch_size -- An integer representing the number of sloops you will run through before back propogation')
    parser.add_argument('-ep', default=32, type=int,
                        help='epochs -- An integer representing the number of times you will pass over your entire dataset')
    parser.add_argument('-cv', default=False, type=bool,
                        help='Convolve -- A Boolean indicating whether or not you wish to convolve your sequence prior to BiLSTM processing.')
    parser.add_argument('-cvf', default=128, type=str,
                        help='filters -- An integer representing the dimensionality of the output space (i.e. the number of output filters in the convolution).')
    parser.add_argument('-cvk', default=3, type=int,
                        help='kernel_size -- An integer specifying the length of the 1D convolution window.')
    parser.add_argument('-cvs', default=3, type=int,
                        help='CV_strides: Convolutional Strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.')
    parser.add_argument('-mps', default=2, type=int,
                        help='pool_size -- An integer representing the size of the max pooling windows.')
    parser.add_argument('-mpst', default=2, type=int,
                        help='MP_strides -- Max Pooling Strides: Integer, or None. Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.')
    parser.add_argument('-ms', default =128, type=int,
                        help='model_size1 -- An integer representation the size of the bidirectional LSTM Layer.')
    parser.add_argument('-cname', default="", type=str,
                        help='custom_expt_name -- A string representing the name of an experiment you are running to interrogate model performance')
    parser.add_argument('-cvar', default="", type=str,
                        help='custom_expt_var -- A string representing the variable/parameter you are changing in your experiment')
    parser.add_argument('-cexpt', default="", type=str,
                        help='custom_expt_value -- An integer to help you keep track of how are are altering the parameter of interest')

    args = parser.parse_args()

    Deep_Sloop_Model(args.rpath, args.Trdt, args.bs, args.ep, args.cv, args.cvf, args.cvk, args.cvs, args.mps, args.mpst, args.ms, args.cname, args.cvar, args.cexpt)
