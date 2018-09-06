"""
This script is designed to evaluate DeepSloop models. It was specifically designed to test how well the ensemble models
perform, but it can work with greater generality.
"""

#
# Dependencies
#

from keras.models import load_model

import DeepSloop_Utils as DSU

import os


#
# Main
#

def DeepSloop_Evaluate(repository_path="../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_0",
                       model_results_path="../Model_Results",
                       model_file = "Bi_BiDo_Hyb_1.hdf5",
                       data_type='Va',
                       batch_size=16):
    """
    repository_path -- A string representing the  path to the directory of the data repository you wish to use
    model_results_path -- A string representing the  path to the directory containing the model you wish to use
    model -- A string repreenting the .hdf5 file name of the model you wish to use
    data_type -- A string indicating the type of data you would like to evaluate on:
                    'Va' -- validation set
                    'Te' -- test set
    batch_size -- An integer representing the number of sloops you will run through before back propogation
    """
    model_name = model_file.split('/')[-1].split('.hdf5')[0]
    fasta_file = repository_path.split('/')[-1].split('split_')[0]

    fasta_file_path = os.path.join(repository_path, fasta_file + '{}_NS_RNS.fasta'.format(data_type))

    X_tens, y_tens, num_validation_sloops, max_sloop_len = DSU.fasta_to_tens(fasta_file_path, 166)

    # Loading in your model
    model = load_model(os.path.join(model_results_path, model_file))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    results_list = model.evaluate(X_tens, y_tens, batch_size=batch_size)
    print(results_list)

DeepSloop_Evaluate()
