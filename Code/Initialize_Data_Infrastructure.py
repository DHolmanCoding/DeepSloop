"""
The purpose of this subroutine is to ultimately generate your data infrastructure with testing, training, and
validation data.

Run from Linux server, as calls are made to RNAfold.

Your data infrastructure within your Deep Sloop should be set up as follows:
>Deep Sloop
->Code
->Data
"""

#
# Dependencies
#

import DeepSloop_Utils as DSU

import os

import random

import math


#
# Definitions
#


bases = 'ACUG'

mutation_dict = {'A': 'CUG', 'U': 'ACG', 'C': 'AUG', 'G': 'ACU'}


def split_fasta_to_sets(fasta_file_path, train_prop=0.8, test_prop=0.1, val_prop=0.1):
    """
    This subroutine will take in a fasta file containing sloops, and separate it into training, testing and validation
    data sets based on the user's desired preferences.

    Arguments:
    fasta_file_path -- A string representing the FULL path to the directory of the fasta file as choice.
    train_prop -- a float between 0 and 1, indicating the proportion of the sloops
                    in your input file you desire to be in the training set.
    test_prop -- a float between 0 and 1, indicating the proportion of the sloops
                    in your input file you desire to be in the testing set.
    val_ prop -- a float between 0 and 1, indicating the proportion of the sloops
                    in your input file you desire to be in the validation set.

    NOTE: test_prop + val_prop + train_prop == 1  is a requirement!
    """
    assert(test_prop + val_prop + train_prop == 1)

    sloop_dict, max_sloop_len, num_sloops = DSU.fasta_to_dict(fasta_file_path)

    # Distribute the keys in accordance with the user's desired set proportions,
    # ensuring that you shuffle the keys randomly every time the subroutine is run.
    num_train_keys = int(math.floor(train_prop*num_sloops))
    num_val_keys = int(math.floor(val_prop*num_sloops))
    num_test_keys = int(math.floor(test_prop*num_sloops))
    assert(num_test_keys + num_val_keys + num_train_keys <= num_sloops)

    all_keys = list(sloop_dict.keys())
    random.shuffle(all_keys)

    # Collect the keys (RNA_IDs) of the proportioned sloops that will be allocated to each set
    train_keys = all_keys[num_test_keys + num_val_keys:]
    test_keys = all_keys[0:num_test_keys]
    val_keys = all_keys[num_test_keys:num_test_keys + num_val_keys]

    # Use the apportioned keys to create three dictionaries, and populate them with the split data
    train_dict = {}
    for RNA_ID in train_keys:
        RNA_ID_Tr = RNA_ID + '/Tr'
        train_dict[RNA_ID_Tr] = sloop_dict[RNA_ID]

    val_dict = {}
    for RNA_ID in val_keys:
        RNA_ID_Va = RNA_ID + '/Va'
        val_dict[RNA_ID_Va] = sloop_dict[RNA_ID]

    test_dict = {}
    for RNA_ID in test_keys:
        RNA_ID_Te = RNA_ID + '/Te'
        test_dict[RNA_ID_Te] = sloop_dict[RNA_ID]

    return fasta_file_path, train_dict, test_dict, val_dict
    # You are returning the fasta_file_path for naming of offspring files.


def mutate_sloop_data(sloop_dict, num_mutations=1):
    """
    This subroutine will take in a dictionary full of sloops, perform a random mutation to each sloop
    (single point mutation), and then return a dictionary full of the mutated sloops with the original sloops.

    Arguments:
    sloop_dict -- The variable name of the sloop containing dictionary you want to subject to mutation.
    num_mutations -- An integer representing the number of mutations you wish to introduce per sloop. Default is one.
    """
    mut_sloop_dict = {}

    for RNA_ID in sloop_dict:
        # pull out a sloop from the dictionary
        sloop = sloop_dict[RNA_ID]

        mutation_coordinates = set([])
        # randomly accumulate unique mutation coordinates
        while len(mutation_coordinates) != num_mutations:

            ran_index = random.randint(0, (len(sloop) - 1))
            mutation_coordinates.add(ran_index)

        for i in mutation_coordinates:
            mut_sloop = ''
            mut_base = sloop[ran_index]
            mut_sloop += (sloop[0:i] + random.choice(mutation_dict[mut_base]) + sloop[(i+1):])

        # populate a new dictionary with the mutated sequences, flagging sequences with _M for future identification
        mut_RNA_ID = RNA_ID + '_M'
        mut_sloop_dict[mut_RNA_ID] = mut_sloop
        mut_sloop_dict[RNA_ID] = sloop

    assert (2 * len(sloop_dict)) == len(mut_sloop_dict)
    return mut_sloop_dict


def reverse_sloop_data(sloop_dict):
    """
    This subroutine will take in a dictionary full of sloops, reverses each sloop,
    and then return a dictionary full of the both forward and reverse sloops.

    The purpose of this subroutine is primarily experimental.

    Arguments:
    sloop_dict -- The variable name of the sloop containing dictionary you want to subject to reversal.
    """
    reverse_sloop_dict = {}

    for RNA_ID in sloop_dict:
        sloop = sloop_dict[RNA_ID]
        rev_sloop = sloop[::-1]

        # populate a new dictionary with the original and reversed sequences,
        # flagging sequences with _R for future identification
        rev_RNA_ID = RNA_ID + '_R'
        reverse_sloop_dict[RNA_ID] = sloop
        reverse_sloop_dict[rev_RNA_ID] = rev_sloop

    return reverse_sloop_dict


def TNS_sloop_data(sloop_dict, multiplier=1, min_deltaG=-0.1):
    """
    This subroutine takes in a sloop dictionary containing sloops. Depending on the multiplier you select,
    it will negatively mirror a proportion of the sloops by rearranging the original sloops base composition,
    mark their fasta line, shuffle them, and then add the sloops (positive and negative) to a new dictionary
    with the marker 'TNS'.

    TNS means negatively mirrored sloops that stay true to the original sloop's base composition
    generated sloops with lengths matching the originals that are negatively mirrored AND shuffled

    Arguments:
    sloop_dict -- The variable name of the sloop containing dictionary you want to subject to negative mirroring.
    multiplier -- A floating point value representing the proportion of sloops from your original
                  data set you want to be negatively mirrored into your new data set. Satisfies 0 <= shuf_prop <= 1.
    min_deltaG -- The minimum average free energy you wish to allow to pass as a negative sloop in your dataset
    """
    orig_num_sloops = len(sloop_dict.keys())
    num_neg_sloops = int((orig_num_sloops * multiplier))

    if orig_num_sloops < num_neg_sloops:
        print('Please use a multiplier less than or equal to 1X when using negate_scramble_sloop_data')
        exit()

    counter = 0
    tneg_sloop_dict = {}
    while counter < num_neg_sloops:
        RNA_ID, sloop = DSU.pick_sloop_from_dict(sloop_dict)
        sloop_len = len(sloop)
        TNS_sloop = DSU.negative_sloop_mirror(sloop)
        TNS_RNA_ID = RNA_ID + '_TNS_{}'.format(counter)
        sloop_db, mfe, pstats_list = DSU.sloop_to_db(TNS_sloop)
        if (mfe / sloop_len) > min_deltaG:
            tneg_sloop_dict[TNS_RNA_ID] = TNS_sloop
            counter += 1
            if counter % 500 == 0:
                print("{}/{} sloops processed out of {} original sloops".format(counter, num_neg_sloops, orig_num_sloops))

    assert 1 * len(sloop_dict) == len(tneg_sloop_dict)

    # Add the original sloops to your new dictionary full of negated sloops
    for RNA_ID, sloop in sloop_dict.items():
        tneg_sloop_dict[RNA_ID] = sloop

    all_keys = list(tneg_sloop_dict.keys())
    random.shuffle(all_keys)  # Shuffle in place

    TNS_sloop_dict = {}
    for RNA_ID in all_keys:
        TNS_sloop_dict[RNA_ID] = tneg_sloop_dict[RNA_ID]

    return TNS_sloop_dict


def RNS_sloop_data(sloop_dict, multiplier=1, min_deltaG=-0.1):
    """
    This subroutine takes in a sloop dictionary containing sloops. Depending on the multiplier you select,
    it will randomly negatively mirror a proportion of the sloops, mark their fasta line, shuffle them, and then
    add the sloops (positive and negative) to a new dictionary with the marker 'RNS'.

    RNS means randomly generated sloops with lengths matching the originals that are negatively mirrored AND shuffled

    Arguments:
    sloop_dict -- The variable name of the sloop containing dictionary you want to subject to negative mirroring.
    multiplier -- A floating point value representing the proportion of sloops from your original
                  data set you want to be negatively mirrored into your new data set. Satisfies 0 <= shuf_prop <= 1.
    min_deltaG -- The minimum average free energy you wish to allow to pass as a negative sloop in your dataset
    """
    base_pool = ['A', 'U', 'G', 'C', 'U', 'C', 'A', 'G']

    orig_num_sloops = len(sloop_dict.keys())
    num_neg_sloops = int((orig_num_sloops * multiplier))

    if orig_num_sloops < num_neg_sloops:
        print('Please use a multiplier less than or equal to 1X when using negate_scramble_sloop_data')
        exit()

    counter = 0
    rneg_sloop_dict = {}
    while counter < num_neg_sloops:
        RNA_ID, sloop = DSU.pick_sloop_from_dict(sloop_dict)
        sloop_len = len(sloop)
        RNS_sloop = ''
        for i in range(sloop_len):
            RNS_sloop += random.choice(base_pool)
        RNS_RNA_ID = RNA_ID + '_RNS_{}'.format(counter)
        RNS_sloop_db, mfe, pstats_list = DSU.sloop_to_db(RNS_sloop)
        if (mfe / sloop_len) > min_deltaG:
            rneg_sloop_dict[RNS_RNA_ID] = RNS_sloop
            counter += 1
            if counter % 500 == 0:
                print("{}/{} sloops processed out of {} original sloops".format(counter, num_neg_sloops, orig_num_sloops))

    assert 1 * len(sloop_dict) == len(rneg_sloop_dict)

    # Add the original sloops to your new dictionary full of negated sloops
    for RNA_ID, sloop in sloop_dict.items():
        rneg_sloop_dict[RNA_ID] = sloop

    all_keys = list(rneg_sloop_dict.keys())
    random.shuffle(all_keys)  # Shuffle in place

    RNS_sloop_dict = {}
    for RNA_ID in all_keys:
        RNS_sloop_dict[RNA_ID] = rneg_sloop_dict[RNA_ID]

    return RNS_sloop_dict


def duplicate_shuffle_data(sloop_dict, multiplier):
    """
    This subroutine takes in a sloop dictionary containing sloops. Depending on the multiplier you select,
    it will duplicate the dataset a number of times, mark their fasta line, shuffle them, and then
    add the sloops to a new dictionary with the marker 'DSX'.

    DSX means negatively duplicated AND shuffled where X is the multiplier

    Arguments:
    sloop_dict -- The variable name of the sloop containing dictionary you want to subject to duplication.
    multiplier -- A integer representing the number of times you would like to duplicate the incoming sloop dataset.
    """
    dup_sloop_dict = {}

    for i in range(multiplier):
        for RNA_ID, sloop in sloop_dict.items():
            dup_sloop_dict[RNA_ID] = sloop

    all_keys = list(dup_sloop_dict.keys())
    random.shuffle(all_keys)  # Shuffle in place

    DS_sloop_dict = {}
    for RNA_ID in all_keys:
        DS_sloop_dict[RNA_ID] = dup_sloop_dict[RNA_ID]

    return DS_sloop_dict


#
# Main
#


def generate_data_split(fasta_file_path):
    """
    This subroutine will take in a path to a fasta file within your Dataset directory that you would like to use to
    create your data infrastructure. It will then generate pairwise disjoint subsets of data for training, validation,
    and testing. Each subset will contain one NS and one M_NS file per replicate, defined below.

    Types of Data:
    NS - negated and scrambled
    M_NS - mutated, negated and scrambled

    Arguments:
    fasta_file_path -- A string representing the FULL path to a fasta file within your Dataset directory
                        that you would like to use to generate your data infrastructure.
    """
    fasta_file_path, train_dict, test_dict, val_dict = split_fasta_to_sets(fasta_file_path)
    train_dict_M = mutate_sloop_data(train_dict)

    data_dir = '/'.join(fasta_file_path.split('/')[:-1])
    fasta_filename = fasta_file_path.split('/')[-1]

    file_ext_num = 0
    dir_made = False
    data_split_dir = ''
    while not dir_made:
        if os.path.exists(os.path.join(data_dir,
                                       "{}_split_datasets_{}".format(fasta_filename.split('.')[0], file_ext_num))):
            file_ext_num += 1
        else:
            data_split_dir = os.path.join(data_dir,
                                          "{}_split_datasets_{}".format(fasta_filename.split('.')[0], file_ext_num))
            os.mkdir(data_split_dir)
            dir_made = True

    # Make a dictionary with keys corresponding to subset datatype labels
    # and values corresponding to the associated dictionary
    subset_dict = {'Tr_NS': train_dict,
                   'Tr_M_NS': train_dict_M,
                   'Va_NS': val_dict,
                   'Te_NS': test_dict
                   }

    for label in subset_dict.keys():
        TNS_subset_file_path = os.path.join(data_split_dir, fasta_filename.split('.')[0] + '_{}_TNS.fasta'.format(label))
        TNS_sloop_dict = TNS_sloop_data(subset_dict[label])
        DSU.dict_to_fasta(TNS_sloop_dict, TNS_subset_file_path)

        RNS_subset_file_path = os.path.join(data_split_dir, fasta_filename.split('.')[0] + '_{}_RNS.fasta'.format(label))
        RNS_sloop_dict = RNS_sloop_data(subset_dict[label])
        DSU.dict_to_fasta(RNS_sloop_dict, RNS_subset_file_path)

    print('We have initialized your data infrastructure')
    print('We have completed splitting up your dataset.')
    print('Please find your split datasets in the data infrastructure.\n')


generate_data_split(r"../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL.fasta")
