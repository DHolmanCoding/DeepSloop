"""
Utility file for the Deep Sloop project. File manipulation utilities and more!
"""

#
# Dependencies
#

import Deep_Sloop_Utils as DSU

from numpy import array

import numpy as np

import random

import math

import os


#
# Definitions
#

def fasta_to_dict(fasta_file_name, max_sloops=None):
    """
    Reads in a FASTA file containing multiple sequences(stem loops (sloops)), and reads all sequences into a dictionary.

    Dictionary is set up as follows:
        key = information in defline (e.g. segment ID and hairpin number)
        value = sequence (sloop corresponding to information in defline)
        sloop_dict = { info : sloop }

    Additionally, this subroutine will scan through your set of sloops and determine the longest sloop in order
    to pad out entries for your LSTM

    Arguments:
    fasta_file_name -- A string representing your FASTA input file name. This FASTA file should contain sloops.
    max_sloops -- An integer representing the maximum number of sloops you want to read in from your FASTA file.
                  This is useful for creating smaller training datasets(subsets).
    """
    sloop_dict = {}
    max_sloop_len = 0
    counter = 0
    print(f'We are reading {fasta_file_name} into a dictionary')

    # Collect unique sloops into your sloop dictionary
    with open(fasta_file_name, 'r') as f:
        for line in f:
            line = line.strip()  # remove newline
            if line[0] == '>':
                defline = line[1:]
                if defline in sloop_dict:
                    print('Sorry, this sloop is already in the data -- fatal error')
                    exit()
                continue

            sloop = line
            sloop_dict[defline] = sloop
            if len(sloop) > max_sloop_len:
                max_sloop_len = len(sloop)

            counter += 1
            if counter == max_sloops:
                break

    print(f'We have read {len(sloop_dict)} sloops from {fasta_file_name} into sloop_dict')
    print(f'max_sloop_len = {max_sloop_len}\n')

    num_sloops = len(sloop_dict.keys())

    return sloop_dict, max_sloop_len, num_sloops


def dict_to_fasta(sloop_dict, fasta_file_name):
    """
    Takes a dictionary containing stem loops and writes them to a FASTA file.

    Dictionary is set up as follows:
        key = Defline
        value = Sloop
        sloop_dict = { info : sloop}

    Arguments:
    sloop_dict -- The variable name of the sloop-containing dictionary you wish to write to a file.
    fasta_file_name -- A string representing your FASTA input file name. This FASTA file should contain sloops.
    """
    with open(fasta_file_name, 'w') as f:
        for RNA_ID, sloop in sloop_dict.items():
            f.write(f'>{RNA_ID}\n{sloop}')


def pad_sloop(sloop, max_sloop_len, pad_5p=False):
    """
    This subroutine takes in a sloop and the maximum sloop length of the dataset you are working with, and if the sloop
    is not at the maximum length, it will format the sloop for the LSTM by padding the sequence with 'X' up to the
    length of the maximum sloop. (This should be the maximum sloop length in the in the FASTA file you are using.)

    This routine performs padding off of the 3' end of the sequence.

    Arguments:
    sloop -- A sloop you would like to pad out with a dummy variable up to the maximum sloop length
    max_sloop_len -- An integer representing the length of the largest sloop in the dataset you are handling
    pad_5p -- A Boolean that if left False, will perform padding off of the 3' end of the seqeunce,
              and if passed as True, will perform padding off of the 5' end of the sequence.
    """
    if len(sloop) > max_sloop_len:
        print('Sorry, this sloop is larger than the maximum permissible sloop length -- fatal error')
        exit()

    elif len(sloop) < max_sloop_len:
        if not pad_5p:
            padded_sloop = sloop + ('X' * (max_sloop_len - len(sloop)))
        else:
            padded_sloop = ('X' * (max_sloop_len - len(sloop))) + sloop

    else:
        padded_sloop = sloop

    return padded_sloop


def sloop_to_ohe(sloop):
    """
    Takes in a stem loop sequence as a string, and outputs an array of one hot vectors encoding the sequence.

    Key:
    A = [1,0,0,0]
    U = [0,1,0,0]
    C = [0,0,1,0]
    G = [0,0,0,1]
    X = [0,0,0,0]

    Arguments:
    sloop -- A string representing the sequence of a stem loop.
    """
    base_to_hot = {'A': array([1., 0., 0., 0.]),
                   'U': array([0., 1., 0., 0.]),
                   'C': array([0., 0., 1., 0.]),
                   'G': array([0., 0., 0., 1.]),
                   'X': array([0., 0., 0., 0.])
                   }

    # Process your sloop into an array of floats that represents your sequence
    sloop_array = np.array([base_to_hot[base] for base in sloop])

    return sloop_array


def ohe_to_sloop(sloop_array):
    """
    Takes in a one hot encoded stem loop sequence as a numpy array, and outputs a string representing the original,
    unpadded sequence.

    Key:
    [1,0,0,0] = A
    [0,1,0,0] = U
    [0,0,1,0] = C
    [0,0,0,1] = G
    [0,0,0,0] = X

    Arguments:
    sloop_array -- A numpy arrray representing the sequence of a stem loop (can be padded).
    """
    hot_to_base = {hash(tuple(array([1., 0., 0., 0.]))): 'A',
                   hash(tuple(array([0., 1., 0., 0.]))): 'U',
                   hash(tuple(array([0., 0., 1., 0.]))): 'C',
                   hash(tuple(array([0., 0., 0., 1.]))): 'G',
                   hash(tuple(array([0., 0., 0., 0.]))): ''
                   }

    sloop = ''
    assert int(sloop_array.shape[1]) == 4

    for i in range(int(sloop_array.shape[0])):
        sloop += hot_to_base[hash(tuple(sloop_array[i]))]

    return sloop


def fasta_to_tens(path_to_fasta,
                  model_max_sloop_len=None,
                  ):
    """
    This subroutine will generate and populate a sloop tensor from the sloops in a fasta file.
    The dimensions of the tensor will be (num_sloops)x(max_sloop_len)x(num_features)

    It will also generate an array that is filled with the binary classifications corresponding to each sloop
    The dimensions of this array will be (num_sloops)

    Note: Your FASTA file must have information for classification in the defline (e.g. '_RNS_')

    Arguments:
    path_to_fasta -- A string representing the FULL path to the file of the fasta file you wish to use
    model_max_sloop_len -- This is an integer representing the length of the longest sloop from the dataset
    that was used to generate the weights you are using for validation. If you are not performing validation,
    pass no arguments and it will default to None.
    """
    # Create a dictionary full of training and testing sloops
    # Find the maximum sloop length for purposes such as padding out your sloops and establishing your LSTM size
    sloop_dict, max_sloop_len, num_sloops = DSU.fasta_to_dict(path_to_fasta)
    ohe_features = 4  # This is hardcoded, as if you are working with genetic data, your features should remain 4

    # Ensure you pass on the correct max_sloop_len for sizing the LSTM if you intend to validate
    if model_max_sloop_len is not None:
        max_sloop_len = model_max_sloop_len

    # Set up and fill up two  tensors: one for OHE sloops, and one for the classifications
    print('Initializing tensors...\n')
    X_tens = np.zeros((num_sloops, max_sloop_len, ohe_features))
    y_tens = np.zeros((num_sloops, 1))

    print('Filling tensors with data...\n')
    i = 0  # set up a counter to fill up your training tensors
    for RNA_ID, sloop in sloop_dict.items():
        if "RNS" in RNA_ID.split('_') or "TNS" in RNA_ID.split('_'):
            y_tens[i] = array([0.])
        else:
            y_tens[i] = array([1.])

        sloop_array = sloop_to_ohe(pad_sloop(sloop, max_sloop_len))
        X_tens[i] = sloop_array
        i += 1

    print('Your tensors have been initialized...\n')

    return X_tens, y_tens, num_sloops, max_sloop_len


def negative_sloop_mirror(sloop):
    """
    Mirrors positive loops to their randomized negative counterparts.

    Takes in a string that represents a sequence of a sloop, then completely shuffles
    it to output a randomized sequence with the same base composition as the input sloop.

    Arguments:
    sloop -- A string representing the sequence of a stem loop
    """
    import random

    neg_sloop = ''.join(random.sample(sloop, len(sloop)))

    assert sloop != neg_sloop

    return neg_sloop


def pick_sloop_from_dict(sloop_dict):
    """
    Indexes a random value in your input dictionary in order to get a sample stem loop, and then outputs it.

    Arguments:
    sloop dict -- the variable name of the dictionary you wish to select sloops from
    """
    random_number = random.randrange(0, len(sloop_dict))

    RNA_ID = list(sloop_dict.keys())[random_number]
    sloop = sloop_dict[RNA_ID]

    return RNA_ID, sloop


def ck_sloops_in_fasta(fasta_file):
    """
    Takes a FASTA file and reports the number of lines, number of sloops,
    the length of the longest sloops, and the contents of the longest line(longest sloop).

    fasta_file = a string representing the path to the fasta file you would like to check
    """
    num_lines = 0
    num_sloops = 0
    current_max_len = 0
    longest_sloop = ''

    with open(fasta_file, 'r') as file:
        for line in file:
            num_lines += 1
            if line[0] == '>':
                continue
            if current_max_len < len(line):
                current_max_len = len(line)
                longest_sloop = line
            num_sloops += 1

    assert num_sloops * 2 == num_lines

    print('Results for {}:'.format(fasta_file))
    print('{} lines'.format(num_lines))
    print('{} sloops'.format(num_sloops))
    print('The longest sloop is {} characters.'.format(current_max_len))
    print('This is the sloop: {}\n'.format(longest_sloop))

    return current_max_len


def sloop_to_db(sloop):
    """
    This subroutine will take in a string representing a sloop, or more generally a sequence, and will return the
    dotbracket form of the sloop, the mean free energy (mfe), the ratio of left to right parenthesis, and the
    percentage of parenthesis on either side of the theoretical middle of the sloop.

    To be run on the linux GPU server equipped with RNAfold.

    Arguments:
    sloop -- A string represnting the RNA sequence of a stem loop.
    """
    RNAfold_cmd = "echo '{}' | RNAfold".format(sloop)
    RNAfold_results = os.popen(RNAfold_cmd)

    for line in RNAfold_results.readlines():
        if '(' in line:
            assert len(line.strip().split(' (')) == 2  # Most of these assertions are to protect from RNAfold bugs
            sloop_db, mfe = line.strip().split(' (')  # This is to avoid an obnoxious bug from RNAfold
            mfe = float(mfe.strip().split(")")[0])

            assert type(mfe) == float
            assert len(sloop) == len(sloop_db)

            if len(sloop) % 2 == 0:
                theo_middle = int(len(sloop) / 2)
            else:
                theo_middle = int(math.floor(len(sloop) / 2))

            sloop_db_left = sloop_db[0:theo_middle]
            sloop_db_right = sloop_db[theo_middle:]

            # p denotes parenthesis, lp denotes left parenthesis, and rp denotes right parenthesis
            total_lp = sloop_db.count("(")
            total_rp = sloop_db.count(")")

            if total_lp == 0 or total_rp == 0:  # Prevent divide by zero
                p_stats_list = []
            else:
                # Compile a number of potentially interesting data into a list
                frac_lp_left = sloop_db_left.count("(") / total_lp
                frac_lp_right = sloop_db_right.count("(") / total_lp
                frac_rp_left = sloop_db_left.count(")") / total_rp
                frac_rp_right = sloop_db_right.count(")") / total_rp

                # Soft assertion to ensure no data is being lost
                assert frac_lp_left + frac_lp_right > 0.95
                assert frac_rp_left + frac_rp_right > 0.95

                p_stats_list = [frac_lp_left, frac_lp_right, frac_rp_left, frac_rp_right]

            return sloop_db, mfe, p_stats_list
