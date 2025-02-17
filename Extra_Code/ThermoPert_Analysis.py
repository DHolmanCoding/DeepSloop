"""
The purpose of this subroutine is to investigate the effects of interrupting continuously paired regions within stem-loop
structures with random insertsions. The motivaiton for this is to understand the extent to which thermodynamic stability influences the
Deep Sloop model's predictions.

Q: Does subsequent introduction of insertions destabilize stems by raising mfe
Q: Does mfe vary with prediction?
"""

#
# Dependencies
#

from keras.models import load_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import DeepSloop_Utils as DSU

import scipy

import re

import random

import math

import os


#
# Definitions
#


#
# Main
#

# Set up a dictionary with the standard base-pairing rules for RNA, and set forth your set of bases
basepairing = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
bases = "ACUG"


def generate_insertions(
        repository_path=r"../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_010_100mult",
        num_insertions=5):
    """
    This subroutine loads in your validation dataset of choice, selects a sloop from the dataset, and then generates a
    list of sequences generated by making random insertions into the most thermodynamically stable region of the sloop.
    (as determined by longest continuous strand of paired bases).
    
    Arguments:
    repository_path -- A string representing the FULL path to the directory of the data repository you wish to use
                       Example: C:/......../filtered_sloops_split_datasets_0
    num_insertions -- An integer indicating the number of insertions you would like to introduce to the chosen sloop
    """
    fasta_file_name = repository_path.split('Data/')[-1].split('split_datasets_')[0]
    val_fasta = os.path.join(repository_path, '{}Va_R_NS_RNS.fasta'.format(fasta_file_name))

    sloop_dict, max_sloop_len, num_sloops = DSU.fasta_to_dict(val_fasta)

    seq_list = []
    mfe_list = []

    char_list = ['(', ')']

    sloop_picked = False

    # Select a positive sloop example, and double check to ensure it contains a paired region
    while not sloop_picked:
        RNA_ID, sloop = DSU.pick_sloop_from_dict(sloop_dict)
        sloop_db, mfe, p_stats_list = DSU.sloop_to_db(sloop)
        if 'RNS' not in RNA_ID and '(' in sloop_db and ')' in sloop_db:
            sloop_picked = True

    print('Original Dotbracket', sloop_db)
    seq_list.append(sloop)
    mfe_list.append(mfe)

    bounds_list = []

    # identify the longest consecutively paired region in your selected sloop
    for char in char_list:
        repeats = re.findall(r'\{}+'.format(char), sloop_db)
        print(repeats)
        max_reps = len(max(repeats))
        for pattern in re.finditer('\{}'.format(char) * max_reps, sloop_db):
            bounds_list.append((pattern.start(), pattern.end()))

    # Compute site of insertion
    ran_bounds = random.choice(bounds_list)
    mid_bound = math.floor((ran_bounds[0] + ran_bounds[1]) / 2)

    # Create your insertions and bundle the original and insertion variants sequences and mfes into a list
    for i in range(num_insertions):
        sloop = sloop[:mid_bound] + random.choice(bases) + sloop[mid_bound:]
        sloop_db, mfe, p_stats_list = DSU.sloop_to_db(sloop)
        print('{} Insertion Dotbracket: '.format(i + 1), sloop_db)
        seq_list.append(sloop)
        mfe_list.append(mfe)

    # Calculate normalized mfes (normalized for length of sloop)
    norm_mfe_list = []
    for i in range(len(mfe_list)):
        norm_mfe_list.append(mfe_list[i] / len(seq_list[i]))

    print(mfe_list)
    print(norm_mfe_list)
    print([len(seq) for seq in seq_list])
    print()

    return mfe_list, seq_list

def mfe_vs_ins():
    """
    This subroutine will allow you to visualize the relationship between mean free energy and the number of insertions.
    """
    mfe_LOL = []  # LOL = List of Lists
    mfe_list = []
    num_insertions = len(mfe_LOL[0])
    # Generate mfe versus insertion data for many randomly selected sloops
    for i in range(100):
        mfe_list_temp = generate_insertions()[0]
        mfe_LOL.append(mfe_list_temp)
        mfe_list.extend(mfe_list_temp)

    # Find the average mfe for each number of insertions
    avg_list = []
    for i in range(num_insertions):
        sum = 0
        for list in mfe_LOL:
            sum += list[i]
        avg_list.append(sum / len(mfe_LOL))

        print("For {} Insertions, the average mean free energy was {}".format(i, avg_list[i]))
    # print(avg_list)

    # Generate Plot
    plt.plot([i for i in range(len(mfe_LOL[0]))], mfe_LOL)
    plt.xlabel("Number of Insertions")
    plt.ylabel("Mean Free Energy")
    plt.show()
    plt.clf()


def mfe_vs_score(weights=r"../Model_Results/run_2018-08-07_10-36-26_res_bl128_bl128_do5_de1_7_0.134_0.948_0.950.hdf5"):
    """
    This subroutine will allow you ot visualize the relationship between mean score and the number of insertions

    Arguments:
    weights -- A string representing the path to the weights file of the model you wish to use.
    """
    mfe_list = []
    seq_list = []
    yhat_list = []
    yprime_list = []

    for i in range(100):
        mfe_list_temp, seq_list_temp = generate_insertions()
        mfe_list.extend(mfe_list_temp)
        seq_list.extend(seq_list_temp)

    model = load_model(weights)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    for sloop in seq_list:
        ohe_sloop = DSU.sloop_to_ohe(DSU.pad_sloop(sloop, 166))
        X = np.expand_dims(ohe_sloop, axis=0)
        yhat = float(model.predict(X, verbose=0)[0][0])
        yprime = float(scipy.special.logit(yhat))
        yhat_list.append(yhat)
        yprime_list.append(yprime)

    # Generate Plot
    plt.plot(mfe_list, yhat_list)
    plt.xlabel("Mean Free Energy")
    plt.ylabel("Yhat Score")
    plt.show()
    plt.clf()


    # for i in range(num_validation_sloops):
    #     X_temp = X_tens[i, :, :]
    #     X = np.expand_dims(X_temp, axis=0)
    #     y = float(y_tens[i, :])
    #
    #
    # print(yhat_list)
    # print(yprime_list)
    #
    # mfe = [-10.7, -8.40, -8.00, -7.60, -7.20]
    # labels = ['{} bulge additions'.format(i) for i in range(5)]
    #
    # fig, ax = plt.subplots()
    # ax.scatter(mfe, yhat_list)
    #
    # for i, txt in enumerate(labels):
    #     ax.annotate(txt, (mfe[i], yhat_list[i]))
    #
    # plt.title("Mean free energy versus yhat score with increasing bulge size")
    # plt.grid(False)
    # plt.xlabel("Mean free energy due to increasing bulge size")
    # plt.ylabel("yhat score of sloop")
    # plt.show()
    # plt.clf()
    #
    # plt.plot(mfe, yhat_list, 'ro')
    # plt.title("Mean free energy versus yhat score with increasing bulge size")
    # plt.grid(True)
    # plt.xlabel("Mean free energy due to increasing bulge size")
    # plt.ylabel("yhat score of sloop")
    # plt.show()
