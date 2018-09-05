"""
The purpose of this script is to take a fasta file that has been subjected to any number of rounds of preliminary
filtering, and apply a FLOL(fraction of left parenthesis on left) filter that permits only sloops above a certain
threshold of FLOL. The ultimate goal is to be able to generate a whole repository full of data with this FLOL
filtering parameter for use in the Deep Sloop Model.

FLOL = Fraction of left parenthesis on the left.
"""

#
# Dependencies
#

import DeepSloop_Utils as DSU

#
# Main
#


def filter_fasta_FLOL(fasta_path=r"../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150.fasta",
                      FLOL_thresh=0.9):
    """
    Takes in a fasta file and then makes a copy where you have filtered out sloops with less than a certain threshold
    of left parenthesis on the left.

    Arguments:
    fasta_path -- A string representing the path to a fasta file full of sloops you wish to subject to FLOL filtering
    FLOL_thresh -- A float in the interval [0,1] representing the threshold for FLOL. Only sloops above this threshold
                    will make it into the final dataset.
    """
    base_fasta_path = fasta_path.split(".fasta")[0]
    FLOL_fasta_path = "{}_FLOL.fasta".format(base_fasta_path)

    sloop_dict_FLOL = {}
    sloop_dict, max_sloop_len, num_sloops = DSU.fasta_to_dict(fasta_path)

    for RNA_ID, sloop in sloop_dict.items():
        sloop_db, mfe, pstats_list = DSU.sloop_to_db(sloop)
        if len(pstats_list) == 4 and pstats_list[0] >= FLOL_thresh:
            sloop_dict_FLOL[RNA_ID] = sloop_dict

    print("{}/{} sloops in your dataset met the desired FLOL threshold of {}".format(len(sloop_dict_FLOL), len(sloop_dict), FLOL_thresh))

    with open(FLOL_fasta_path, 'w') as f:
        for FLOL_RNA_ID, FLOL_sloop in sloop_dict_FLOL.items():
            f.write('>{}\n{}\n'.format(FLOL_RNA_ID, FLOL_sloop))

filter_fasta_FLOL()
