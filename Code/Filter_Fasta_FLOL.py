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


def filter_fasta_FLOL(fasta_path=r""):
    """
    Takes in a fasta file and then makes a copy where you have filtered out sloops with less than a certain threshold
    of left parenthesis on the left.
    """
    base_fasta_path = fasta_path.split(".fasta")[0]
    FLOL_fasta_path = "{}_FLOL.fasta".format(base_fasta_path)

    sloop_dict_FLOL = {}
    sloop_dict = DSU.fasta_to_dict()
    for RNA_ID, sloop in sloop_dict.items():
        sloop_db, mfe, pstats_list = DSU.sloop_to_db(sloop)
        if pstats_list != 0:
            sloop_dict_FLOL[RNA_ID] = sloop_dict

    with open(FLOL_fasta_path) as f:
        for FLOL_RNA_ID, FLOL_sloop in sloop_dict_FLOL.items():
            f.write('>{}\n{}\n'.format(FLOL_RNA_ID, FLOL_sloop))



filter_fasta_FLOL()
