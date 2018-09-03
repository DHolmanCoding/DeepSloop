"""
This script is designed to consolidate the raw data from allSegments_Filtered.txt and allHairpins_Filtered.txt in order to create
a dataset full of stem loops (sloops) from hairpin and segment data.

Hairpin data is arranged as such:
{ RNA ID : [ ( (hp_start, hp_end) , hp_seq ) ] }
where:
    RNA ID -- the RNA ID value carried through from the database
    hp_start -- absolute position of the first nucleotide in the segment
    hp_end -- absolute position of the last nucleotide in the segment
    hp_seq -- sequence of segment x2 as a string

Segment data is arranged as such:
{ RNA ID : [ ( (end_5p, start_3p) , seq_5p , seq_3p , bp ) ] }
where:
    RNA ID -- the RNA ID value carried through from the database
    end_5p -- absolute position of the first nucleotide on the 5p segment
    start_3p -- absolute position of the last nucleotide on the 3p segment
    seq_5p -- sequence of segment seq_5p as a string
    seq_3p -- sequence of segment seq_3p as a string
    bp -- keeps a record of the number of basepairs between seq_5p and seq_3p
"""

#
# Dependencies
#

import os


#
# Definitions
#

def read_segment_file(segment_file, min_segment_len, max_segment_len):
    """
    This subroutine will take in a tab delimited file full of the segment portion of RNA hairpins,
    and will extract all the information about them necessary to reconstruct the sloops from the sequences.

    Arguments:
    segment_file -- A tab delimited type file containing information about the segment portion of RNA hairpins
    min_segment_len -- An integer representing the minimum allowable segment length
    max_segment_len -- An integer representing the maximum allowable segment length
    Note: In this context, the segment length is referring to the SUM OF THE SEGMENTS
    """
    seg_dict = {}

    # Step through your file, grabbing up information on segments
    with open(segment_file, 'r') as file_in:
        for line in file_in:
            RNA_ID, seg_id, bp, start_5p, end_5p, seq_5p, len_5p, start_3p, end_3p, seq_3p, len_3p = line.strip().split()

            # After unpacking each line, ensure that all variables are of their desired type
            len_5p = int(len_5p)
            len_3p = int(len_3p)

            # Enforce size limit to the sum of the segment lengths to ensure data remains biologically relevant
            if max_segment_len < (len_5p + len_3p) or (len_5p + len_3p) < min_segment_len:
                continue
            if RNA_ID not in seg_dict:
                seg_dict[RNA_ID] = [(seg_id, (end_5p, start_3p), seq_5p.upper(), seq_3p.upper(), bp)]
            else:
                seg_dict[RNA_ID].append((seg_id, (end_5p, start_3p), seq_5p.upper(), seq_3p.upper(), bp))

        return seg_dict


def read_hairpin_file(hairpin_file, min_loop_len, max_loop_len):
    """
    This subroutine will take in a tab delimited file full of the loop portion of RNA hairpins, and will extract all the
    critical information about them necessary to reconstruct the hairpins from the sequences.

    Arguments:
    Hainpin_file -- A tab delimited type file containing information about the loop portion of RNA hairpins
    min_loop_len -- An integer representing the minimum allowable hairpin length
    max_loop_len -- An integer representing the maximum allowable hairpin length
    """
    loop_dict = {}

    # Step through your file, grabbing up crucial information on segments
    with open(hairpin_file) as file_in:
        for line in file_in:
            if len(line.strip().split()) != 8:
                continue
            RNA_ID, hp_id, hp_start, hp_end, hp_seq, hp_len, closing_pair, is_pk = line.strip().split()  # is_pk means is psuedoknot
            hp_len = int(hp_len)

            # Enforce size limit to the sum of the loop lengths to ensure data remains biologically relevant
            if max_loop_len < hp_len or hp_len < min_loop_len:
                continue
            if RNA_ID not in loop_dict:
                loop_dict[RNA_ID] = [(hp_id, (hp_start, hp_end), hp_seq.upper())]
            else:
                loop_dict[RNA_ID].append((hp_id, (hp_start, hp_end), hp_seq.upper()))

        return loop_dict


def build_sloople_list(seg_dict, loop_dict):
    """
    Search through each RNA_ID, pairing all loops and segments with compatible
    segment pairs to generate sloops. This will only make matches of loops
    and segments that are of the same RNA_ID.
    """
    sloople_list = []

    for RNA_ID in loop_dict:
        if RNA_ID not in seg_dict:
            continue
        loop_list = loop_dict[RNA_ID]

        for seg_info in seg_dict[RNA_ID]:
            # empty out all your information on the current segment
            seg_id, seg_bounds, seq_5p, seq_3p, bp = seg_info
            seg_end_5p, seg_start_3p = seg_bounds

            for loop_info in loop_list:
                # empty out all your information on the current hairpin
                hp_id, hp_bounds,  hp_seq = loop_info
                hp_start, hp_end = hp_bounds

                # if the bounds that you extracted align, this means they correspond to a valid sloop
                if (int(seg_end_5p) + 1) == int(hp_start) and int(hp_end) == (int(seg_start_3p) - 1):
                    # Preserve all information about the origin of the sloop in a new informative ID
                    sloop_id = '{}.{}.{}'.format(RNA_ID, seg_id, hp_id)
                    sloop_seq = seq_5p + hp_seq + seq_3p
                    sloople_list.append((sloop_id, sloop_seq))

    return sloople_list


def filter_sloople_list(raw_sloople_list, allow_N=False):
    """
    This is a subroutine designed to take in a raw sloople list and subject it to preliminary filtering to make sure
    that there are no aberrant nucleotides that got through.

    This is ONLY preliminary filtering, you will also have to process your data to remove duplicates, etc..

    Arguments:
    raw_sloople_list -- A list full of tuples that contain the following information : (sloop_ID, sloop_sequence)
    allow_N -- A boolean value that indicates whether N is a permissible base for the dataset, it is suggested that you
               leave this as the default false so that it is compatible with the Deep_Sloop Model.
    """
    filtered_sloople_list = []

    # Define the allowable set of nucleotides based on your user's preferences
    if allow_N:
        nuc_set = set('AUCGN')
    else:
        nuc_set = set('AUCG')

    counter1 = 0
    counter2 = 0
    sloop_dict = {}

    for sloople in raw_sloople_list:
        tainted = False

        counter1 += 1
        if counter1 % 1000 == 0:
            print('We have checked {} sloops for tainted nucleotides'.format(counter1))

        sloop_id, sloop_seq = sloople
        for nuc in sloop_seq:
            if nuc not in nuc_set:
                print('We have encountered a tainted datapoint: ', nuc)
                tainted = True
                break
        if not tainted:
            sloop_dict[sloop_seq] = sloop_id

    for sloop_seq, sloop_id in sloop_dict.items():
        counter2 += 1
        if counter2 % 1000 == 0:
            print('We have added a total of {} unique sloops to your filtered dataset'.format(counter2))
        temp_sloop = (sloop_id, sloop_seq)
        filtered_sloople_list.append(temp_sloop)

    print('We have checked {} sloops for tainted nucleotides'.format(counter1))
    print('We have added a total of {} unique sloops to your filtered dataset'.format(counter2))
    return filtered_sloople_list

#
# Main
#


def generate_dataset(data_dir=r"../Data",
                     seg_file="allSegments_Filtered.txt",
                     hp_file="allHairpins_Filtered.txt",
                     min_segment_len=20,
                     max_segment_len=150,
                     min_loop_len=3,
                     max_loop_len=22):
    """
    This subroutine takes in the original raw segment and raw hairpin files, and uses them to construct a dataset
    of RNA stem loops by matching RNA_IDs. This dataset is then subjected to a preliminary filtering process wherein a
    minimum and maximum total segment length and loop lengths are imposed on the data.
    """

    os.chdir(data_dir)

    seg_dict = read_segment_file(seg_file, min_segment_len, max_segment_len)
    loop_dict = read_hairpin_file(hp_file, min_loop_len, max_loop_len)
    raw_sloople_list = build_sloople_list(seg_dict, loop_dict)
    filtered_sloople_list = filter_sloople_list(raw_sloople_list)

    file_ext_num = 0
    file_unique = False
    out_file = "Raw_Sloops_Loop_{}_{}_Seg_{}_{}_ext{}.fasta".format(min_loop_len, max_loop_len, min_segment_len, min_segment_len, file_ext_num)

    while not file_unique:
        if os.path.exists(os.path.join(data_dir, out_file)):
            file_ext_num += 1
        else:
            file_unique = True

    with open(outfile, 'w') as f:
        for sloople in filtered_sloople_list:
            sloop_id, sloop_seq = sloople
            f.write('>{}\n'.format(sloop_id))
            f.write(sloop_seq + '\n')
