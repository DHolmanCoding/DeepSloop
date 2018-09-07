"""
This script is designed to take in a fasta file with a single RNA sequence which you would like to investigate, and
scans through the sequence, identifying regions likely to be a hairpin.
"""

#
# Dependencies
#

from keras.models import load_model

import matplotlib.pyplot as plt

import DeepSloop_Utils as DSU

from pylab import *

import numpy as np

#
# Definitions
#

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """
    Original Documentation and Script modified from the following original source:
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"


    Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x

    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def plot_peaks(x, indexes):
    """
    Plot results of the peak dectection.
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, 'b', lw=1)

    if indexes.size:
        label = 'peak'
        label = label + 's' if indexes.size > 1 else label
        ax.plot(indexes, x[indexes], '+', mfc=None, mec='r', mew=2, ms=8,
                label=f'{indexes.size} {label}')
        ax.legend(loc='best', framealpha=.5, numpoints=1)

    # I went through the HOTAIR structure and manually entered the positions of intra-strand hair pins in D1
    manual_barcode_idx = np.array([n for n in range(1, (len(x) + 1))])
    manual_barcode_vals = np.array([0.2 for _ in range(len(x))])

    manual_sloop_idx = np.array([1 for _ in range(len(x))])
    manual_sloop_bounds = [(33, 63),
                           (70, 94),
                           (123, 173),
                           (187, 217),
                           (295, 321),
                           (328, 394),
                           (438, 457),
                           (468, 522)]

    for bounds in manual_sloop_bounds:
        for idx in list(range(bounds[0], bounds[1] + 1)):
            manual_sloop_idx[(idx - 1)] = 5

    for i in range(len(x) - 1):
        plot(manual_barcode_idx[i:i + 2], manual_barcode_vals[i:i + 2], 'k', linewidth=manual_sloop_idx[i])

    sum_sloop_len = 0
    ct = 0
    for bounds in manual_sloop_bounds:
        sum_sloop_len += (bounds[1] - bounds[0])
        ct += 1

    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel('Position along RNA Sequence (base)')
    ax.set_ylabel('Average Hairpin Presence Score')
    ax.set_title('HOTAIR Domain 1 Hairpin Detection Test')
    plt.show()


#
# Main
#

# Hairpin_Finder_D1.py will only take the first entry of your fasta file as input
fasta_file = r"../Data/HOTAIR_Domain1.fasta"
weights_path = "../Model_Results/run_2018-09-06_15-47-17_res_conv_blstm128_blstm128_de64_de16_de1_5_0.146_0.943_0.946.hdf5"

# Read in file to obtain RNA sequence and its length
sloop_dict, max_sloop_len, num_sloops = DSU.fasta_to_dict(fasta_file)
DNA_ID, DNA_sequence = list(sloop_dict.items())[0]
DNA_sequence = DNA_sequence.upper()
RNA_sequence = DNA_sequence.replace('T', 'U')
seq_len = len(RNA_sequence)

window_list = [20, 24, 28, 32, 36, 40, 44, 48]  # Over 40 is a useless window
plotting_index = [n for n in range(1, (seq_len + 1))]

master_score_bins = [0 for n in range(seq_len)]
master_ct_bins = [0 for n in range(seq_len)]

model = load_model(weights_path)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

step_size = 1

# Go through your sequence using a sliding window and use DeepSloop to provide a estimation of the presence of a sloop
for window_size in window_list:
    temp_max_bins = [0 for n in range(seq_len)]
    temp_avg_bins = [0 for n in range(seq_len)]
    temp_avg_norm_bins = [0 for n in range(seq_len)]
    temp_ct_bins = [0 for n in range(seq_len)]
    if window_size < seq_len:
        start = 0
        stop = window_size
        while stop <= seq_len:
            sub_seq = RNA_sequence[start:stop]

            # First Average the 5p and 3p padded sloops to eliminate any possible artifacts
            ohe_sloop_3pad = DSU.sloop_to_ohe(DSU.pad_sloop(sub_seq, 166))
            ohe_sloop_3pad = np.expand_dims(ohe_sloop_3pad, axis=0)
            yhat_3pad = 0.9999999 * float(model.predict(ohe_sloop_3pad, verbose=0)[0][0])

            ohe_sloop_5pad = DSU.sloop_to_ohe(DSU.pad_sloop(sub_seq, 166, True))
            ohe_sloop_5pad = np.expand_dims(ohe_sloop_5pad, axis=0)
            yhat_5pad = 0.9999999 * float(model.predict(ohe_sloop_5pad, verbose=0)[0][0])

            yhat = ((yhat_3pad + yhat_5pad) / 2)

            for i in range(start, stop):
                if yhat > temp_max_bins[i]:
                    temp_max_bins[i] = yhat
                temp_avg_bins[i] += yhat
                temp_ct_bins[i] += 1

            start += step_size
            stop += step_size

    # Average your temporary score bins and then merge them to the master bins
    assert len(temp_avg_bins) == len(temp_max_bins)

    for i in range(len(temp_avg_bins)):
        temp_avg_bins[i] = temp_avg_bins[i] / temp_ct_bins[i]

    max_avg = max(temp_avg_bins)
    min_avg = min(temp_avg_bins)
    print(max_avg)
    print(min_avg)

    for i in range(len(temp_avg_bins)):
        if i == (len(temp_avg_bins) - 1):
            print(f'Temp max bins for {window_size}:', temp_max_bins)
            print(f'Temp average bins for {window_size}:', temp_avg_bins, '\n')
        temp_avg_norm_bins[i] = (temp_avg_bins[i] - min_avg) / (max_avg - min_avg)

        max_score = temp_max_bins[i]

        score = temp_avg_norm_bins[i] * max_score

        master_score_bins[i] += score
        master_ct_bins[i] += 1

# Average your master score bins
for i in range(len(master_score_bins)):
    n_score = master_score_bins[i] / master_ct_bins[i]
    master_score_bins[i] = n_score

peak_idx = detect_peaks(master_score_bins, mph=0.3, mpd=33, threshold=0, edge='rising', kpsh=False, valley=False,
                        show=False, ax=None)

indexes = np.array(peak_idx) - 1

plot_peaks(np.array(master_score_bins), indexes=np.array(peak_idx))