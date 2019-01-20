# features.py ---
#
# Filename: features.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 21:06:57 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import numpy as np
from skimage.color import rgb2hsv
from skimage.feature import hog


def extract_h_histogram(data):
    """Extract Hue Histograms from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    h_hist : ndarray (float)
        Hue histgram per image, extracted and reshaped to be NxD, where D is
        the number of bins of each histogram.

    """

    # TODO: Implement the method
    h_hist = np.ndarray(shape=(data.shape[0], 16))

    def hist(img):
        hue = rgb2hsv(img)[:, :, 0]
        binseq = np.linspace(min(hue), max(hue), num=16)
        return np.histogram(hue, bins=binseq, density=False)

    h_hist = np.array([hist(v) for v in data])

    # Assertion to help you check if implementation is correct
    assert h_hist.shape == (data.shape[0], 16)

    return h_hist


def extract_hog(data):
    """Extract HOG from data.

    Parameters
    ----------
    data : ndarray
        NHWC formatted input images.

    Returns
    -------
    hog_feat : ndarray (float)
        HOG features per image, extracted and reshaped to be NxD, where D is
        the dimension of HOG features.

    """

    # Using HOG
    print("Extracting HOG...")

    # TODO: Implement the method
    hog_feat = np.array([hog(v, block_norm='L1') for v in data])

    # Assertion to help you check if implementation is correct
    assert hog_feat.shape == (data.shape[0], 324)

    return hog_feat


#
# features.py ends here
