# linear_svm.py ---
#
# Filename: linear_svm.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:45:06 2018 (-0800)
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


def model_loss(W, b, x, y):
    """Loss function.

    Parameters
    ----------
    W : ndarray
        The weight parameters of the linear classifier. D x C, where C is the
        number of classes, and D is the dimenstion of input data.

    b : ndarray
        The bias parameters of the linear classifier. C, where C is the number
        of classes.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    loss : float
        The average loss coming from this model. In the lecture slides,
        represented as \frac{1}{N}\sum_i L_i.
    """

    # Not finished yet, but close
    n = y.shape[0]
    c = b.shape[0]
    s = x@W
    s_yi = s[np.arange(n), y]  # from 'https://stackoverflow.com/a/37292206/9801788'
    s_yi = np.tile(s_yi[np.newaxis].T, (1, c))  # from 'https://stackoverflow.com/a/1582742/9801788'
    L_i = s - s_yi + 1
    L_i[np.arange(n), y] = 0
    L_i = (L_i > 0) * L_i
    loss = (1 / n) * np.sum(L_i)

    return loss


#
# linear_svm.py ends here
