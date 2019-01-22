# logistic_regression.py ---
#
# Filename: logistic_regression.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jan 15 13:07:21 2018 (-0800)
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

    # TODO: implement the function
    n = y.shape[0]
    s = x@W + b
    s_yi = s[np.arange(n), y]
    c = np.max(s)
    L_i = np.exp(s_yi - c) / np.sum(np.exp(s - c), axis=1)
    loss = (-1/n) * np.sum(np.log(L_i))

    return loss


#
# logistic_regression.py ends here
