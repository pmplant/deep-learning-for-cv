# preprocess.py ---
#
# Filename: preprocess.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jan 15 10:10:03 2018 (-0800)
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


def normalize(data, data_mean=None, data_range=None):
    """Normalizes input data.

    Notice that we have two optional input arguments to this function. When
    dealing with validation/test data, we expect these to be given, since we
    should not learn or change *anything* of the trained model. This includes
    the data preprocessing step.

    In other words, if `data_mean` and `data_range` is provided, we will use
    them. Otherwise we will compute them in this function and use those values.

    Parameters
    ----------
    data : ndarray
        Input data that we want to normalize. NxD, where D is the
        dimension of the input data.

    data_mean : ndarray (optional)
        Mean of the data that we should use. 1xD is expected. If not given,
        this will be computed from `data`.

    data_range : ndarray (optional)
        Standard deviation of the data. 1xD is expected. If not given, this
        will be computed from `data`.

    Returns
    -------
    data_n : ndarray
        Normalized data. NxD, where D is the dimension of the input data.

    data_mean : ndarray
        Mean. 1xD. This will be used to apply the same normalization to the
        test data if necessary.

    data_range : ndarray
        The divisor applied to normalizing data. This is to be used later on
        val/test data as in the case as `data_mean`

    """

    # TODO: implement the function
    if data_mean is None:
        data_mean = data.mean(axis=0, keepdims=True)
        assert data_mean.shape == (1, data.shape[1])

    if data_range is None:
        data_range = data.std(axis=0, keepdims=True)
        assert data_range.shape == (1, data.shape[1])

    data_n = (data - data_mean) / data_range

    # Assertion to help you
    assert data_n.dtype == np.float
    assert data_n.shape == data.shape

    return data_n, data_mean, data_range


#
# preprocess.py ends here
