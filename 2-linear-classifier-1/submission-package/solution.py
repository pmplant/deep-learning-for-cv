# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:32:47 2018 (-0800)
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


import matplotlib.pyplot as plt
import numpy as np

from config import get_config, print_usage
from utils.cifar10 import load_data
from utils.features import extract_h_histogram, extract_hog
from utils.preprocess import normalize


def compute_loss(W, b, x, y, config):
    """Computes the losses for each module."""

    # Lazy import of propper model
    if config.model_type == "linear_svm":
        from utils.linear_svm import model_loss
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_loss
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    loss = model_loss(W, b, x, y)

    return loss


def predict(W, b, x, config):
    """Predict function.

    Returns the correct class number, given the weights and biases of the
    linear classifier.

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

    config : namespace
        Arguments and configurations parsed by `argparse`

    Returns
    -------
    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # TODO: implement the function
    pred = x.dot(W) + b

    assert pred.shape == (x.shape[0], W.shape[1])

    return pred


def main(config):
    """The main function."""

    # ----------------------------------------
    # Load cifar10 train data
    print("Reading training data...")
    data_trva, y_trva = load_data(config.data_dir, "train")

    # ----------------------------------------
    # Load cifar10 test data
    print("Reading test data...")
    data_te, y_te = load_data(config.data_dir, "test")

    # ----------------------------------------
    # Extract features
    print("Extracting Features...")
    if config.feature_type == "hog":
        # HOG features
        x_trva = extract_hog(data_trva)
        x_te = extract_hog(data_te)
    elif config.feature_type == "h_histogram":
        # Hue Histogram features
        x_trva = extract_h_histogram(data_trva)
        x_te = extract_h_histogram(data_te)
    elif config.feature_type == "rgb":
        # raw RGB features
        x_trva = data_trva.astype(float).reshape(len(data_trva), -1)
        x_te = data_te.astype(float).reshape(len(data_te), -1)

    # ----------------------------------------
    # Load W, b, mean, range
    x_mean = np.load("{}_mean.npy".format(config.feature_type))
    x_range = np.load("{}_range.npy".format(config.feature_type))
    W = np.load("{}_W.npy".format(config.feature_type))
    b = np.load("{}_b.npy".format(config.feature_type))

    # Check that the newly computed mean and range is the same as the loaded
    # one
    print("Testing mean and variance")
    _, x_tr_mean, x_tr_range = normalize(x_trva)

    if not np.all(np.isclose(x_tr_mean, x_mean)) or \
       not np.all(np.isclose(x_tr_range, x_range)):
        print("Mean, range computation is not propper!")
    else:
        print("Mean, range is identical")

    # Apply normalization to the data
    x_te_n, _, _ = normalize(x_te, x_mean, x_range)

    # Predict the labels
    pred = predict(W, b, x_te_n, config)

    # Check accuracy (i.e. count how many are equal)
    acc = np.mean(pred == y_te)

    print("Test Accuracy: {}%".format(acc * 100))

    # Check losses (not necessary, helps you understand what's going on)
    loss = np.mean(compute_loss(W, b, x_te_n, y_te, config))
    print("Average Test loss: {}".format(loss))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


#
# solution.py ends here
