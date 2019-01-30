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

    Also computes the individual per-class losses to be used for gradient
    computation, as well as the predicted labels.

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
        represented as \sum_i L_i.

    temp : ndarray

        Temporary values to be used later that are calulated alonside when we
        compute the loss. It turns out much of the computation that we do on
        our ``forward'' pass can be used for the ``backward'' pass. In case of
        multinomial logistic regression, this is the per-class probabilities.

    pred : ndarray
        Predictions from the model. N numbers where each number corresponds to
        a class.

    """

    # Scores for all class (N, 10)
    s_all = np.matmul(x, W) + b
    # Predections to use later
    pred = np.argmax(s_all, axis=1)
    # Do exponential and normalize to get probs
    probs = np.exp(s_all - s_all.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    # For the cross entropy case, we will return the probs in temp

    # Compute loss
    loss = np.mean(-np.log(probs[np.arange(len(y)), y]))

    return loss, probs, pred


def model_grad(loss_c, x, y):
    """Gradient.

    Parameters
    ----------
    loss_c : ndarray
        The individual losses per sample and class, or any other value that
        needs to be reused when computing gradients.

    x : ndarray
        Input data that we want to predic the labels of. NxD, where D is the
        dimension of the input data.

    y : ndarray
        Ground truth labels associated with each sample. N numbers where each
        number corresponds to a class.

    Returns
    -------
    dW : ndarray
        Gradient associated with W. Should be the same shape (DxC) as W.

    db : ndarray
        Gradient associated with b. Should be the same shape (C) as b.

    """

    # loss_c : (N, C)
    # x : (N, D)
    # y : (N, )


    # TODO (20 points): Implment the gradients
    N, C = loss_c.shape
    D = x.shape[1]

    # mask of dirac vectors, 1 where j = y_i, 0 otherwise
    mask = np.zeros((N, C))
    mask[np.arange(N), y] = 1

    # subtract dirac mask from losses, multiply by samples
    dW = np.reshape(x, (N, D, 1)) * np.reshape(loss_c - mask, (N, 1, C))
    # take the mean gradient weight of every sample
    dW = np.mean(dW, axis=0)

    # subtract dirac mask from losses, take the mean of every gradient bias
    db = np.mean(loss_c - mask, axis=0)

    return dW, db


#
# logistic_regression.py ends here
