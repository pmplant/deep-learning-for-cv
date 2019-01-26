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
from utils.regularizor import l2_grad, l2_loss


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

    loss, loss_c, pred = model_loss(W, b, x, y)
    loss += config.reg_lambda * l2_loss(W)

    return loss, loss_c, pred


def compute_grad(W, x, y, loss_c, config):
    """Computes the gradient for each module."""

    # Lazy import of propper model
    if config.model_type == "linear_svm":
        from utils.linear_svm import model_grad
    elif config.model_type == "logistic_regression":
        from utils.logistic_regression import model_grad
    else:
        raise ValueError("Wrong model type {}".format(
            config.model_type))

    dW, db = model_grad(loss_c, x, y)
    dW += config.reg_lambda * l2_grad(W)

    return dW, db


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

    # Scores for all class (N, 10)
    s_all = np.matmul(x, W) + b
    # Predections to use later
    pred = np.argmax(s_all, axis=1)

    return pred


def train(x_tr, y_tr, x_va, y_va, config):
    """Training function.

    Parameters
    ----------
    x_tr : ndarray
        Training data.

    y_tr : ndarray
        Training labels.

    x_va : ndarray
        Validation data.

    y_va : ndarray
        Validation labels.

    config : namespace
        Arguments and configurations parsed by `argparse`

    Returns
    -------
    train_res : dictionary
        Training results stored in a dictionary file. It should contain W and b
        when best validation accuracy was achieved, as well as the average
        losses per epoch during training, and the average accuracy of each
        epoch to analyze how training went.
    """

    # ----------------------------------------
    # Preprocess data

    # Report data statistic
    print("Training data before: mean {}, std {}, min {}, max {}".format(
        x_tr.mean(), x_tr.std(), x_tr.min(), x_tr.max()
    ))

    # Normalize data using the normalize function. Note that we are remembering
    # the mean and the range of training data and applying that to the
    # validation/test data later on.
    x_tr_n, x_tr_mean, x_tr_range = normalize(x_tr)
    x_va_n, _, _ = normalize(x_va, x_tr_mean, x_tr_range)
    # Always a good idea to print some debug messages
    print("Training data after: mean {}, std {}, min {}, max {}".format(
        x_tr_n.mean(), x_tr_n.std(), x_tr_n.min(), x_tr_n.max()
    ))

    # ----------------------------------------
    # Initialize parameters of the classifier
    print("Initializing...")
    num_class = 10
    # TODO (4 points): Initialize W to very small random values. e.g. random values
    # between -0.001 and 0.001. To get results identical to the one in the
    # assignment package, you should initialize with uniform random numbers
    # between this range.

    # TODO (1 point): Initialize b to zeros

    # Test on validation data
    print("Testing...")
    pred = predict(W, b, x_va_n, config)
    acc = np.mean(pred == y_va)
    print("Initial Validation Accuracy: {}%".format(acc * 100))

    batch_size = config.batch_size
    num_epoch = config.num_epoch
    num_batch = len(x_tr_n) // batch_size
    loss_epoch = []
    tr_acc_epoch = []
    va_acc_epoch = []
    W_best = None
    b_best = None
    best_acc = 0
    # For each epoch
    for idx_epoch in range(num_epoch):
        # TODO (5 points): Create a random order to go through the data. Note
        # that we pre-create the random order and then proceed through the
        # data. Recently, people often simply grab data purely randomly from
        # the entire dataset. However, we'll stick to the more traditional way.

        # For each training batch
        losses = np.zeros(num_batch)
        accs = np.zeros(num_batch)
        for idx_batch in range(num_batch):
            # TODO (5 points): Construct batch
            x_b = TODO
            y_b = TODO
            # Get loss with compute_loss
            loss_cur, temp_b, pred_b = compute_loss(W, b, x_b, y_b, config)
            # Get gradient with compute_grad
            #
            # HINT: When implementing the gradient function, it is a good idea
            # to first set your epochs to a low value e.g. 5, so that you see
            # the loss going down immediately. 100 epochs takes quite a while.
            dW, db = compute_grad(W, x_b, y_b, temp_b, config)
            # TODO (5 points): Update parameters (use `config.learning_rate`)
            W = TODO
            b = TODO
            # Record this batches result
            losses[idx_batch] = loss_cur
            accs[idx_batch] = np.mean(pred_b == y_b)

        # Report average results within this epoch
        print("Epoch {} -- Train Loss: {}".format(
            idx_epoch, np.mean(losses)))
        print("Epoch {} -- Train Accuracy: {:.2f}%".format(
            idx_epoch, np.mean(accs) * 100))

        # TODO (5 points): Test on validation data and report results
        pred = TODO
        acc = np.mean(pred == y_va)
        print("Epoch {} -- Validation Accuracy: {:.2f}%".format(
            idx_epoch, acc * 100))

        # TODO (2 points): If best validation accuracy, update W_best, b_best,
        # and best accuracy. We will only return the best W and b later.

        # Record per epoch statistics
        loss_epoch += [np.mean(losses)]
        tr_acc_epoch += [np.mean(accs)]
        va_acc_epoch += [acc]

    # TODO (3 points): Pack results as a dictionary. Remeber to pack
    # pre-processing related things here as well. We currently pack only three
    # things, but you probably want to pack more things.
    train_res = {}
    train_res["tr_acc_epoch"] = tr_acc_epoch
    train_res["va_acc_epoch"] = va_acc_epoch
    train_res["loss_epoch"] = loss_epoch
    TODO

    return train_res


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
    # Create folds
    num_fold = config.cv_num_folds

    # TODO (5 points): Randomly shuffle data and labels. IMPORANT: make sure
    # the data and label is shuffled with the same random indices so that they
    # don't get mixed up!

    # Cross validation setup. Note here that if we set cross validation to
    # False, which is the default, we'll just return results for the first
    # fold.
    if config.cross_validate:
        va_fold_to_test = np.arange(num_fold)
    else:
        va_fold_to_test = np.arange(1)

    # ----------------------------------------
    # Cross validation loop
    #
    # HINT: To first be able to implement without the cross validation in
    # place, you can first use the full training data as the traning set, and
    # the test data as the validation set, just to get the program running. To
    # do that, simply set `x_tr=x_trva` and `x_va=x_te`. Same for the y
    # labels. Once you have the framework running, you can then implement the
    # cross validation loop.
    train_res = []
    for idx_va_fold in va_fold_to_test:
        # TODO (10 points): Create train and validation data using folds. Note
        # that not all of this process needs to be here. You can also have some
        # parts of this implemented outside of the for loop.

        # ----------------------------------------
        # Train
        print("Training for fold {}...".format(idx_va_fold))
        # Run training
        cur_train_res = train(x_tr, y_tr, x_va, y_va, config)

        # Save results
        train_res += [cur_train_res]

    # Average results from all folds and display. Note that this is not
    # necesarilly the best way to display things
    tr_acc_epoch = np.mean([
        _res["tr_acc_epoch"] for _res in train_res
    ], axis=0)
    va_acc_epoch = np.mean([
        _res["va_acc_epoch"] for _res in train_res
    ], axis=0)
    loss_epoch = np.mean([
        _res["loss_epoch"] for _res in train_res
    ], axis=0)

    # Draw train and validation accuracies
    plt.figure()
    plt.plot(np.arange(config.num_epoch), tr_acc_epoch)
    plt.plot(np.arange(config.num_epoch), va_acc_epoch)
    plt.show(False)

    # Draw loss evolution
    plt.figure()
    plt.plot(np.arange(config.num_epoch), loss_epoch)
    plt.show()

    # Different final execution based on run type
    if config.cross_validate:
        # TODO (5 points): If we are cross validating, simply report the
        # average of all maximum validation accuracy. We are tuning hyper
        # parameters, and to get a model that we want to test later, we need to
        # have retrain with the best hyperparameter setup.
        val_acc = TODO
        print("Average best validation accuracy: {}%".format(
            val_acc * 100))

    else:
        assert len(train_res) == 1
        # TODO (5 points): Get its W, b, x_tr_mean, x_tr_mean
        # TODO (5 points): Test the model
        pred = TODO
        acc = np.mean(pred == y_te)
        print("Test Accuracy: {}%".format(acc * 100))


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
