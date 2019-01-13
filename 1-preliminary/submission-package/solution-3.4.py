#!/usr/bin/env python3
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    # Main code goes here

    # load h5
    f = h5.File('filtered.h5', 'r')
    arr = np.array(f.get('image'))

    # find the 95% threshold value
    vals = np.sort(arr.flatten())
    lwr95 = int(len(vals) * 0.95)
    cutoff = vals[lwr95]

    # threshold image
    thresh = arr
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < cutoff:
                thresh[i][j] = 0

    # show image
    plt.imshow(thresh, cmap=cm.Greys_r)
    plt.show()

    exit(0)
