#!/usr/bin/env python3
from sys import argv
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import matplotlib.cm as cm

if __name__ == "__main__":
    # sigma params
    k1 = int(argv[1])
    k2 = int(argv[2])

    # load h5
    f = h5.File('output.h5', 'r')
    arr = np.array(f.get('image'))

    # reduce to a single channel
    gs = arr.mean(axis=2)

    # apply DoG filter
    g1 = gaussian_filter(gs, sigma=k1, mode='reflect')
    g2 = gaussian_filter(gs, sigma=k2, mode='reflect')
    dif = g2 - g1

    # normalize image
    norm = (dif - np.amin(dif)) / np.amax(dif)

    with h5.File('filtered.h5', 'w') as f:
        f.create_dataset('image', data=norm)
        f.close()

    # show image
    plt.imshow(norm, cmap=cm.Greys_r)
    plt.show()

    exit(0)
