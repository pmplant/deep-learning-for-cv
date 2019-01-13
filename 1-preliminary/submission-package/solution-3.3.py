#!/usr/bin/env python3
from sys import argv
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import h5py as h5

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def show_arr(arr):
    Image.fromarray(arr).show()


if __name__ == "__main__":
    # params
    k1 = int(argv[1])
    k2 = int(argv[2])
    # load h5
    f = h5.File('output.h5', 'r')
    arr = f['arr'][:]
    show_arr(arr)
    # to single channel
    gs = arr.mean(axis=2)
    # apply filters
    g1 = gaussian_filter(gs, sigma=k1, mode='reflect')
    g2 = gaussian_filter(gs, sigma=k2, mode='reflect')
    dif = g2 - g1
    show_arr(dif)
    norm = dif/255.0
    print(norm[0][0])
    show_arr(norm)
    plt.imshow(norm, cmap=cm.Greys_r)
