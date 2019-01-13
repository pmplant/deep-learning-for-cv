#!/usr/bin/env python3
from sys import argv
from PIL import Image
from scipy import ndimage
import numpy as np
import h5py as h5


def show_arr(arr):
    Image.fromarray(arr).show()


if __name__ == "__main__":
    # params
    k1 = int(argv[1])
    k2 = int(argv[2])
    # load h5
    f = h5.File('output.h5', 'r')
    arr = f['arr'][:]
    # to single channel
    gs = np.ndarray((len(arr), len(arr[0])), dtype=np.uint8)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            val = np.array(int(np.mean(arr[i][j])))
            gs[i][j] = np.uint8(np.mean(arr[i][j]))
    show_arr(gs)
    # apply filters
    g1 = ndimage.gaussian_filter(gs, sigma=k1)
    show_arr(g1)
    g2 = ndimage.gaussian_filter(g1, sigma=k2)
    show_arr(g2)
