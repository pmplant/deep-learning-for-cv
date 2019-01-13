#!/usr/bin/env python3
from PIL import Image
import numpy as np
import h5py as h5


def show_arr(arr):
    Image.fromarray(arr).show()


if __name__ == "__main__":
    # load image
    img = Image.open('input.jpg')
    img.show()
    # invert image
    inv = 255 - np.asarray(img)
    show_arr(inv)
    # save inverted image
    with h5.File('output.h5', 'w') as f:
        f.create_dataset('arr', data=inv)
        f.close()
