#!/usr/bin/env python3
from PIL import Image
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # load image
    img = Image.open('input.jpg')

    # invert image
    inv = 255 - np.asarray(img)

    # save inverted image
    with h5.File('output.h5', 'w') as f:
        f.create_dataset('image', data=inv)
        f.close()

    # show images
    plt.imshow(img)
    plt.show()
    plt.imshow(inv)
    plt.show()

    exit(0)
