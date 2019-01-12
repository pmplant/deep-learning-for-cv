#!/usr/bin/env python3
from PIL import Image
import numpy as np
import h5py as h5

if __name__ == "__main__":
    # load image
    img = Image.open('input.jpg')
    img.show()
    # invert image
    inv = Image.fromarray(255 - np.asarray(img))
    inv.show()
    # save inverted image
    with h5.File('output.h5', 'w') as f:
        f.create_dataset('inv', data=inv)
        f.close()
