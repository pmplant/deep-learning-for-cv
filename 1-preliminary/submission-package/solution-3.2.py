#!/usr/bin/env python3

from PIL import Image
import cv2
import numpy as np
import h5py as h5

# references
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
# http://docs.h5py.org/en/stable/quick.html

if __name__ == "__main__":
    # Main code goes here
    #img1 = np.asarray(Image.open('input.jpg'))
    img1 = cv2.imread('input.jpg')
    img2 = 255 - img1
    cv2.imshow('Figure 1', img1)
    cv2.imshow('Figure 2', img2)

    # what's happening?
    img3 = np.asarray(Image.open('input.jpg'))
    cv2.imshow('Figure 3', img3)

    print(img1[0])
    print(img3[0])

    with h5.File("output.h5", "w") as f:
        f.create_dataset("inverted image", data=img2)
        f.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
