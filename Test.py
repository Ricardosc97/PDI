import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('rose.jpg')
dst = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()