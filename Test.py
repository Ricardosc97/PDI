import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('CITE.jpg')
img2 = cv.resize(img1, None, fx = 1, fy = .9, interpolation = cv.INTER_CUBIC)
cv.imshow("input",img1)
cv.imshow("output", img2)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('CITE2.jpg',img2)

#plt.subplot(121)
#plt.title('input')
#plt.imshow(img1)   
#plt.subplot(122)
#plt.title('output')
#plt.imshow(img2)
#plt.show()

