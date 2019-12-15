import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('coral2.png')
img = cv2.resize(img,None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_LINEAR)

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
K = 6
area_pixels = np.zeros(K,int)
area_total = 0
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
print(img.shape)

center = np.uint8(center)
cluster_m = np.reshape(label,(518,691))
res = center[label.flatten()]
res2 = res.reshape((img.shape))
img2 = img.copy()
channel = img2[:,:,0]
for cluster_value in range (K):
    print(cluster_value)
    img2 = img.copy()
    img2[cluster_m != cluster_value,:] = 0
    cv2.imshow('img2 cluster:' + str(cluster_value),img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img2[cluster_m == cluster_value,:] = 255
    area_pixels[cluster_value] = cv2.calcHist([img2],[0],None,[256],[0,256])[255]
    if (cluster_value != 4 and cluster_value != 2):
        area_total += area_pixels[cluster_value]
    print('area pixels: '+ str(area_pixels[cluster_value]))

img2 = img.copy()
img2[cluster_m == 4,:] = 0
img2[cluster_m == 2,:] = 0
print('Area total del coral: ' + str(area_total))

cv2.imshow('img2 cluster 0 1 3 5', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('K-means2.jpg', img2)

