import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import argparse
import random as rng


filename = 'image.jpeg'

def img_show(title,src):
    cv2.imshow(title,src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def plot_hist(image,title,subplot):
    color = ('b','g','r')
    plt.subplot(subplot)
    plt.title('Hist final')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col, marker = ".")
        plt.xlim([0,256])
    return


def clahe_hsv(image,channel):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    #hsv_planes[channel] = clahe.apply(hsv_planes[channel])
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv_planes[1] = clahe.apply(hsv_planes[1])
    hsv_planes[0] = clahe.apply(hsv_planes[0])
    lab = cv2.merge(hsv_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)
    if channel == 0:
        title = "Clahe in HSV"
    elif channel == 1:
        title = "Clahe in HSV"
    elif channel == 2:
        title = "Clahe in HSV"
    cv2.imshow(title, bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bgr


def clahe_lab(image,channel):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # lab_planes[2] = clahe.apply(lab_planes[2])
    # lab_planes[1] = clahe.apply(lab_planes[1])
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if channel == 0:
        title = "Clahe in L"
    elif channel == 1:
        title = "Clahe in A"
    elif channel == 2:
        title = "Clahe in B"
    cv2.imshow(title, bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bgr


img = cv2.imread(filename)

cv2.imshow('before clahe',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# img = cv2.medianBlur(img, 3)
# cv2.imshow('median blur',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##

# gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray scale',gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(gray_image)

# cv2.imshow('clahe gray scale',cl1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


clahe_hsv = clahe_hsv(img,2)
#clahe_hsv = img
# filtered = cv2.fastNlMeansDenoisingColored(clahe_v, None, 4, 4, 7, 17)
# median = cv2.medianBlur(clahe_v, 5)
# cv2.imshow('Denoise', median)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


filtered = cv2.fastNlMeansDenoisingColored(clahe_hsv, None, 4, 4, 7, 17)
gray = cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
img_show('gray2 scale', gray)

#Thresh Balck & white
_, thresh = cv2.threshold(gray,125,255,cv2.THRESH_BINARY)
#
img_show('Thresh', thresh)

median = cv2.medianBlur(thresh, 3)
img_show('Median',median)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9), (3, 3))
img_open = cv2.morphologyEx(median, cv2.MORPH_OPEN, element)
img_show('Open',img_open)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35), (11, 11))
mask_bw = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, element)
img_show('Close',mask_bw)

clahe_hsv_planes = cv2.split(clahe_hsv)
mask_planes = clahe_hsv_planes.copy()

mask_planes[0] = clahe_hsv_planes[0] & mask_bw
mask_planes[1] = clahe_hsv_planes[1] & mask_bw
mask_planes[2] = clahe_hsv_planes[2] & mask_bw

mask = cv2.merge(mask_planes)

img_show('prueba',mask)


Z = mask.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
print(mask.shape)

#K means
center = np.uint8(center)
cluster_m = np.reshape(label,(607,1080))
res = center[label.flatten()]
res2 = res.reshape((mask.shape))
#filtro
filtered = cv2.medianBlur(res2, 5)

compare = np.concatenate((mask, res2), axis=1) #side by side comparison
img_show('mask  - kmeans',compare)

compare = np.concatenate((mask, filtered), axis=1)
img_show('mask  - filtered kmeans',compare)

#Gray scale filtered
gray_mask = cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
gray_mask_split = cv2.split(mask)
gray_mask_split[0] = gray_mask
gray_mask_split[1] = gray_mask
gray_mask_split[2] = gray_mask

gray_mask = cv2.merge(gray_mask_split)
print(gray_mask.shape)
print(mask.shape)
compare = np.concatenate((mask, gray_mask), axis=1)
img_show('mask  - gray scale filtered kmeans',compare)

_, thresh = cv2.threshold(gray_mask,125,255,cv2.THRESH_BINARY)
compare = np.concatenate((mask, thresh), axis=1)
img_show('mask - Thresh Gscale Filt Kmeans', compare)

compare = np.concatenate((median, thresh[:,:,0]), axis=1)
img_show('median - thresh', compare)


element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (3, 3))
img_gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, element)
img_show('MORPH GRADIENT',img_gradient)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (3, 3))
img_close = cv2.morphologyEx(img_gradient, cv2.MORPH_CLOSE, element)
img_show('CLOSE GRADIENT',img_close)


element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11), (3, 3))
img_open= cv2.morphologyEx(img_close, cv2.MORPH_OPEN, element)

compare  = np.concatenate((img_open, thresh), axis=1)
img_show('CLOSE GRADIENT - thresh', compare)

### First merge
first_merge = mask_bw & (~thresh[:,:,0] & img_open[:,:,0])
compare  = np.concatenate((first_merge, thresh[:,:,0]), axis=1)
img_show('first_merge - thresh', compare)

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
img_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)
compare  = np.concatenate((img_close, thresh), axis=1)
img_show('thresh close - thresh', compare)

compare  = np.concatenate((img_close[:,:,0], first_merge), axis=1)
img_show('thresh close - first_merge', compare)


#Aqui busco erosionar la mascara blanco y negro para eliminar los bordes
element = cv2.getStructuringElement(cv2.MORPH_RECT, (37, 37), (17, 17))
mask_bw2 = cv2.erode(mask_bw,element)
compare  = np.concatenate((mask_bw, mask_bw2), axis=1)
img_show('mask_bw - mask_bw2', compare)

second_merge  = first_merge & (~img_close[:,:,0] & mask_bw2 ) 


second_merge_temp =  cv2.split(mask)    

second_merge_temp[0] = second_merge
second_merge_temp[1] = second_merge
second_merge_temp[2] = second_merge

second_final_merge = cv2.merge(second_merge_temp)

compare  = np.concatenate((mask,second_final_merge), axis=1)
img_show('color mask - second_final_merge', compare)

area_pixels = cv2.calcHist([second_final_merge],[0],None,[256],[0,256])[255]
print('Total Area:' + str(area_pixels))

a = 0 
thresh = cv2.split(second_final_merge)
img_show('prueba', thresh[0])
im2, contours, hierarchy = cv2.findContours(thresh[0],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)
  
    
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    a = a + 1
    cv2.circle(clahe_hsv, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(clahe_hsv, "", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  
    # display the image
    
cv2.imshow("Image", clahe_hsv)
cv2.waitKey(0)

cv2.imwrite("Output.png",clahe_hsv)
print(a)

###
# merge = img_close & median
# img_show('Merge',merge)
# img_show('Median',median)

# element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7), (3, 3))
# img_open = cv2.morphologyEx(merge, cv2.MORPH_OPEN, element)
# img_show('Open after merge',img_open)


# element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), (3, 3))
# img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, element)
# img_show('Close after merge',img_close)

# element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9), (3, 3))
# img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, element)
# img_show('Open gerardo',img_open)



###

