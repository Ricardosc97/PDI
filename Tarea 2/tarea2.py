from __future__ import print_function
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import argparse

filename = 'prueba.png'


max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

src = cv2.imread(filename)

def find_corals(src, iterations):
    #thresh color
    img_thresh_rgb = thresh_rgb(src,lower_color_bounds,upper_color_bounds)
    #
    img_show('tresh BGR',img_thresh_rgb)
    #Gray Scale 
    gray = cv2.cvtColor(img_thresh_rgb,cv2.COLOR_BGR2GRAY)
    #
    img_show('gray scale', gray)
    #Thresh Balck & white
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
    #
    img_show('Thresh', ~thresh)
    #Temp
    img3 = ~thresh
    #Erotion
    element_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*8+1, 2*8+1), (8, 8))
    img_erosion = cv2.erode(thresh, element_ero, iterations=1)
    #
    img_show('Erotion', ~img_erosion)
    #Open
    element_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*size+1, 2*size+1), (size, size))
    img_opened = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, element_open)
    size += 1
    #

    #Close 
    element_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*size+1, 2*size+1), (size, size))
    img_closed = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, element)

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# src = cv2.resize(img2,None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_LINEAR)
# gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
# #gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# #src = img2
# print(src.size)


# cv2.imshow('gray',thresh)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 

#cv2.imshow('bilinear img',bilinear_img)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv2.imshow(window_name, dst)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
args = parser.parse_args()

if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv2.waitKey()