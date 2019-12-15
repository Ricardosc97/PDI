
import numpy as np
import cv2
import math
import argparse

size = 4
value = 1
ratio = 3
filename = 'coral3.png'
kernel = np.ones((5,5), np.uint8)
# The order of the colors is blue, green, red
#lower_color_bounds = (0, 40, 40)
#upper_color_bounds = (205,245,245)
#
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
iterations = 10


def img_show(title,src):
    cv2.imshow(title,src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def thresh_rgb(src,lower_color_bounds,upper_color_bounds):
    mask = cv2.inRange(src,lower_color_bounds,upper_color_bounds )
    mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    output = src & mask_rgb
    return output

def find_corals(src, iterations):
    upper_color_bounds = (245,255,255)
    for i in range (0,iterations):
        lower_color_bounds = (0 , 120 - i*7,120 - i*7) 
        #thresh color
        img_thresh_rgb = thresh_rgb(src,lower_color_bounds,upper_color_bounds)
        #
        #img_show('tresh BGR',img_thresh_rgb)
        #Gray Scale 
        gray = cv2.cvtColor(img_thresh_rgb,cv2.COLOR_BGR2GRAY)
        #
        #img_show('gray scale', gray)
        #Thresh Balck & white
        _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
        #
        #img_show('Thresh', ~thresh)
        #Temp
        #img3 = ~thresh 
        
        #Open
        element_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*3+1, 2*3+1), (3, 3))
        img_closed = cv2.morphologyEx(~thresh, cv2.MORPH_CLOSE, element_close)
        #
        #img_show('Close', img_closed)
        
        #Close 
        element_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*5+1, 2*5+1), (5, 5))
        img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, element_open)
        #img_show('Opened', img_opened)
        

        #Ero 
        if ( i < 16):
            element_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(16 - i)+1, 2*(16 - i)+1), (16 - i, 16 - i))
            img_erosion = cv2.erode(~img_opened, element_ero, iterations=1)
        else:
            img_erosion = ~img_opened
            print('pajuo')
        #img_show('Erotion', ~img_erosion)

        temp = cv2.cvtColor(~img_erosion,cv2.COLOR_GRAY2BGR)
        img_new = temp & src
        #img_show('NEWWW'+ str(i), img_new)
        src = img_new
    return img_opened, src




img = cv2.imread(filename)
img = cv2.resize(img,None, fx = 0.2, fy = 0.2, interpolation = cv2.INTER_LINEAR)

blur = cv2.GaussianBlur(img,(11,11),0)
img_show('Gaussian', blur)
area, corals = find_corals(img,iterations)
img_show('Corals',corals)
img_show('Area',area)
cv2.imwrite('Morfologic2.jpg', corals)
kernel_size = 1
img_blur = cv2.blur(area, (3,3))
detected_edges = cv2.Canny(img_blur, value, value*ratio, kernel_size)

img_show('Edges', detected_edges)


print(detected_edges.shape)
edge_lenght = cv2.calcHist([detected_edges],[0],None,[256],[0,256])[255]
area_pixels = cv2.calcHist([area],[0],None,[256],[0,256])[255]
print('Edge lenght: ' + str(edge_lenght))
print('Total Area: ' + str(area_pixels))



im2, contours, hierarchy = cv2.findContours(area,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if (area[cY][cX] == 255):
        cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # display the image
        cv2.imshow("Image", img)
        cv2.waitKey(0)

