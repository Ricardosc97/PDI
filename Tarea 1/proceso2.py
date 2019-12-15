import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
alfha = 0.35
#filename = './Chichi/img2.jpeg' #Choose your file here
filename = './Chichi/img2.jpeg'

def rayleigh(channel, normalized):
    numPixels = channel.size
    normalized = normalized / numPixels
    channel = np.array(channel)
    new_channel = np.array(channel)
    new_channel[new_channel == 255] = 254
    x = 0
    while(x < 255):
        a = np.sqrt(2*alfha*alfha* math.log(1/(1-normalized[x])))/np.sqrt(2*alfha*alfha* math.log(1/(1-normalized[254])))
        new_channel[channel == x] = a*255
        x += 1
    return new_channel

img = cv2.imread(filename)
color = ('b','g','r')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

blue = cv2.equalizeHist(img[:,:,0])
green = cv2.equalizeHist(img[:,:,1])
red = cv2.equalizeHist(img[:,:,2])

img_eq = img.copy()
img_eq[:,:,0] = blue
img_eq[:,:,1] = green
img_eq[:,:,2] = red
img2 = img_eq.copy()

b,g,r = cv2.split(img_eq)
blue_histr = cv2.calcHist([img_eq],[0],None,[256],[0,256])
green_histr = cv2.calcHist([img_eq],[1],None,[256],[0,256])
red_histr = cv2.calcHist([img_eq],[2],None,[256],[0,256])

new_blue = rayleigh(b, blue_histr.cumsum())
new_green = rayleigh(g, green_histr.cumsum())
new_red = rayleigh(r, red_histr.cumsum())

img2[:,:,0] = new_blue
img2[:,:,1] = new_green
img2[:,:,2] = new_red


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)  
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)                #Img rayleigh
img3 = cv2.fastNlMeansDenoisingColored(img2,None,15,10,7,21)#Img Denoised

plt.subplot(241)
plt.title('Input')
plt.imshow(img)

plt.subplot(242)
plt.title('IMG Eq')
plt.imshow(img_eq)

plt.subplot(243)
plt.title('Rayleigh')
plt.imshow(img2)

plt.subplot(244)
plt.title('Denoised')
plt.imshow(img3)

plt.subplot(245)
plt.title('IMG input')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col,marker = ".")
    plt.xlim([0,256])

plt.subplot(246)
plt.title('Hist Eq')
for i,col in enumerate(color):
    histr = cv2.calcHist([img_eq],[i],None,[256],[0,256])
    plt.plot(histr,color = col,marker = ".")
    plt.xlim([0,256])

plt.subplot(247)
plt.title('Hist final')
for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],None,[256],[0,256])
    plt.plot(histr,color = col, marker = ".")
    plt.xlim([0,256])

plt.subplot(248)
plt.title('Hist final')
for i,col in enumerate(color):
    histr = cv2.calcHist([img3],[i],None,[256],[0,256])
    plt.plot(histr,color = col, marker = ".")
    plt.xlim([0,256])

plt.show()

#   GUARDAR IMAGEN
#img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
#cv2.imwrite('Output2.png',img3)