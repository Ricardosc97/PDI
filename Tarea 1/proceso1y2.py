import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

filename = './Chichi/IMG_2412.jpg'
alfha = 0.4
pMin = 0.002
pMax = 0.998
pMinHsv = 0.02
pMaxHsv = 0.98
normalize = False

def limsSupInf (img, histogram):
    numPixels = img.size / 3
    i =0
    px = 0
    limSup = 0
    limInf = 0
    pxMinHist = numPixels * pMin
    pxMaxHist = numPixels * pMax
    while (px < numPixels):
        px += histogram[i]
        if(px <= pxMinHist):
            limInf = i
        elif(px <= pxMaxHist):
            limSup = i 
        else: break
        i += 1
    return (limSup, limInf)

def stretchHistogram(channel, limSup, limInf):
    #channel = np.array(channel, int)
    newchannel = (channel - limInf)*(255/(limSup - limInf))
    newchannel = newchannel.astype(int)
    if (normalize):
        newchannel[newchannel < 0] = 0
        newchannel[newchannel > 255] = 255
    else:
        newchannel[newchannel < 0] = channel[(newchannel < 0)]   #Los valores negativos los hace cero ver npy
        newchannel[newchannel > 255] = channel[newchannel > 255] #Los valores mayores a 255 los hace 255
    return newchannel

def rayleigh(channel, normalized):
    normalized = normalized / numPixels
    channel = np.array(channel)
    new_channel = np.array(channel)
    new_channel[new_channel == 255] = 254
    x = 0
    while(x < 255):
        a = np.sqrt(2*alfha*alfha* math.log(1/(1-normalized[x])))
        new_channel[channel == x] = a*255
        x += 1
    return new_channel

    
img1 = cv2.imread(filename)
numPixels = img1.size / 3
b,g,r = cv2.split(img1)

img2 = img1.copy()
plt.subplot(131)
plt.title('Input')
plt.imshow(img1)
color = ('b','g','r')

plt.subplot(221)
plt.title('Hist1')
for i,col in enumerate(color):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

#histgram blue
histr = cv2.calcHist([img1],[0],None,[256],[0,256])
limSup, limInf = limsSupInf(img1,histr)
b = stretchHistogram(b,limSup,limInf)
#histogram green
histr = cv2.calcHist([img1],[1],None,[256],[0,256])
limSup, limInf = limsSupInf(img1,histr)
g = stretchHistogram(g,limSup,limInf)
#histogram red
histr = cv2.calcHist([img1],[2],None,[256],[0,256])
limSup, limInf = limsSupInf(img1,histr)
r = stretchHistogram(r,limSup,limInf)

#img2 = cv2.merge([r,g,b])
img2[:,:,0] = b
img2[:,:,1] = g
img2[:,:,2] = r

plt.subplot(222)
plt.title('Stretched Hist')
for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.title('Input')
plt.imshow(img1)

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.subplot(224)
plt.title('Stretched RGB')
plt.imshow(img2)
plt.show()

#=================== Estiramiento de S y V =========
#normalize = False

pMin = pMinHsv
pMax = pMaxHsv

hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
plt.subplot(231)
plt.title('HSV input')
plt.imshow(hsv)

plt.subplot(232)
plt.title('Hist S')
hist = cv2.calcHist( [hsv], [1], None, [256], [0, 256] )
plt.plot(histr)
plt.xlim([0,256])

plt.subplot(233)
plt.title('Hist V')
hist = cv2.calcHist( [hsv], [2], None, [256], [0, 256] )
plt.plot(histr)
plt.xlim([0,256])
    

#print (hsv.shape)
h,s,v = cv2.split(hsv)
#print (h.shape)
#print (h)

#histogram Saturation
histr = cv2.calcHist([hsv],[1],None,[256], [0, 256])
limSup, limInf = limsSupInf(hsv,histr)
s = stretchHistogram(s,limSup,limInf)
#histogram Value
histr = cv2.calcHist([hsv],[2],None,[256], [0, 256])
limSup, limInf = limsSupInf(hsv,histr)
v = stretchHistogram(v,limSup,limInf)

hsv[:,:,1] = s
hsv[:,:,2] = v

plt.subplot(236)
plt.title('Stretched Hist V')
hist = cv2.calcHist( [hsv], [2], None, [256], [0, 256] )
plt.plot(histr)
plt.xlim([0,256])

plt.subplot(235)
plt.title('Stretched Hist S')
hist = cv2.calcHist( [hsv], [1], None, [256], [0, 256] )
plt.plot(histr)
plt.xlim([0,256])

plt.subplot(234)
plt.title('HSV output')
plt.imshow(hsv)
plt.show()

#=============== ver final ============
imgOutput = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
plt.subplot(131)
plt.title('input')
plt.imshow(img1)

plt.subplot(132)
plt.title('Stretch RGB')
plt.imshow(img2)

plt.subplot(133)
plt.title('Final output')
plt.imshow(imgOutput)
plt.show()


#============== proceso 2 =================

img_yuv = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

b,g,r = cv2.split(img_output2)

blue_histr = cv2.calcHist([imgOutput],[0],None,[256],[0,256])
green_histr = cv2.calcHist([imgOutput],[1],None,[256],[0,256])
red_histr = cv2.calcHist([imgOutput],[2],None,[256],[0,256])

new_blue = rayleigh(b, blue_histr.cumsum())
new_green = rayleigh(g, green_histr.cumsum())
new_red = rayleigh(r, red_histr.cumsum())


img_output2[:,:,0] = new_blue
img_output2[:,:,1] = new_green
img_output2[:,:,2] = new_red

plt.subplot(131)
plt.title('input')
plt.imshow(img1)

plt.subplot(132)
plt.title('Stretch RGB')
plt.imshow(imgOutput)

plt.subplot(133)
plt.title('Final output')
plt.imshow(img_output2)
plt.show()


