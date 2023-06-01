                                                                                 #COLOUR SPACES

import matplotlib.pyplot as plt
img= cv.imread('kanye.jpg')
cv.imshow('kanye_original',img)

plt.imshow(img)
plt.show()

                                                                                # BGR TO HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('bgr to hsv',hsv)

                                                                                 #BGR TO L*A*B
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('bgr to lab',lab)

                                                                                     # BGR TO RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('bgr to rgb',rgb)
plt.imshow(rgb)
plt.show()
#you cannot convert grey scale image to HSV
#to do that you have to convert greyscale to bgr and then to hsv

                                                          # COLOR CHANNELS (we can split the image into its respective channel)
b,g,r = cv.split(img)
cv.imshow('blue',b)
cv.imshow('green',g)
cv.imshow('red',r)

print(img.shape)  # in output (1200,1200,3) here 3 represent 3 channels
print(b.shape)    #here and below there are only one channel
print(g.shape)
print(r.shape)



                                                                                  # to merge all the channels
merged = cv.merge([b,g,r])
cv.imshow('merged image',merged)

                                                                                 # to get individuals channels
import numpy as np

blank = np.zeros(img.shape[:2],dtype='uint8')
b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('blue',blue)
cv.imshow('green',green)
cv.imshow('red',red)
