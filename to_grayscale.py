#CONVERTING TO GRAYSCALE

img= cv.imread('kanye.jpg')
cv.imshow('kanye_orignial',img)
#gray scale image show you the distributation of pixel imtensities at particular locations of the image
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY,)
cv.imshow('gray',gray) 
