# EDGE DETECTION  
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#laplacian edges
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)

#SOBEL 
sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)
combined_sobel = cv.bitwise_or(sobelx,sobely)

cv.imshow('sobel x',sobelx)
cv.imshow('sobel y',sobely)
cv.imshow('combined sobel',combined_sobel)
