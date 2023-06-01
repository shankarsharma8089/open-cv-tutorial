                                                               # THRESHOLDING / BINARIZING IMAGES

gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

                                                                         #simple thersholding
threshold , thresh = cv.threshold(gray , 150 ,255,cv.THRESH_BINARY) #what thresh_binary function do is it looks at the image compares each pixel value to this threshold value and if it 
#is above this value (which is 150 above it sets it to 255 as 255 is the max value )and if the value fall below it sets it to zero 
#so it returns 2 things thresh (which is the binarized image) and threshold which is the same value that you passed (which is 150 as per above)
cv.imshow('SIMPLE THRESHOLDING',thresh)

threshold , thresh_inv = cv.threshold(gray , 150 ,255,cv.THRESH_BINARY_INV) # threshold_binary_inv creates a inverse binary image
cv.imshow('SIMPLE THRESHOLDING',thresh_inv)

                               # ADAPTIVE THRESHOLDING(we let the computer find the optimal threshold value by itself and using that value it binarize over the image )
adaptiveThreshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('adaptive threshold',adaptiveThreshold)
