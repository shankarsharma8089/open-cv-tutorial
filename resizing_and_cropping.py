#resize
resized=cv.resize(img,(500,500),interpolation=cv.INTER_AREA ) # cv.INTER_AREA method is useful if you are shrinking the image # cv.INTER_LINEAR or cv.INTER_CUBIS CAN BE USED TO ENLARGE THE IMAGE TO HIGHER DIMENSION
cv.imshow('resized',resized)

# CROPPING
cropped=img[50:200,200:400]
cv.imshow('cropped',cropped)

cv.waitKey(0)
