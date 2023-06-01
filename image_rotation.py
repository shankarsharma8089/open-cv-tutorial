#ROTATION OF IMAGE
def rotate(img,angle,rotPoint=None): # this fun takes an image, angle to rotate around , rotation point
    (height,width)=img.shape[:2]     

    if rotPoint is None:
        rotPoint = (width//2,height//2) # if rotpoint is none we will rotate around center
             
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)  # rotation matrix (center ,angle ,scale value)
    dimensions = (width,height)

    return cv.warpAffine(img,rotMat,dimensions)

rotated = rotate(img,45)
cv.imshow('rotated',rotated)
