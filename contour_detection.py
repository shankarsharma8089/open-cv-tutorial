                                                                                  # CONTOUR DETECTION 
# (contour are the boundary  of objects ,it is the line or curve that joins the countinous points along the boundary of  object,they are not the same as edges, useful in shape analysis
# ,object detection and recognition )

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

blur = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)

canny = cv.Canny(blur , 125,175)
cv.imshow('canny',canny)

contours, hierarchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) #(src,MODE,METOD)
# cv.contours method looks at the structuring elemnet or the edges in the image and return values , contours is a python list of all the coordinates 
#hierarchie is the hierarchial representation of the contours
#cv.RETRN_LIST  is the mode in which the findcontours method returns and find the contours
# cv.RETR_EXTERNAL retrives the external contours 
# cv.RETR_TREE returns all the hierarchical contours
# contour approximation method (how we want to approximate the counter)
# cv. CHAIN_APPROX_SIMPLE it compreses all the  contours that are returned
# cv.CHAIN_APPROX_NONE does nothing it jusr return all of the contours
print(f'{len(contours)} contour(s) found!')


# num of contour before bluring the image=3561
# num of contour after bluring the image with a ksize of 5 by 5 =415

# instead of using canny to find the contours we can use another function in opencv that is Threshold 
#using threshold function (threshold trys to looks at an image and try to binarize the image  so if the pixel as per the values written by me which can be changed is below 125 it is going to be set to zero ot blank and if it is above 125 it is set to white or 255)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
ret , thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)
contours, hierarchies = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

                                                                             #to draw over contours
blank = np.zeros(img.shape, dtype='uint8')  # to make a blank image   (height,width,shape(num of colour channel))
cv.imshow('Blank',blank)

cv.drawContours(blank,contours,-1,(0,0,255),thickness=2)
cv.imshow('contours_drawn',blank)

#try to use canny method first than using the threshold method
