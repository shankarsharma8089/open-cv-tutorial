#MASKING (masking helps us to focus on certain parts of image)
blank = np.zeros(img.shape[:2],dtype='uint8')  #the dimension of the mask should be same as the main image
cv.imshow('blank',blank)

mask = cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,thickness=-1)
cv.imshow('circle',mask)

masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked image',masked)
