# TO BLUR AN IMAGE (helps to remove the noice existing in the image)
blur=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)#ksize or kernal size should be a odd num , to increase the blur you can increase the ksize value ex(7,7)
cv.imshow("blur_image",blur)  
