TRANSLATION  (shifting the image in x and y axis)

def translate(img,x,y):        # x and y stands for num of pixels you want to shift along the x and y axis
    transMAT = np.float32([[1,0,x],[0,1,y]])    # to translate a image we want to create a matrix 
    dimension = (img.shape[1],img.shape[0])  # image.shape[1] = width ,image.shape[0]=height
    return cv.warpAffine( img,transMAT,dimension) 

# if you have
#negative value of x you are translating the image to left  -x -->left
#                                                           -y-->up
#                                                            x-->right
#                                                            y-->down

translated = translate(img,100,100)#shifting can be done by changing the values from positive to neg ex (img,-100,100)shifts the image to left
cv.imshow('translated',translated)
