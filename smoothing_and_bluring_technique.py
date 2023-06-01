#  SMOOTHING AND BLURING TECHNIQUE
# we generally smooth image when it tends to have a lot of noice(noice can be caused by camera sensor or the problem in lighting when the image was taken )
# kernal window works same as the cnn window.
#  methods of bluring

#                                                                               1) averaging 
# (when we define a kernal or window over a specific portion of an image, this window will compute the pixel intensity of the middle pixel of the true center and the average of the surrounding pixel intensities  )
# the higher the kernal size value more is the blur
average = cv.blur(img,(3,3))
cv.imshow('average blur',average)

#                                                                               2)Gaussian Blur
# (it is similar to avg blur but instead of  averaging the pixel intensity each pixel is given weight as the product of those weight gives the value for the true center )
# this method gives less blur than the avg method because weight values where added insted of  taking average of pixel
GaussianBlur = cv.GaussianBlur(img,(3,3),0)  #sigmax= standard de=eviation in the x direction 
cv.imshow('Gaussianblur',GaussianBlur)

#                                                                                3)MEDIAN BLUR 
# it find the median of the surrounding pixel (it is more effective than avg blur and guassian blur)
medianBlur = cv.medianBlur(img,3)
cv.imshow('medianblur',medianBlur)

#                                                                  4) bilateral blur (most effecitive blur method)
# this method apply bluring but retains the edges in the images which other method dont do
bilateralFilter = cv.bilateralFilter(img,5,15,15)
cv.imshow('bilateral',bilateralFilter)
