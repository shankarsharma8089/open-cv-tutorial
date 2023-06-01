import cv2 as cv

                                                            #to read an image
img = cv.imread('1010687.jpg') #to read the image

#to read a image with large dimensions we write like this
img = cv.imread('image with higher dimension name_large.jpg') 

cv.imshow('name for the iamge pop-up',img)  #to show the image ,this method display the image as a new window, so the 2 parameter which are passed into are (name of the window, the matrixx of pixel to display)

cv.waitKey(0) # it is a keyboard binding function , it waits for a specfic delay or a time in miliseconds or a key to be pressed, passing 0 means it wait for an infinite amount of time for a key to be pressed
