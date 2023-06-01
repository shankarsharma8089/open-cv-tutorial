import numpy as np

blank = np.zeros((500,800,3), dtype='uint8')  # to make a blank image   (height,width,shape(num of colour channel))
cv.imshow('Blank',blank)

                                                                         # 1. paint a image in certain colour
blank[:]=255,0,255   #gives green colour rgb format, to color certain portion of the image ex blank[200:300,300:400]=0,255,0
#cv.imshow('purple',blank)
#cv.waitKey(0)

                                                                                  #2 .to draw a rectangle
cv.rectangle(blank,(0,0),(250,250),(0,255,0),thickness=3) # to fill the rectangle use ex thickness=cv.Filled  # insted of writing 0,0 and 250,250 we can write it as blank.shape[1]//2,blank.shape[0]//2
cv.imshow('rect',blank)
cv.waitKey(0)


                                                                                    #3. to draw a circle

cv.circle(blank,(250,250),(40),(0,0,255),thickness=2) #to fill the image we can give the thickness=-1
cv.imshow('circle',blank)
cv.waitKey(0)

                                                                                     #4. to draw the line
cv.line(blank,(2,2),(234,345),(0,0,255),thickness=3)
cv.imshow('line',blank)
cv.waitKey(0)

                                                                                      #5. to write text
cv.putText(blank,'i love my mom',(355,355),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)
cv.imshow('text',blank)
cv.waitKey(0)
