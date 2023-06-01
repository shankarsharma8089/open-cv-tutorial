# this method works for images, video, live video
                                                                  #rescaling images
img = cv.imread('kanye.jpg')
def rescaleFrame(frame,scale=1.2):  # 
    width = int(frame.shape[1]* scale)
    height = int(frame.shape[0]* scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
recized_image = rescaleFrame(img)
cv.imshow('kanye',recized_image)
cv.waitKey(0)

                                                                    #rescaling videos
def rescaleFrame(frame,scale=2):  
    width = int(frame.shape[1]* scale)
    height = int(frame.shape[0]* scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
cv.waitKey(0)

capture = cv.VideoCapture('curry.mp4')

while True:

    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)
      
    cv.imshow('curry',frame_resized ) # # imshow method helps to display each frame of the rescized video
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break 
    
capture.release() 
cv.destroyAllWindows()  

#there is also a method for rescaling videos
#this method only work work livevideos or webcam
def changeRes(width,height):
     capture.set(3,width)
     capture.set(4,height)
