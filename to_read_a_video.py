capture = cv.VideoCapture('file.mp4')

while True: #this while loop reads the video frame by frame

    isTrue, frame = capture.read()
    cv.imshow('video_pop-up',frame) # imshow method helps to display each frame
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break #to break out of the while loop we say if waitkey is 20 and 0xFF==ord('d') which means if the letter d is pressed than break out of the loop and stop displaying the video
    


    
capture.release() # here we realese the capture device
cv.destroyAllWindows()     # use to destroy the window 
