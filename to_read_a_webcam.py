import cv2 as cv
import numpy as np
framWidth = 640
frameHeight = 480
cap = cv.VideoCapture(0)
cap.set(3,framWidth)
cap.set(4,frameHeight)
cap.set(10,130)  #setting brightness
while True:
    success , img = cap.read()
    print(type(success))
    cv.imshow("results", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
