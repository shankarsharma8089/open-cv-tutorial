#DILATING THE IMAGE
dilated = cv.dilate(edge,(3,3),iterations=1)
cv.imshow('dilated',dilated)

#Eroding
erode = cv.erode(dilated,(3,3),iterations=1)
cv.imshow('eroded',erode)
