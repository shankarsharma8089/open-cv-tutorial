# EDGE CASCADE ( it helps to find the edges in the image)
edge=cv.Canny(img,125,175)  #to reduce the num of edges instead of passing the img you can pass blur
cv.imshow('canny edges',edge)
