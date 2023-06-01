#                                                                             BITWISE OPERATIONS

# there are 4 basic bitwise operator
#1) AND
#2) OR
#3) XOR
#4) NOT

blank = np.zeros((400,400),dtype='uint8')
rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,thickness=-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)

cv.imshow('blank',blank)
cv.imshow('rectangle',rectangle)
cv.imshow('circle',circle)

# we will work with the above images 

#                                                                bitwise AND (gives out the intersecting region)
bitwise_and= cv.bitwise_and(rectangle,circle)
cv.imshow('bitwise and',bitwise_and)

#                                                                bitwise OR (gives out non intersecting and intersecting region)
bitwise_or = cv.bitwise_or(rectangle,circle)
cv.imshow('bitwise or',bitwise_or)

#                                                                bitwise NOT
bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('rectangle not',bitwise_not)

bitwise_not = cv.bitwise_not(rectangle)
cv.imshow('rectangle not',bitwise_not)

#                                                                bitwise XOR   (gives out only not intersecting region)
bitwise_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow('bitwise xor',bitwise_xor)
