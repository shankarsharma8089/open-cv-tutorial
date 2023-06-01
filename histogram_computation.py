# HISTOGRAM COMPUTATION 
# (histogram allow you to visualize the distribution of pixel intensities in an image)

# histogram for gray scale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

gray_hist = cv.calcHist([gray],[0],None,[256],[0,256]) # in calchist function images are list,histsize is the num of bins that we wnat to use for computting the histogram
plt.figure()
plt.title('grayscale histogram')
plt.xlabel('bins')
plt.ylabel('num of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

 
#color histogram
plt.figure()
plt.title('color histogram')
plt.xlabel('bins')
plt.ylabel('num of pixels')

color = ('b','g','r')
for i ,col in enumerate(color):
    hist = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])

plt.show()
