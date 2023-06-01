#Image Classification
#Steps for image pre-processing:
#Step 1
#Reading Image
# importing libraries
from pathlib import Path
import glob
import pandas as pd

# reading images from path
images_dir = Path('img')
images = images_dir.glob("*.tif")

train_data = []

counter = 0
for img in images:
  counter += 1
  if counter <= 130:
    train_data.append((img,1))
  else:
    train_data.append((img,0))
 
# converting data into pandas dataframe for easy visualization 
train_data = pd.DataFrame(train_data,columns=['image','label'],index = None)


#Step 2.
#Resize image
#resizing images into 229x229 dimensions:
img = cv2.resize(img, (229,229))

#Step 3
#Data Augmentation
#Data Augmentation Techniques:

#Gray Scaling
import cv2
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# horizontal flip
img = cv2.flip(img, 0) 

# vertical flip
img = cv2.flip(img,1)

#Gaussian Blurring
from scipy import ndimage
img = ndimage.gaussian_filter(img, sigma= 5.11)

# histogram equalization function
def hist(img):
  img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
  img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
  hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
  return hist_equalization_result

import random
# function for rotation
def rotation(img):
  rows,cols = img.shape[0],img.shape[1]
  randDeg = random.randint(-180, 180)
  matrix = cv2.getRotationMatrix2D((cols/2, rows/2), randDeg, 0.70)
  rotated = cv2.warpAffine(img, matrix, (rows, cols), borderMode=cv2.BORDER_CONSTANT,borderValue=(144, 159, 162))
  return rotated     

#Translation
img = cv2.warpAffine(img, np.float32([[1, 0, 84], [0, 1, 56]]), (img.shape[0], img.shape[1]),
borderMode=cv2.BORDER_CONSTANT,borderValue=(144, 159, 162))

#                                                                                                   Image Classification Techniques
#    1. Support Vector Machines
#This is the base model/feature extractor using Convolutional Neural Network, using Keras with Tensorflow backend

model = Sequential()
model.add(Conv2D(16,(5,5),padding='valid',input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.4))
model.add(Conv2D(32,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.6))
model.add(Conv2D(64,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)
feat_train = model_feat.predict(X_train)

#Fitting of SVM as a classifier
svm = SVC(kernel='rbf')
svm.fit(feat_train,np.argmax(y_train,axis=1))
svm.score(feat_test,np.argmax(y_test,axis=1))
 
#2. Decision Trees

#Feature Extractor

model = Sequential()
model.add(Conv2D(16,(5,5),padding='valid',input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.4))
model.add(Conv2D(32,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.6))
model.add(Conv2D(64,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_2').output)
feat_train = model_feat.predict(X_train)

#Decision Tree Classifier

dt = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
dt.fit(feat_train,np.argmax(y_train,axis=1))
dt.score(feat_test,np.argmax(y_test,axis=1))

#3. K Nearest Neighbor

#Base Model/feature extractor

model = Sequential()
model.add(Conv2D(16,(5,5),padding='valid',input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.4))
model.add(Conv2D(32,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.6))
model.add(Conv2D(64,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model_feat = Model(inputs=model.input,outputs=model.get_layer('dense_2').output)
feat_train = model_feat.predict(X_train)

#KNN classifier

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(feat_train,np.argmax(y_train,axis=-1))

knn.score(feat_test,np.argmax(y_test,axis=1))

#4. Artificial Neural Networks

#ANN as feature extractor using softmax classifier

model_ann = Sequential()
model_ann.add(Dense(16, input_shape=X_train.shape[1:], activation='relu'))
model_ann.add(Dropout(0.4))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dropout(0.6))
model_ann.add(Flatten())
model_ann.add(Dense(2, activation='softmax'))

model_ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_ann.fit(X_train, y_train,epochs=100,batch_size=100)
history

#5. Convolutional Neural Networks

#CNN as feature extractor using softmax classifier

model = Sequential()
model.add(Conv2D(16,(5,5),padding='valid',input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.4))
model.add(Conv2D(32,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding = 'valid'))
model.add(Dropout(0.6))
model.add(Conv2D(64,(5,5),padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

batch_size = 100
epochs= 100

optimizer = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)

model.compile(loss = 'binary_crossentropy',optimizer = optimizer, metrics = ['accuracy',keras_metrics.precision(), keras_metrics.recall()])

history = model.fit(X_train,y_train,steps_per_epoch = int(len(X_train)/batch_size),epochs=epochs)
history 


#Implementation of Thresholding based segmentation

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
image = cv2.imread("/content/1.jpg", 0)
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
flattened_image = image.reshape((image.shape[0] * image.shape[1],))
flattened_image.shape
#PDF of image intensities
plt.figure()
sns.distplot(flattened_image, kde=True)
plt.show()
#Applying Otsu Thresholding to the image for segmentation
ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.figure(figsize=(8, 8))
plt.imshow(thresh1, cmap="binary")
plt.axis("off")
plt.show()

# Implementation of Edge-based segmentation
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from skimage import filters
import skimage
warnings.filterwarnings("ignore")
image = cv2.imread("/content/1.jpg", 0)
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.show()
#Applying Sobel Edge operator
sobel_image = filters.sobel(image)
# cmap while displaying is not changed to gray for better visualisation
plt.figure(figsize=(8, 8))
plt.imshow(sobel_image)
plt.axis("off")
plt.show()
#Applying Roberts Edge operator
roberts_image = filters.roberts(image)
# cmap while displaying is not changed to gray for better visualisation
plt.figure(figsize=(8, 8))
plt.imshow(roberts_image)
plt.axis("off")
plt.show()
#Applying Prewitt edge operator
prewitt_image = filters.prewitt(image)
# cmap while displaying is not changed to gray for better visualisation
plt.figure(figsize=(8, 8))
plt.imshow(prewitt_image)
plt.axis("off")
plt.show()

