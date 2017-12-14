import cv2
import imutils 
import os 
import numpy as np 
import chainer
from imutils import paths
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.utils import shuffle

#Get training data
X = []
y = []
class_num = -1
class_map = {}
size = (256,256)
for (name,_,files) in os.walk("./training"):
    if name == "./training":
        continue
    word = name.split( os.path.sep )[-1]
    class_num += 1
    class_map[word] = class_num
    for file in files:
        img_file = os.path.join( name,file )
        img = cv2.imread( img_file, 0 )
        if img is None:
            print("Not an image " + img_file)
            continue
        img = cv2.resize(img, size)
        X.append(img)
        y.append(class_num)
X = np.array(X)
y = np.array(y).reshape( (len(y),1) )
X, y = shuffle( X, y, random_state=0)
X = X.reshape(X.shape[0],size[0],size[1],1)
X = X / 255 
y = np_utils.to_categorical(y,class_num+1)

model = Sequential()
model.add(Conv2D(32,(3, 3),activation='relu',input_shape=(size[0],size[1],1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(class_num+1, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(X,y,batch_size=100,epochs=10,verbose=1)
score = model.evaluate(X, y, batch_size=128)
print("Training score = " + str(score))


#Predictions
X_test = []
img_paths = paths.list_images("./testing")
for image in img_paths:
    img = cv2.imread( image, 0)
    img = cv2.resize( img, size)
    X_test.append(img)
X_test = np.array(X_test)
X_test = X.reshape(X_test.shape[0],size[0],size[1],1)
X_test = X_test / 255 
predictions = model.predict(X_test)
print(predictions)