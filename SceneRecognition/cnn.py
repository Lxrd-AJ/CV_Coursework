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
from keras.utils import plot_model
from keras.models import load_model
from sklearn.utils import shuffle

#Get training data
size = (256,256)

def read_dir( dir_name ):
    X = []
    y = []
    class_num = -1
    class_map = {}
    for (name,_,files) in os.walk(dir_name):
        if name == dir_name:
            continue
        word = name.split( os.path.sep )[-1]
        class_num += 1
        class_map[class_num] = word
        for file in files:
            img_file = os.path.join( name,file )
            img = cv2.imread( img_file, 0 )
            if img is None:
                # print("Not an image " + img_file)
                continue
            img = cv2.resize(img, size)
            X.append(img)
            y.append(class_num)
    X = np.array(X)
    y = np.array(y).reshape( (len(y),1) )
    X, y = shuffle( X, y, random_state=0) #Keras already shuffles the data when you fit
    X = X.reshape(X.shape[0],size[0],size[1],1)
    X = X / 255 
    y = np_utils.to_categorical(y,class_num+1)
    return X, y, class_map, class_num

X, y, class_map, class_num = read_dir('./training')
num_classes = class_num + 1
print(class_map)

#TODO: Convnet parameter tuning, try different kernel sizes
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
plot_model(model, to_file='cnn_arch.png')

if os.path.exists('./cnn_model.h5'):
    print("Using pre_trained model ...")
    model = load_model('./cnn_model.h5')
else:
    model.fit(X,y,batch_size=50,epochs=15,verbose=1) 
    model.save('./cnn_model.h5')


#Predictions
X_test = []
X_label = []
img_paths = paths.list_images("./testing")
for image in img_paths:
    img = cv2.imread( image, 0)
    img = cv2.resize( img, size)
    X_test.append(img)
    name = image.split( os.path.sep )[-1]
    X_label.append(name)
X_test = np.array(X_test)
print("Making predictions for " + str(X_test.shape[0]) + " images ...")
X_test = X_test.reshape(X_test.shape[0],size[0],size[1],1)
X_test = X_test / 255 
predictions = model.predict_classes(X_test)
# prediction = probs.argmax(axis=1)
with open('./run3.txt','a') as f:
    for (name,label) in zip(X_label, predictions):
        entry = "{:} {:}".format(name, class_map.get(label))
        print(entry)
        f.write( entry + "\n")
    
print("Evaluating accuracy of model on unseen data")
X_, y_, class_map, class_num = read_dir('./testing_labelled')
score = model.evaluate(X_, y_, batch_size=128)
print("Test score = " + str(score[1]))