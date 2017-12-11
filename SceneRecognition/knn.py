import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

def feature_vector(image, size=(16,16)):
    return cv2.resize(image, size).flatten()

def histogram(image, bin_size=8):
    hist = cv2.calcHist([image], [0], None, [bin_size], [0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_label( img_path ):
    return img_path.split( os.path.sep )[2]

if __name__ == "__main__":
    img_paths = paths.list_images("./training")
    # img_paths = list(img_paths)[:100]
    feature_vecs = []
    labels = []
    for (i,img_path) in enumerate(img_paths):
        image = cv2.imread(img_path)
        ftv = feature_vector( image )
        hist = histogram( ftv )
        label = extract_label(img_path)

        feature_vecs.append(hist)
        labels.append(label)

        print("{:} as label {:}".format(img_path,label))
    
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit( feature_vecs, labels )
    
    #Making predictions
    img_paths = paths.list_images("./testing")
    # img_paths = list(img_paths)[:50]
    feature_vecs = []
    labels = []
    for img_path in img_paths:
        image = cv2.imread(img_path)
        ftv = feature_vector( image )
        hist = histogram( ftv )
        label = extract_label(img_path)

        feature_vecs.append(hist)
        labels.append(label)

        prediction = model.predict( [hist] )[0]
        #TODO Save to a file
        with open('./run1.txt','a') as f:
            entry = "{:} {:}\n".format(label, prediction)
            f.write(entry)
            print(entry)

    acc = model.score( feature_vecs, labels )
    print("kNN accuracy: {:.2f}%".format(acc * 100))
    
    
        