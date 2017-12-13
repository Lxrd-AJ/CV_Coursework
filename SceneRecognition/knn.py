import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

"""
Feature extracting function
Extracts features from an image and returns it which can 
be used to train a classifier
"""
def feature_vector(image, size=(16,16)):
    return cv2.resize(image, size).flatten()

"""
Calculates the histogram for a vector
"""
def histogram(vec, bin_size=8):
    hist = cv2.calcHist([vec], [0], None, [bin_size], [0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

"""
Helper function that simply returns the label or class of the image given its path
"""
def extract_label( img_path ):
    return img_path.split( os.path.sep )[2]

"""
- Train the kNN Classifier, this is achieved in the following simple steps
    - Extract feature vectors from the images
    - Create histograms from the feature vectors
    - Train a k-nearest neighbour classifier using the 
        histograms as input and the image class labels as
        targets
"""
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
    
    #TODO: Shuffle the datat
    #TODO: Parameter tuning for the kNN classifier i.e GridSearch http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit( feature_vecs, labels )
    #TODO: Perform cross validation on the dataset
    acc = model.score( feature_vecs, labels )
    print("kNN accuracy: {:.2f}%".format(acc * 100))
    
    #Making predictions
    img_paths = paths.list_images("./testing")
    
    for img_path in img_paths:
        image = cv2.imread(img_path)
        ftv = feature_vector( image )
        hist = histogram( ftv )
        label = extract_label(img_path)

        prediction = model.predict( [hist] )[0]
        with open('./run1.txt','a') as f:
            entry = "{:} {:}\n".format(label, prediction)
            f.write(entry)
            print(entry)
    
    #TODO: Generate plots for use in report
    
        