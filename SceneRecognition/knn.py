import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit


"""
Feature extracting function
Extracts features from an image and returns it which can 
be used to train a classifier
"""
def feature_vector(image, size=(16,16)):
    img = cv2.resize(image, size)
    return img

"""
Calculates the histogram for a vector
"""
def histogram(vec, bin_size=16):
    hist = cv2.calcHist([vec], [0], None, [bin_size], [0,256])
    hist = hist.flatten()
    hist = cv2.normalize( hist , hist )

    # Invert commented and uncommented part to enable/disable histogram or zero mean&unit length; seems to decrease accuracy of test set
    #hist = vec
    #mean = cv2.mean(hist)[0]
    #hist = hist.flatten()
    #hist = hist-mean
    #mag = np.sqrt(hist.dot(hist))
    #hist = hist/mag
    return hist

"""
Helper function that simply returns the label or class of the image given its path
"""
def extract_label( img_path ):
    return img_path.split( os.path.sep )[1]

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
    feature_vecs = []
    labels = []
    for (i,img_path) in enumerate(img_paths):
        image = cv2.imread(img_path,0)
        ftv = feature_vector( image )
        hist = histogram( ftv )
        label = extract_label(img_path)

        feature_vecs.append(hist)
        labels.append(label)

        print("{:} as label {:}".format(img_path,label,))
    
    # shuffle the data using a cross-validation shuffleSplitter
    cv = ShuffleSplit(n_splits=50, test_size=0.3, random_state=None)
    
    # creating odd list of K for KNN
    myList = list(range(1,100))
    # subsetting just the odd ones
    neighbors = list(filter(lambda x: x % 2 != 0, myList))
    # empty list that will hold cv scores
    cv_scores = []
        
    # perform cross validation
    for k in neighbors:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        scores = cross_val_score(model, feature_vecs, labels, cv=cv, scoring='accuracy')
        cv_scores.append(scores.mean())
        print("Finished CV: {:.2f}%".format(k))
    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print ("The optimal number of neighbors is %d" % optimal_k)

    # plot misclassification error vs k
    plt.plot(neighbors, MSE)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()

    # model.fit( feature_vecs, labels )
    model = KNeighborsClassifier(n_neighbors=optimal_k, n_jobs=-1,algorithm='kd_tree')
    model.fit( feature_vecs, labels )
    acc = model.score(feature_vecs, labels )
    print("kNN accuracy: {:.2f}%".format(acc * 100))

    # making predictions
    img_paths = paths.list_images("./testing_labelled")

    feature_vecs = []
    labels = []
    
    for img_path in img_paths:
        image = cv2.imread(img_path,0)
        ftv = feature_vector( image )
        hist = histogram( ftv )
        label = extract_label(img_path)
        feature_vecs.append(hist)
        labels.append(label)

        # uncomment to predict and write into file; suggested after kNN accuracy > 10% for test set

        #prediction = model.predict( [hist] )[0]
        #with open('./run1.txt','a') as f:
        #    entry = "{:} {:}\n".format(label, prediction)
        #    f.write(entry)
        #    print(entry)   

    # TODO: Improve the accuracy to at least 10% for the test set
    acc = model.score(feature_vecs, labels )
    print("kNN accuracy: {:.2f}%".format(acc * 100))


        