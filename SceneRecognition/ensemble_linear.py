import cv2
import imutils 
import os 
import numpy as np 
import random
from imutils import paths
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import paired_distances
from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


"""
Transforms a patch to a feature vector and then to a histogram
"""
def feature_vector(vec, bin_size):
    hist = cv2.calcHist([vec], [0], None, [bin_size], [0,bin_size^2]).flatten()
    return cv2.normalize( hist , hist )

"""
Feature extracting function:
Extracts features from an image and returns it which can 
be used to train a classifier. Selects patches of size 'p_s' every step 'step' calls 'feature_vector(patch)'
for each of the patches
"""
# TODO: change p_s back to 8 and step back to 4
def features( image ):
    # Patch size (quadratic)
    p_s = 3
    step = 50
    p_c = np.floor(p_s/2)
    img_s = np.size(image)
    desc=[]
    x=y=0
    while x < img_s- p_c:
        while y < img_s-p_c:
            patch = image[x:x+p_s, y:y+p_s]
            desc.append(feature_vector(patch,p_s))
            y+=step
        x+=step
    # Shift extractor:
    #sift_extractor = cv2.xfeatures2d.SIFT_create()
    #desc = sift_extractor.detectAndCompute( image,None )[1]

    #TODO: Remove later, temp hack to make it run faster
    # The clusters would have to be recalculated if this is changed
    #desc = desc[:,50:60]
    return desc

"""
Builds a dictionary mapping every label to its images
"""
def build_dictionary( dir_name ):
    img_dict = {}
    count = 0
    for (name,_,files) in os.walk(dir_name):
        if name == dir_name:
            continue
        word = name.split( os.path.sep )[-1]
        img_dict[word] = []
        for file in files:
            img_file = os.path.join( name,file )
            img = cv2.imread( img_file,0 )
            if img is None:
                print("Not an image " + img_file)
                continue
            img_dict[word].append(img)
            count += 1
    return (img_dict, count)

"""
Builds the visual word dictionary/vocabulary from every images' features
"""
def build_features_dict( img_dict ):
    features_dict = {}
    prog = 0
    dic_size = 0
    for key in img_dict.keys():
        dic_size+=1

    for key in img_dict.keys():
        
        images = img_dict[key]
        fts = np.array([features(x) for x in images])
        features_dict[key] = fts
        prog += 1
        print("Progress dictionary building: {:.2f}%".format((prog/dic_size)*100))
    return features_dict

"""
returns a vector (m x 1) with each row being the cluster label of each row in image_feat(m x 128) from clusters(k x 128)
"""
def distance_cluster( image_feats, clusters ):
    X = []
    len_clusters = clusters.shape[0]
    """
    For each row, calculate its distance to each k-cluster and determine the k-cluster with the smallest distance
    and build a matrix of these k-clusters
    """
    #for row in range(0, image_feat.shape[0]):
    for e in image_feats:

        #e = image_feat[row,:]
        row_matrix = np.tile(e, (len_clusters,1))
        distances = paired_distances(row_matrix, clusters)
        X.append(np.argmin(distances))
    return X

"""
Calculates the Bag-of-Visual-Words for an image, given its features and the pre-calculated vocabulary clusters
"""
def bag_of_words( image_feat, clusters ):
    distances = distance_cluster( image_feat, clusters )
    counts = {}
    for num in distances:
        counts[num] = counts.get(num,0) + 1
    X = np.zeros(clusters.shape[0])
    for key,value in counts.items():
        X[key] = value
    return X

"""
Trains each SVM given its labels together with their positive examples and the remaining lables linked connected to their BoVWs as negatives
"""
def train_classifiers( labels, X, y):
    classifiers = []
    for label in labels:
        #Split the data into 1-vs-all
        y_train = np.where( y == label, label, "Non-" + label)
        pos_x = X[np.where(y == label)]
        pos_y = y_train[np.where(y == label)]
        neg_x = X[np.where(y != label)]
        neg_y = y_train[np.where(y != label)]
        amount = pos_y.shape[0]

        #Shuffle negative y and x values and take `amount` from them
        neg_x, neg_y = shuffle( neg_x, neg_y, random_state=0)
        neg_x = neg_x[:amount,:]
        neg_y = neg_y[:amount]

        y_train = np.concatenate( (pos_y,neg_y), axis=0 )
        X_train = np.concatenate( (pos_x,neg_x), axis=0 )

        X_train, y_train = shuffle( X_train, y_train, random_state=0)
        svm = SVC(probability=True)

        #Doing some CV grid search to find an optimal C 
        # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        # clf = GridSearchCV(svm, parameters)
        # clf.fit(X_train, y_train)

        svm.fit( X_train, y_train)
        acc = svm.score( X_train, y_train )
        print("Train score for {:} vs Non-{:} SVM classifier; accuracy: {:.2f}%".format(label, label, acc * 100))
        classifiers.append( (label,svm) )
    return classifiers

"""
Predicts for each trained SVM the probability that the given BoVW from an image belongs to its (the SVMs class)
"""
def make_prediction( input_vec, classifiers ):
    input_vec = input_vec.reshape(1,-1)
    prediction = None 
    largest_prob = -1
    for (label, clf) in classifiers:
        res = clf.predict(input_vec)[0]
        probs = clf.predict_proba(input_vec)[0]
        classes = clf.classes_.tolist()
        prob = probs[classes.index(label)]
        if largest_prob < prob:
            largest_prob = prob
            prediction = label
    return prediction

# Should create a dictionary with n random features per image for each label
def random_features_perImage( img_dict):
    n=10
    rand_features_dict = {}
    for key in img_dict.keys():
        X=[]
        for image in img_dict[key]:
            feats=features(image)
            randfeats = random.shuffle(feats)
            X.append(randfeats[0:n])
        fts = np.array(X)
        rand_features_dict[key] = fts
    return rand_features_dict


"""
- Builds the features models and trains the classifiers in these steps:
    - Creates the dictionaries
    - Imports the pre-computed cluster (if available)
        - Else: Compute them with 10 random features selected from each image
    -
"""
if __name__ == "__main__":
    dic, count = build_dictionary("./training") 
    print("Calculating the features for " + str(count) + " images")   
    ft_dict = build_features_dict( dic )

    if os.path.exists('pt2_clusters.pkl'):
        # Load the KMeans cluster data
        print("Loading precomputed clusters ...")
        kmeans = joblib.load('pt2_clusters.pkl')
    else:  
        print("Building input matrix ...")  

        #TODO: Random sampling for k-Means, not all features from training images (just 10 patches)
        X = []
        prog = 0
        ft_dict_size = 0
        for images in list(ft_dict.values()):
            ft_dict_size+=1
        for images in list(ft_dict.values()):
            entry = np.concatenate(images)
            X.extend(entry)
            prog+=1
            print("Progress building matrix: {:.2f}%".format((prog/ft_dict_size)*100))

        X = np.array(X)
        print("Input data shape = {:}".format(X.shape))   
        kmeans = KMeans(n_clusters=500, n_jobs=2, verbose=True ).fit(X)
        joblib.dump( kmeans, 'pt2_clusters.pkl')

    X = []
    y = []
    for (label,images_ft) in ft_dict.items():
        for image_ft in images_ft:
            bg = bag_of_words( image_ft, kmeans.cluster_centers_)
            X.append(bg)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    
    classifiers = train_classifiers( ft_dict.keys(), X, y)
    
    #TODO change to 'testing' when finished everything else
    img_paths = paths.list_images("./testing_labelled")
    # img_paths = list(img_paths)[:2]
    clusters = kmeans.cluster_centers_

    with open('./run2.txt', 'a') as f:
        for img_path in img_paths:
            label = img_path.split( os.path.sep )[1]
            # print(label) #Visual Verification only
            image = cv2.imread(img_path,0)
            fts = features( image )
            ft_vec = bag_of_words( fts, clusters)
            pred_label = make_prediction( ft_vec, classifiers)            
            entry = "{:} {:}\n".format(label, pred_label)
            f.write(entry)
            print(entry)
