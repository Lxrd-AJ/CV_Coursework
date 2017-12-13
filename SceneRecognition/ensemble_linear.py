import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import paired_distances
from sklearn.externals import joblib
from functools import reduce
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

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
            img = cv2.imread( img_file )
            if img is None:
                print("Not an image " + img_file)
                continue
            img_dict[word].append(img)
            count += 1
    return (img_dict, count)

def features( image ):
    sift_extractor = cv2.xfeatures2d.SIFT_create()
    desc = sift_extractor.detectAndCompute( image,None )[1]
    #TODO: Remove later, temp hack to make it run faster
    # The clusters would have to be recalculated if this is changed
    desc = desc[:,50:60]
    return desc

def build_features_dict( img_dict ):
    features_dict = {}
    for key in img_dict.keys():
        images = img_dict[key]
        fts = np.array([features(x) for x in images])
        features_dict[key] = fts
    return features_dict

"""
returns a vector (m x 1) with each row being the cluster label of each row in image_feat(m x 128) from clusters(k x 128)
"""
def distance_cluster( image_feat, clusters ):
    X = []
    len_clusters = clusters.shape[0]
    """
    For each row, calculate its distance to each k-cluster and determine the k-cluster with the smallest distance
    and build a matrix of these k-clusters
    """
    for row in range(0, image_feat.shape[0]):
        e = image_feat[row,:]
        row_matrix = np.tile(e, (len_clusters,1))
        distances = paired_distances(row_matrix, clusters)
        X.append(np.argmin(distances))
    return X

def bag_of_words( image_feat, clusters ):
    distances = distance_cluster( image_feat, clusters )
    counts = {}
    for num in distances:
        counts[num] = counts.get(num,0) + 1
    X = np.zeros(clusters.shape[0])
    for key,value in counts.items():
        X[key] = value
    return X

def train_classifiers( labels, X, y):
    #TODO: Grid search for the SVC for choosing best parameters
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
        svm.fit( X_train, y_train)
        acc = svm.score( X_train, y_train )
        print("Train score for {:} vs Non-{:} SVM classifier; accuracy: {:.2f}%".format(label, label, acc * 100))
        classifiers.append( (label,svm) )
    return classifiers

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
        # print("{:}, prob = {:}, classes = {:}".format(
        #     res,
        #     probs,
        #     classes
        # ))
    # print("Largest probab = " + str(largest_prob))
    return prediction

if __name__ == "__main__":
    dic, count = build_dictionary("./training") 
    print("Calculating the features for " + str(count) + " images")   
    ft_dict = build_features_dict( dic )

    if os.path.exists('pt2_clusters.pkl'):
        # Load the KMeans cluster data
        print("Loading precomputed clusters ...")
        kmeans = joblib.load('pt2_clusters.pkl')
    else:  
        print("Computing clusters using kmeans ...")  
        X = []
        for images in list(ft_dict.values()):
            entry = np.concatenate(images)
            X.extend(entry)
        X = np.array(X)
        print("Input data shape = {:}".format(X.shape))   
        #TODO: Preprocess X, normalise(X)
        kmeans = KMeans(n_clusters=100, n_jobs=-1) .fit(X) #TODO: Hack should be 500 clusters
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
    
    img_paths = paths.list_images("./testing")
    # img_paths = list(img_paths)[:2]
    clusters = kmeans.cluster_centers_

    with open('./run2.txt', 'a') as f:
        for img_path in img_paths:
            label = img_path.split( os.path.sep )[2]
            # print(label) #Visual Verification only
            image = cv2.imread(img_path)
            fts = features( image )
            ft_vec = bag_of_words( fts, clusters)
            pred_label = make_prediction( ft_vec, classifiers)            
            entry = "{:} {:}\n".format(label, pred_label)
            f.write(entry)
            print(entry)
