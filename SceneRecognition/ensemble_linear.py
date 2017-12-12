import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import paired_distances
from sklearn.externals import joblib
from functools import reduce

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
                print("Empty image " + img_file)
                continue
            img_dict[word].append(img)
            count += 1
    return (img_dict, count)

def features( image ):
    sift_extractor = cv2.xfeatures2d.SIFT_create()
    desc = sift_extractor.detectAndCompute( image,None )[1]
    #TODO: Remove later, temp hack to make it run faster
    desc = desc[:,:5]
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

if __name__ == "__main__":
    precomputed = True
    dic, count = build_dictionary("./training")    
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
        kmeans = KMeans(n_clusters=50, n_jobs=-1) .fit(X) #TODO: Hack should be 500 clusters
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
    print(X)
    print("X shape = {:}".format(X.shape))
    print(y.shape)
    
    #TODO: Grid search for the SVC for choosing best parameters
    #TODO: Train one vs all classifier for each label