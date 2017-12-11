import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths
from sklearn.cluster import KMeans
from functools import reduce
from sklearn.externals import joblib

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
    return np.array(desc)

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
        distances = np.linalg.norm(row_matrix - clusters)
        X.append(np.argmin(distances))
    return X

if __name__ == "__main__":
    precomputed = True
    if precomputed:
        # Load the KMeans cluster data
        clusters = joblib.load('pt2_clusters.pkl')
    else:    
        dic, count = build_dictionary("./training")
        
        # images = dic['Forest']
        # desc = features( images[0] )
        # print( desc.shape )

        ft_dict = build_features_dict( dic )
        X = []
        for images in list(ft_dict.values()):
            entry = np.concatenate(images)
            print(entry.shape)
            X.extend(entry)
        X = np.array(X)
        print(X.shape)
        
        clusters = KMeans(n_clusters=500, n_jobs=-1).fit_predict(X)
        print(clusters.shape)
        joblib.dump( clusters, 'pt2_clusters.pkl')
