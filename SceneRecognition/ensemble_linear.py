import cv2
import imutils 
import os 
import numpy as np 
from imutils import paths

def build_dictionary( dir_name ):
    img_list = {}
    count = 0
    for (name,_,files) in os.walk(dir_name):
        if name == dir_name:
            continue
        word = name.split( os.path.sep )[-1]
        img_list[word] = []
        for file in files:
            img_file = os.path.join( name,file )
            img = cv2.imread( img_file )
            img_list[word].append(img)
            count += 1
    return (img_list, count)

def features( image ):
    sift_extractor = cv2.xfeatures2d.SIFT_create()
    return sift_extractor.detectAndCompute( image,None )


if __name__ == "__main__":
    dic, count = build_dictionary("./training");
    # for key in dic.keys():
    #     print(key)
    images = dic['Forest']
    (kp,desc) = features( images[0] )
    print(kp)
    print(desc.shape)
    print(count)
