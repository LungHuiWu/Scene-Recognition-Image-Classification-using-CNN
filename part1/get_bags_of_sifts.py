from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2
from tqdm import tqdm

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    image_feats = []
    for fn in tqdm(image_paths):
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        _, descriptors = dsift(img, step=[3,3], fast=True)
        
        dist = distance.cdist(vocab, descriptors, metric='minkowski', p=1)
        
        idx = np.argmin(dist, axis=0)
    
        # 建立 Histogram
        hist, _ = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        
        image_feats.append(hist_norm)
    
    image_feats = np.array(image_feats)
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
