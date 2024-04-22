import numpy as np
import cv2
import os
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *


def predict_nearestcentroid(train_features, test_features,train_labels, test_labels,metric='cosine'):
    #the classifier computes the centroid for the feature vectors of each class in the training set
    clf = NearestCentroid(metric=metric)
    clf.fit(train_features, train_labels)
    # comparing the test vector to all the centroids and assigning the label of the nearest centroid
    # The closest centroid determines the predicted class for each test vector
    predicted = clf.predict(test_features)
    return cal_crr(test_labels, predicted), clf.centroids_,predicted

# apply Dimension Reduction
def perform_fld(train_features, test_features,train_labels, n_components=None):
    n_classes = len(np.unique(train_labels))
    max_components = n_classes - 1
    if n_components is None or n_components > max_components:
        n_components = max_components
    clf = LDA(n_components=n_components)
    clf.fit(train_features, train_labels)
    return clf.transform(train_features), clf.transform(test_features)

# calculate accuracy
def cal_crr(labels, predicted):
    return (labels == predicted).sum() / len(labels)
