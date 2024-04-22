import numpy as np
import cv2
import os
import re
import time
import matplotlib.pyplot as plt
from IrisLocalization  import *
from IrisNormalization import *
from IrisEnhancement import *
from IrisFeatureExtraction import *
from IrisPerformanceEvaluation import *
import warnings


# read all images and store all the vectors of images apply preprocessing into train_array,test_array
def read_imgs():
    start_time = time.time()
    base_dir = '/Users/yutingwang/Desktop/5293/Iris Recognition/CASIA Iris Image Database (version 1.0)'
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d in [f"{i:03d}" for i in range(1, 109)]]
    classes.sort()
    train, test = [], []
    for c in classes:
        train_directory = os.path.join(base_dir, c, '1')
        test_directory = os.path.join(base_dir, c, '2')
        if os.path.isdir(train_directory):
            files = [f for f in os.listdir(train_directory) if f.endswith('.bmp')]
            files.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
            for f in files:
                train_img_path = os.path.join(train_directory, f)
                print(f)
                image = cv2.imread(train_img_path)
                vector = FeatureExtraction(image)
                train.append(vector)

        if os.path.isdir(test_directory):
            files = [f for f in os.listdir(test_directory) if f.endswith('.bmp')]
            files.sort(key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])
            for f in files:
                test_img_path = os.path.join(test_directory, f)
                print(f)
                image = cv2.imread(test_img_path)
                vector = FeatureExtraction(image)
                test.append(vector)
    train_array = np.array(train)
    test_array = np.array(test)
    return train_array,test_array


def main():
    train_vector,test_vector = read_imgs()
    # Initialize train_label and test_label for later use
    train_labels = np.repeat(range(108), 3)
    test_labels = np.repeat(range(108), 4)
    reduced_train, reduced_test = identification_evaluation(train_vector, test_vector, train_labels, test_labels)
    verification_evaluation(reduced_train, reduced_test, train_labels, test_labels)

if __name__ =='__main__':
    warnings.filterwarnings("ignore")
    main()

