import numpy as np
import cv2
import matplotlib.pyplot as plt
import import_ipynb
from IrisLocalization import IrisLocalization
def IrisNormalization(image):
    #Retrieve center and radius of iris and pupil circle using IrisLocalization function
    cropped_image = IrisLocalization(image)[0]
    pupil_circle = IrisLocalization(image)[1]
    iris_circle = IrisLocalization(image)[2]
    #Define fixed size of normalized image
    height = 64
    width = 512
    #Initiate normalized image with fixed size
    normalized_img = np.zeros((height, width))
    #Iris Normalization
    #Iterate through coordinates of normalized image to input normalized pixels
    for normalized_x in range(64):
        for normalized_y in range(512):
            theta = 2*np.pi*normalized_y/width
            #Define pupil boundary coordinates
            pupil_boundary_coordx = pupil_circle[1]-pupil_circle[2]*np.sin(theta)
            pupil_boundary_coordy = pupil_circle[0]+pupil_circle[2]*np.cos(theta)
            #Define iris boundary coordinates
            iris_boundary_coordx = iris_circle[1]-iris_circle[2]*np.sin(theta)
            iris_boundary_coordy = iris_circle[0]+iris_circle[2]*np.cos(theta)
            #Output coordinates in the original image which we will use corresponding pixel values to fill in the normalized image
            iris_x = min(round(pupil_boundary_coordx+(iris_boundary_coordx-pupil_boundary_coordx)*normalized_x/height), cropped_image.shape[0]-1)
            iris_y = min(round(pupil_boundary_coordy+(iris_boundary_coordy-pupil_boundary_coordy)*normalized_x/height), cropped_image.shape[1]-1)
            #Input pixel values into empty normalized image
            normalized_img[normalized_x, normalized_y] = cropped_image[iris_x, iris_y]
    return normalized_img