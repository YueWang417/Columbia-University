import numpy as np
import cv2
import matplotlib.pyplot as plt
import import_ipynb
from IrisEnhancement import IrisEnhancement
import scipy.signal


import numpy as np
import scipy.signal

def FeatureExtraction(image):
    # Define a filter function
    def defined_filter(x, y, f, sigma_x, sigma_y):
        m1 = np.cos(2 * np.pi * f * np.sqrt(x ** 2 + y ** 2))
        defined_filter = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
            (-1 / 2) * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2)) * m1
        return defined_filter

    # Define a function to apply the filter to an image
    def filtered_image(frequency, sigma_x, sigma_y):
        #Since we use relative distance as inputs in the kernal function, each filtered pixel coordinates is determined by kernal size
        #Therefore we can calculate the filtered pixel coordinates as center of the kernal
        kernel_size = (3, 3)
        kernel = np.zeros(kernel_size)
        kernel_center_coord = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))

        #Generate the kernel by applying the defined filter function
        #Iterate through kernal coordinates in order to calculate the relative distance between kernal coordinates and coordinates of filtered pixel
        #We use this relative distance as inputs for our kernal function
        for coord_x in range(kernel_size[0]):
            for coord_y in range(kernel_size[1]):
                kernel[coord_x, coord_y] = defined_filter(
                    kernel_center_coord[0] - coord_x, kernel_center_coord[1] - coord_y, frequency, sigma_x, sigma_y)

        #Convolve the image with the kernel to create a filtered image, we use same padding in order to maintain spatial relationship between pixels
        filtered_image = scipy.signal.convolve2d(roi, kernel, mode='same')
        return filtered_image

    #Extract a region of interest (ROI) from the input image
    roi = IrisEnhancement(image)[0:48, :]

    #Define filter parameters for two filtered images (Given in the paper)
    sigma_x1 = 1.5
    sigma_y1 = 3
    frequency1 = 1 / sigma_x1
    sigma_x2 = 1.5
    sigma_y2 = 4.5
    frequency2 = 1 / sigma_x2

    #Apply filtering to the ROI and generate corresponding filtered image
    filtered_image1 = filtered_image(frequency1, sigma_x1, sigma_y1)
    filtered_image2 = filtered_image(frequency2, sigma_x2, sigma_y2)
    filtered_image_list = [filtered_image1, filtered_image2]

    #For each filtered image, iterate through coordinates range to calculate features (mean and average absolute deviation)
    block_size = (8, 8)
    block_x = int(roi.shape[0] / block_size[0])
    block_y = int(roi.shape[1] / block_size[1])
    vector = []
    for filtered_img in filtered_image_list:
        for i in range(block_x):
            start_x = i * block_size[0]
            end_x = (i + 1) * block_size[0]
            for j in range(block_y):
                start_y = j * block_size[1]
                end_y = (j + 1) * block_size[1]
                mean = np.mean(np.abs(filtered_img[start_x:end_x, start_y:end_y]))
                avg_abs_deviation = np.mean(np.abs(np.abs(filtered_img[start_x:end_x, start_y:end_y]) - mean))
                #Append all the features into a vector
                vector.append(mean)
                vector.append(avg_abs_deviation)
    feature_vector = np.array(vector)
    return feature_vector