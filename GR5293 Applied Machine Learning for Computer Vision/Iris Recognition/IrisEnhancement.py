import numpy as np
import cv2
import matplotlib.pyplot as plt
import import_ipynb
from IrisNormalization import IrisNormalization


def IrisEnhancement(image):
    normalized_image = IrisNormalization(image)

    # Estimating background illumination
    # This can reduce the influence of lighting variations in the normalized image, thus enhancing iris features
    block_size = (16, 16)
    block_x = int(normalized_image.shape[0] / block_size[0])
    block_y = int(normalized_image.shape[1] / block_size[1])
    # Initialize background illumination
    background_illumination = np.zeros((block_x, block_y))
    # Iterate through coordinates of background illumination to input mean pixel values of each 16x16 block in normalized image
    for i in range(block_x):
        start_x = i * block_size[0]
        end_x = (i + 1) * block_size[0]
        for j in range(block_y):
            start_y = j * block_size[1]
            end_y = (j + 1) * block_size[1]
            block_mean = np.mean(normalized_image[start_x:end_x, start_y:end_y])
            # Coarse estimate of background illumination
            background_illumination[i, j] = block_mean
    # Expand this background illumination estimate to the same size as normalized image by bicubic interpolation
    background_illumination = cv2.resize(background_illumination, (512, 64), interpolation=cv2.INTER_CUBIC)
    normalized_image_lighting_corrected = normalized_image - background_illumination

    # Further enhance normalized image with histogram equalization
    # Histogram equalization can increase image contrast, making iris features more distinguishable
    # By applying histogram equalization in each 32x32 region, we can adjust the contrast and brightness of each block independently.
    # This mitigates the nonuniform illumination problem
    region_size = (32, 32)
    region_x = int(normalized_image_lighting_corrected.shape[0] / region_size[0])
    region_y = int(normalized_image_lighting_corrected.shape[1] / region_size[1])
    # Initialize enhancement image
    enhancement_image = np.zeros(normalized_image_lighting_corrected.shape)
    # Iterate through coordinate range of each 32x32 region to subset normalized image into parts for histogram equalization
    for i in range(region_x):
        start_x = i * region_size[0]
        end_x = (i + 1) * region_size[0]
        for j in range(region_y):
            start_y = j * region_size[1]
            end_y = (j + 1) * region_size[1]
            # After we subtract background illumination to reduce influence of lighting,
            # normalized image is no longer on the scale of 0-255. Thus we need to normalize it using min_max
            normalized_image_lighting_corrected = cv2.normalize(normalized_image_lighting_corrected, None, 255, 0,
                                                                cv2.NORM_MINMAX, cv2.CV_8U)
            # Histogram equalization
            equalization = cv2.equalizeHist(normalized_image_lighting_corrected[start_x:end_x, start_y:end_y])
            # Input equalization pixel values into enhancement image
            enhancement_image[start_x:end_x, start_y:end_y] = equalization
    return enhancement_image