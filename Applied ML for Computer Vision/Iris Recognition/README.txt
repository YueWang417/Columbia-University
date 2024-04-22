IrisLocalization:
1. Pupil Localization: 
Converting the input image to grayscale and then creating a histogram to find the most suitable threshold for segmenting the image into binary form, targeting the darkest area which typically corresponds to the pupil. The centroid of this binary region is calculated to estimate the center of the pupil. The radius of the pupil is derived from the area of the binary region.

2. Iris Localization:
A specific region around the initial pupil estimate is cropped to concentrate the edge detection effort and reduce noise. A median filter smoothes the cropped area before applying the Canny edge detector to identify potential boundaries. A Hough Circle Transform identifies the pupil boundary within the edge-detected image. To isolate the iris, a Sobel filter emphasizes vertical edges, followed by the application of a mask that focuses on the limbus for clearer boundary detection. A final Hough Circle Transform on the masked edges precisely locates the iris boundary.


IrisNormalization:
1. Identify the iris within the image by calling the IrisLocalization

2. defines a fixed size for the normalized image (64x512 pixels), ensuring that all iris images are represented in a uniform scale. 

3. The normalized image is filled by mapping points from the original iris image (in polar coordinates) to the fixed-size rectangular grid by calculating the polar coordinates of each point within the normalized grid and then converting these into Cartesian coordinates within the original image to sample the pixel values.



IrisEnhancement:
1. It begins by segmenting images into blocks and regions.

2. For each block, calculate the average intensity to estimate the background lighting, which is then expanded and subtracted from the original image to correct for lighting disparities.

3. The image is divided into smaller regions where histogram equalization is applied to each individually.

4. re-normalizing the lighting-corrected image, ensuring that the histogram equalization can be effectively applied and return enhanced image


IrisFeactureExtraction:
1. Defined two spatial filters, modulated by Gabor-like functions, to emphasize different aspects of the iris texture. Each filter is characterized by its frequency and the space constants of its Gaussian envelope. This is performed by the defined_filter function.

2. The spatial filters are applied to the region of interest (ROI) of the normalized iris image through convolution to produce filtered images. This is achieved by the filtered_image function.

3. The feature vector is constructed by dividing the filtered images into small blocks and calculating statistical measures (mean and average absolute deviation) for each block.

Iris Matching:
1.predict_nearestcentroid: employs the NearestCentroid classifier to match each test image vector to the most similar training image vector and return the accuracy.

2.perform_fld: reduces the dimensionality of the feature vectors. We set the maximum number of components, fit the LDA model using training features and labels, and transform them into reduced space. 

3.cal_crr(labels, predicted): calculates the classification accuracy by comparing the predicted labels to the true labels of the test data.  


Iris Performance Evaluation:
1.identification_evaluation: Calculate the identification accuracy under metrics L1, L2, and cosine metric and apply FLD analysis to observe if there is an improvement in accuracy. Return the reduced training and test feature sets for verification 

2.calculate_fmr_fnmr: determines the closest matches within a score matrix against varying thresholds to label matches or non-matches. Compute the FMR = FP / (FP+ TN) and the FNMR = FN/ (FN + TP). 

3.verification_evaluation: Evaluate the classification accuracy using the nearest centroid classifier on the reduced dataset with the cosine similarity metric. We chose the reduced dataset and cosine because it has been found to yield the highest accuracy by observing table 3. The output of this function includes a table of FMR/FNMR values and an ROC curve image.

Iris Recognition:
1.read_image: return train_vector and test_vector of all the image after preprocessing 
2.main: read image into vector. Call function ‘identification_evaluation’ and ‘verification_evaluation’ to generate figures.
