import numpy as np
import cv2
import matplotlib.pyplot as plt

def IrisLocalization(image):
    #Estimating pupil center and pupil radius
    #Convert image to gray scale for easy calculation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Define a structuring element (a 3x3 ellipse) and perform morphological opening to reduce noise and enhance the image
    #ksize = 3 achieves the highest accuracy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # Compute pupil center using binarization
    hist, _ = np.histogram(image.ravel(), 256, [0, 256])
    #Some eyes have difficulty in finding the coarse pupil center not to deviate too much from actual center
    #This makes cropping image into smaller region not applicable for some of eye images.
    #Due to this, we will binarize the image directly. In order to set the threshold which can sperate pupil from the image
    #We adjust threshold based on each image's highest frequency in the histogram.
    #However, since we didn't subset the image in the begining, Sometimes highest frequency tend to belong to some other pixel values(e.g 255)
    #Therefore, we set a constraint below so the threshold has to be under 100
    max_freq_index = np.argmax(hist)
    if max_freq_index > 100:
        hist[max_freq_index] = 0
        threshold = np.argmax(hist)
    else:
        threshold = np.argmax(hist)
    #Binarize the image
    _, binary_crop = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    #Convert the binary_crop to a binary image(0,1) for simplicity
    binary_mask = (binary_crop == 0).astype(np.uint8)
    #Estimate the center of pupil using center of mass
    moments = cv2.moments(binary_mask)
    X_p, Y_p = int(moments['m01'] / moments['m00']), int(moments['m10'] / moments['m00'])

    #Compute pupil radius
    pupil_area = cv2.moments(binary_mask)['m00']
    pupil_radius = np.sqrt(pupil_area / np.pi)

    #Pupil Cricle

    #Create cropped image to reduce region for edge detection and Hough transform based on pupil center estimates
    cropped_image = image[max(0, X_p - 120): X_p + 120, max(0, Y_p - 125): Y_p + 125]
    #Use median filter to reduce noise in the cropped image
    #We tested optimal kernal size to be 9
    median_filtered = cv2.medianBlur(cropped_image, ksize=9)
    #Use canny edge detection on filtered image
    #We choose low threshold as 10 since iris boundaries are generally not obvious enough like pupil
    #We set a low value to detect weaker and faint edges but not low enough to bring in more noises
    #As for high threshold, we also didn't set it too high to avoid missing essential edges

    edges = cv2.Canny(median_filtered, 10, 30, L2gradient=True)
    #Detect pupil circle
    #Pupil circle are easier to detect than iris circle, so we can get the iris circle directly from edge map
    #Set maxRadius based on calculated pupil radius and we add 2 for adjustment
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=255,
                               param1=10, param2=10, minRadius=15, maxRadius=round(pupil_radius) + 2)
    if circles is not None and len(circles) == 1:
        circles = np.uint16(np.around(circles))
        x, y, radius = circles[0][0]
        pupil_circle = [x, y, radius]
        #cv2.circle(cropped_image, (x, y), radius, (0, 255, 0), 1)
    else:
        print("Pupil circle is not found")

    #Iris Circle

    #Apply sobel vertical edge detection filter to edge map in order to keep only vertical edges from the edge map.
    #This helps us detecting iris circle since the limbus part are nearly vertical.
    sobel_vertical = cv2.Sobel(edges, cv2.CV_8U, 1, 0, ksize=3)
    #Create a mask to mask out pupil, area above pupil and area under pupil to further reduce noise in the edge map
    mask = np.ones_like(cropped_image) * 255
    mask[:, 125 - (round(pupil_radius) + 20):125 + (round(pupil_radius) + 20)] = 0
    masked_edges = cv2.bitwise_and(sobel_vertical, mask)
    #Detect Iris Circle
    circles = cv2.HoughCircles(masked_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=250, param1=10, param2=10, minRadius=50,
                               maxRadius=120)
    if circles is not None and len(circles) == 1:
        circles = np.uint16(np.around(circles))
        x, y, radius = circles[0][0]
        iris_circle = [x, y, radius]
        #cv2.circle(cropped_image, (x, y), radius, (0, 255, 0), 1)
    else:
        print("Iris circle is not found")

    return cropped_image, pupil_circle, iris_circle

