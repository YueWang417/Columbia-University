import cv2
import os
import time
import numpy as np
import random
random.seed(42)

def get_box(img):
    # change to pixle value
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.7)
    k = 11
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # find the center that's closest to the pure red color
    red_distance = np.linalg.norm(centers - [0, 0, 255], axis=1)
    red_cluster = np.argmin(red_distance)

    # created a mask to highlight target cluster
    mask = (labels == red_cluster).reshape(img.shape[:2])
    mask = mask.astype(np.uint8)

    # refined the mask 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Tracing the outer boundary to identify the stop sign area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, x + w, y + h


if __name__ == "__main__":
    start_time = time.time()
    dir_path = './images/'
    for i in range(1, 25):
        img_name = f'stop{i}.png'
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        xmin, ymin, xmax, ymax = get_box(img)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        output_path = f'./results/{img_name}'
        cv2.imwrite(output_path, img)
    end_time = time.time()
    print(f"Running time: {end_time - start_time} seconds")