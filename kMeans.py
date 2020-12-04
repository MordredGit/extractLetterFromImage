import numpy as np
import matplotlib.pyplot as plt
import cv2

# %matplotlib inline

# Read in the image
image = cv2.imread('CNN/generated_data/letters/K/cross_K_24.jpg')

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.imshow(image)

cv2.imshow('Original', image)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1, 3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initally chosed for k-means clustering
k = 4
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

labels = labels.flatten()

smallest = np.bincount(labels).argmin()

print(smallest)

# convert data into 8-bit values
centers = np.uint8(centers)

print(labels, centers)

segmented_data = centers[labels]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

lower = upper = centers[smallest]
lower -= 10
upper += 10
mask = cv2.inRange(segmented_image, np.array(lower), np.array(upper))

letter = cv2.bitwise_and(segmented_image, segmented_image, mask=mask)

cv2.imshow('Window1', letter)
# thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# plt.imshow(segmented_image)

cv2.imshow('Window', segmented_image)
cv2.waitKey(0)
