import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\simple.py -i IMAGEPATH -t THRESHOLD

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-t", "--threshold", type=int, help="Threshold value")

ap.add_argument("-n", "--neighbourhood", type=int,
                help="Size of neighbourhood area")

args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"], 0)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

threshold = args["threshold"]

if threshold is None or threshold < 0 or threshold > 255:
    threshold = 127

neighbourhood = args["neighbourhood"]

if neighbourhood is None or neighbourhood < 0 or neighbourhood > 55 or neighbourhood % 2 != 1:
    print("Neighbourhood size cannot be prime number, neighbourhood = 11")
    neighbourhood = 11

ret, bin = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY, neighbourhood, 2)
gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, neighbourhood, 2)

# Original
plt.subplot(2, 2, 1)
plt.imshow(image,  'gray')
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# THRESH_BINARY
plt.subplot(2, 2, 2)
plt.imshow(bin,  'gray')
plt.title('THRESH_BINARY')
plt.xticks([]), plt.yticks([])

# THRESH_BINARY_INV
plt.subplot(2, 2, 3)
plt.imshow(mean, 'gray')
plt.title('ADAPTIVE_THRESH_MEAN_C')
plt.xticks([]), plt.yticks([])

# THRESH_TRUNC
plt.subplot(2, 2, 4)
plt.imshow(gaussian, 'gray')
plt.title('ADAPTIVE_THRESH_GAUSSIAN_C')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
