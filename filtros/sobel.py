import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

# To execute .\sobel.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"], 1)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Remove noise with gaussian blur
image = cv2.GaussianBlur(image, (3, 3), 0)

# Kernel size
ksize = 5

# Convolute kernel = 5x5
sobelx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize)  # x
sobely = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize)  # y

# Convert to uint8
abssobelx = cv2.convertScaleAbs(sobelx)
abssobely = cv2.convertScaleAbs(sobely)

# Gradient X &
grad = cv2.addWeighted(abssobelx, 0.5, abssobely, 0.5, 0)

# Original
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen original')
plt.xticks([]), plt.yticks([])

# Roberts mask X & Y axis
plt.subplot(2, 2, 2)
plt.imshow(grad, cmap='gray')
plt.title('Sobel X & Y')
plt.xticks([]), plt.yticks([])

# Roberts mask X axis
plt.subplot(2, 2, 3)
plt.imshow(abssobelx, cmap='gray')
plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

# Roberts mask Y axis
plt.subplot(2, 2, 4)
plt.imshow(abssobely, cmap='gray')
plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
