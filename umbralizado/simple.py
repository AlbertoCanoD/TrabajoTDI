import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\edges.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"], 0)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Convert to RGB colors
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Remove noise with gaussian blur
bina = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
binInv = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
trunc = cv2.threshold(image, 100, 255, cv2.THRESH_TRUNC)
tozero = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)
tozeroInv = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO_INV)


# Original
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# THRESH_BINARY
plt.subplot(2, 3, 2)
plt.imshow(bina)
plt.title('Roberts')
plt.xticks([]), plt.yticks([])

# THRESH_BINARY_INV
plt.subplot(2, 3, 3)
plt.imshow(binInv, 'gray')
plt.title('Sobel')
plt.xticks([]), plt.yticks([])

# THRESH_TRUNC
plt.subplot(2, 3, 4)
plt.imshow(trunc, 'gray')
plt.title('Prewitt')
plt.xticks([]), plt.yticks([])

# THRESH_TOZERO
plt.subplot(2, 3, 5)
plt.imshow(tozero, 'gray')
plt.title('Laplaciano')
plt.xticks([]), plt.yticks([])

# THRESH_TOZERO_INV
plt.subplot(2, 3, 6)
plt.imshow(tozeroInv, 'gray')
plt.title('Canny')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
