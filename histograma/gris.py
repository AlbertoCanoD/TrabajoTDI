import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image")
args = vars(ap.parse_args())

#nombre = "a.bmp"
#image = cv2.imread(nombre)

# Read the image
image = cv2.imread(args["image"])

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Grayscale histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# First we show the grayscale image
plt.figure(1)
plt.title(args["image"])
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

# Plot the histogram
plt.figure(2)
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()