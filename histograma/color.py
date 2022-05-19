import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\color.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"])

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)

# Image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Imagen')
ax1.axis("off")

# RGB Histogram of image
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.xlabel("Niveles de color")
    plt.ylabel("Frecuencia relativa")
    ax2.set_title('Histograma RGB')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
