import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\canny.py -i IMAGEPATH -m (INT) -M (INT)

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-m", "--minthreshold", required=True, type=int,
                help="Minimum threshold value")

ap.add_argument("-M", "--maxthreshold", required=True, type=int,
                help="Maximum threshold value")

args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Take threshold arguments
minthreshold = args["minthreshold"]
maxthreshold = args["maxthreshold"]

if minthreshold and maxthreshold is None:
    minthreshold = 80
    maxthreshold = 120

if minthreshold < 0:
    minthreshold = 80
if minthreshold > 254:
    minthreshold = 80

if maxthreshold < 1:
    maxthreshold = 120
if maxthreshold > 255:
    maxthreshold = 120

# More blur
image = cv2.GaussianBlur(image, (3, 3), 0)

# Canny algorithm
canny = cv2.Canny(image, minthreshold, maxthreshold)

# Original
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Blurred image
plt.subplot(122)
plt.imshow(canny, cmap='gray')
plt.title('Detector de bordes Canny')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
