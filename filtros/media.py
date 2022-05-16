import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

#To execute .\media.py -i NOMBREIMAGEN -k KERNELSIZE

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Image path or name if in the same folder")

ap.add_argument("-k", "--kernel", type=int, help="Kernel size, > 3")

args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"], cv2.IMREAD_COLOR)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Processing kernel size, 
ksize = args["kernel"]

if ksize < 3 :
    kernel = (3,3)
else:
    kernel = (args["kernel"],args["kernel"])

blur = cv2.blur(image,kernel)

# Image
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Histogram of image
plt.subplot(122)
plt.imshow(blur)
if ksize < 3 :
    plt.title('Filtro media con kernel 3x3')
plt.title('Filtro media con kernel ' + str(ksize) + 'x' + str(ksize))
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()