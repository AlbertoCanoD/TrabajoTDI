import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\mediana.py -i IMAGEPATH -k KERNELSIZE

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-k", "--kernel", type=int, help="Kernel size, > 3")

args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Processing kernel size,
ksize = args["kernel"]

if ksize is None or ksize % 2 != 1:
    print("Kernel size cannot be prime number, ksize = 3")
    ksize = 3

# Apply blur with ksize*ksize
mediana = cv2.medianBlur(image, ksize)

# Original
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Blurred image
plt.subplot(122)
plt.imshow(mediana)
plt.title('Filtro media con kernel ' + str(ksize) + 'x' + str(ksize))
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
