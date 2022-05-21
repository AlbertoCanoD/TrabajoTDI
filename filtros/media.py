import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\media.py -i IMAGEPATH -k KERNELSIZE

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-k", "--kernel", type=int, help="Kernel size, > 3")

args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Processing kernel size,
ksize = args["kernel"]

if ksize is None or ksize < 3:
    ksize = 3

kernel = (ksize, ksize)

# Apply blur with ksize*ksize
blur = cv2.blur(image, kernel)

# Original
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Blurred image
plt.subplot(122)
plt.imshow(blur)
plt.title('Filtro media con kernel ' + str(ksize) + 'x' + str(ksize))
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
