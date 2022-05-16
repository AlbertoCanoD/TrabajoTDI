import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

#To execute .\gris.py -i NOMBREIMAGEN

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Image path or name if in the same folder")

ap.add_argument("-k", "--kernel", help="Kernel size")

args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"], cv2.IMREAD_COLOR)

#path = r'C:\Users\Alberto\Downloads\TrabajoTDI\filtros\test.png'
#image = cv2.imread(path)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Read the kernel size
if(args["kernel"] != 0) :
    kernel = args["kernel"]
else :
    kernel = 3

#kernel = 3

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

# Image
ax1.imshow(image,cmap=plt.cm.gray)
ax1.set_title('Imagen original')
ax1.axis("off")

# Histogram of image
blur = cv2.blur(image,(kernel,kernel))
ax2.imshow(blur)
ax2.set_title('Imagen con filtro media con kernel ' + kernel + ' x ' + kernel)
ax2.axis("off")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()