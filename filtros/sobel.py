import argparse
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-a", "--axis", required=False,
                help="1 for the horizontal axis, 2 for the vertical axis, none for X & Y")


args = vars(ap.parse_args())

# Read the image and convert to grayscale
img = cv.imread(args["image"], cv.COLOR_BGR2GRAY)
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


cv.waitKey(0)
cv.destroyAllWindows()