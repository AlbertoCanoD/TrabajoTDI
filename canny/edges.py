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
image = cv2.imread(args["image"], 1)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Convert to RGB colors
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Remove noise with gaussian blur
image = cv2.GaussianBlur(image, (3, 3), 0)

# X & Y axis Roberts mask
gx = np.array([[1, 0], [0, -1]])
robertx = cv2.filter2D(image, -1, gx)
gy = np.array([[0, 1], [-1, 0]])
roberty = cv2.filter2D(image, -1, gy)
robertxy = cv2.addWeighted(robertx, 0.5, roberty, 0.5, 0)


# Sobel mask
sobelx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
abssobelx = cv2.convertScaleAbs(sobelx)
abssobely = cv2.convertScaleAbs(sobely)
grad = cv2.addWeighted(abssobelx, 0.5, abssobely, 0.5, 0)


# X & Y axis Prewitt mask
gx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewittx = cv2.filter2D(image, -1, gx)
gy = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
prewitty = cv2.filter2D(image, -1, gy)
prewittxy = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)


# Laplacian
aux = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
Laplaciano = cv2.convertScaleAbs(aux)


# Canny
minthreshold = 150
maxthreshold = 230
canny = cv2.Canny(image, minthreshold, maxthreshold)


# Original
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Roberts
plt.subplot(2, 3, 2)
plt.imshow(robertxy, cmap='gray')
plt.title('Roberts')
plt.xticks([]), plt.yticks([])

# Sobel
plt.subplot(2, 3, 3)
plt.imshow(grad, cmap='gray')
plt.title('Sobel')
plt.xticks([]), plt.yticks([])

# Prewitt
plt.subplot(2, 3, 4)
plt.imshow(prewittxy, cmap='gray')
plt.title('Prewitt')
plt.xticks([]), plt.yticks([])

# Laplacian
plt.subplot(2, 3, 5)
plt.imshow(Laplaciano, cmap='gray')
plt.title('Laplaciano')
plt.xticks([]), plt.yticks([])

# Canny
plt.subplot(2, 3, 6)
plt.imshow(canny, cmap='gray')
plt.title('Canny')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
