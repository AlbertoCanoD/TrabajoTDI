import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\prewitt.py -i IMAGEPATH

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

# X axis Prewitt mask
gx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
prewittx = cv2.filter2D(image, -1, gx)

# Y axis Prewitt mask
gy = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
prewitty = cv2.filter2D(image, -1, gy)

# X & Y axis Prewitt mask
imagex = cv2.filter2D(image, -1, gx)
imagey = cv2.filter2D(image, -1, gy)
prewittxy = cv2.addWeighted(imagex, 0.5, imagey, 0.5, 0)

# Original
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Prewitt mask X axis
plt.subplot(2, 2, 2)
plt.imshow(prewittxy)
plt.title('Filtro realce Prewitt eje X')
plt.xticks([]), plt.yticks([])

# Prewitt mask Y axis
plt.subplot(2, 2, 3)
plt.imshow(prewittx)
plt.title('Filtro realce Prewitt eje Y')
plt.xticks([]), plt.yticks([])

# Prewitt mask X & Y axis
plt.subplot(2, 2, 4)
plt.imshow(prewitty)
plt.title('Filtro realce Prewitt eje X e Y')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
