import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\laplaciano.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Remove noise with gaussian blur
image = cv2.GaussianBlur(image, (3, 3), 0)

# Apply the laplacian mask
aux = cv2.Laplacian(image, cv2.CV_16S, ksize=3)
Laplaciano = cv2.convertScaleAbs(aux)

# Original
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Blurred image
plt.subplot(122)
plt.imshow(Laplaciano)
plt.title('Laplaciano')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
