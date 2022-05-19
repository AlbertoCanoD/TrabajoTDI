from re import X
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\roberts.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-a", "--axis", required=False,
                help="1 for the horizontal axis, 2 for the vertical axis, none for X & Y")


args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

axisarg = args["axis"]

def axis(axisarg):
    if axisarg == 1:
        gx = np.array([[1, 0], [0, -1]])
        robert = cv2.filter2D(image, -1, gx)
        return robert

    elif axisarg == 2:
        gy = np.array([[0, 1], [-1, 0]])
        robert = cv2.filter2D(image, -1, gy)
        return robert

    else:
        gx = np.array([[1, 0], [0, -1]])
        gy = np.array([[0, 1], [-1, 0]])
        imagex = cv2.filter2D(image, -1, gx)
        imagey = cv2.filter2D(image, -1, gy)
        robert = cv2.addWeighted(imagex, 0.5, imagey, 0.5, 0)
        return robert

robert = axis('axisarg')
'''
# Mask Gx
gx = np.array([[1, 0],
              [0, -1]])

# Mask Gy
gy = np.array([[0, 1],
              [-1, 0]])

# Apply filter to image
imagex = cv2.filter2D(image, -1, gx)
imagey = cv2.filter2D(image, -1, gy)

# Calculate gradient
grad = cv2.addWeighted(imagex, 0.5, imagey, 0.5, 0)'''


# Original
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Filtered image
plt.subplot(122)
plt.imshow(robert)
plt.title('Filtro realce Roberts')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
