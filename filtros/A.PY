# Python program to explain cv2.blur() method

# importing cv2
import cv2
from cv2 import blur
from matplotlib import pyplot as plt

# path
path = r"C:/Users/Alberto/Downloads/TrabajoTDI/filtros/test.png"

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# ksize
ksize = (10, 10)

# Using cv2.blur() method
blur = cv2.blur(image, ksize)

# Displaying the image
#cv2.imshow(window_name, image)

fig,ax = plt.subplots(1,1,figsize=(10,10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)

ax1.imshow(image)
ax2.imshow(blur)

cv2.waitKey(0)
cv2.destroyAllWindows()