# Python program to explain cv2.blur() method

# importing cv2
import cv2

# path
path = r"C:/Users/Alberto/Downloads/TrabajoTDI/filtros/test.png"

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# ksize
ksize = (10, 10)

# Using cv2.blur() method
image = cv2.blur(image, ksize)

# Displaying the image
cv2.imshow(window_name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()