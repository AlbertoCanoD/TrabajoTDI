import imp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#parser = argparse.ArgumentParser(description='Code for Histogram Calculation tutorial.')
#parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
#args = parser.parse_args()

print("Introduzca el nombre de la imagen")
nombre = "a.jpg" #input()

img = cv.imread(nombre,0)

if img is None:
    print('No se puede abrir la imagen:')
    exit(0)
cv.imshow(nombre, img)

plt.hist(img.ravel(),256, [0,256])
plt.show()

equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imwrite('equalizada.png',res)

cv.imshow('equalizada.png',res)

cv.waitKey(0)
cv.destroyAllWindows()