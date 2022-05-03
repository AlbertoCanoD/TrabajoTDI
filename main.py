import imp
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

print("Introduzca el nombre de la imagen")
nombre = "coca.jpg" #input()

img = cv.imread(nombre,0)

if img is None:
    print('No se puede abrir la imagen:')
    exit(0)
cv.imshow(nombre, img)

histr = cv.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()

#plt.hist(img.ravel(),256, [0,256])
#plt.show()

equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv.imwrite('equalizada.png',res)

cv.imshow('equalizada.png',res)

cv.waitKey(0)
cv.destroyAllWindows()