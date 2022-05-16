import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

#To execute .\color.py -i NOMBREIMAGEN

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"])

cv2.imshow('image', image)