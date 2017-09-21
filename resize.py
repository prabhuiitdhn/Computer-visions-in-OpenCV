import numpy as numpy
import argparse
import cv2
import imutils

parser=argparse.ArgumentParser()
parser.add_argument("-i","--image",required =True, help="Path to the image")
arg=vars(parser.parse_args())
image=cv2.imread(arg["image"])
cv2.imshow("original", image)
r= 150.0/ image.shape[1]
dim=(150, int(image.shape[0]*r))


resized=cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("resized(width)", resized)

r=50.0/image.shape[0]
dim=(int(image.shape[1]*r), 50)

resized=cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("resized(Height)", resized)
cv2.waitKey(0)