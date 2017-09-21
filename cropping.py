import numpy as np
import cv2
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="Path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("original", image)

cropped=image[30:120, 240:335]
cv2.imshow("T-rex Fce", cropped)
cv2.waitKey(0)