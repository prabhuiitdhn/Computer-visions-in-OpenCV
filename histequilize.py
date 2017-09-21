from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="path to the image")
args=vars(ap.parse_args())
image=cv2.imread(args["image"])
cv2.imshow("Original", image)
image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
eq=cv2.equalizeHist(image)
cv2.imshow("Histogram quilization", np.hstack([image, eq]))
cv2.waitKey(0)