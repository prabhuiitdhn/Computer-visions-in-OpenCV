import numpy as np
import cv2
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="Path to the imahe")
args=vars(ap.parse_args())
image=cv2.imread(args["image"])
print "width:%d pixel" %(image.shape[1])
print "height:%d pixel" %(image.shape[0])
print  "Channels:%d" %(image.shape[2])

cv2.imshow("image", image)
cv2.waitKey(0)



