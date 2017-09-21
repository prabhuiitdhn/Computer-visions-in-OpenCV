import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", help="path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Original", image)

flipped =cv2.flip(image, 1)# it is for horizotally, 1 means horizonatalllily flipped
cv2.imshow("flipped horizonatally", flipped)

flipped=cv2.flip(image, 0) # it is for vertically
cv2.imshow("flipped vertically", flipped)

flipped=cv2.flip(image, -1) #-1 is for both horizontal and vertical
cv2.imshow("Flipped both", flipped)
cv2.waitKey(0)