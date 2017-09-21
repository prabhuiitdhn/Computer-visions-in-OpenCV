import argparse
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="Path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("original", image)

'''
M=np.float32([1, 0, 25],[0, 1, 50])
shifted=cv2.warpAffine(image, M, ([image.shape[1], image.shape[0]]))
cv2.imshow("Shifted down and right", shifted)

M=np.flaot32([1, 0, -50],[0, 1, -90])
shifted=cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv.imshow("Shifted up and left", shifted)'''

shifted=imutils.translate(image, 0, 100)
cv2.imshow("Shifted down", shifted)
cv2.waitKey(0)
