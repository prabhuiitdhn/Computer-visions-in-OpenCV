import numpy as np
import cv2

rectangle=np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25),(275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)

circle=np.zeros((300, 300), dtype="uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("circle", circle)
'''
bitwiseAnd=cv2.bitwise_and(rectangle, circle)
cv2.imshow("And", bitwiseAnd)
cv2.waitKey(0)
'''

bitwiseOr=cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)

bitwiseXOr=cv2.bitwise_xor(rectangle, circle)
cv2.imshow("OR", bitwiseXOr)
cv2.waitKey(0)
