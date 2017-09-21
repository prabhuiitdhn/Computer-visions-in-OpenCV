from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="Path to the image")

args=vars(ap.parse_args())
image=cv2.imread(args["image"])
cv2.imshow("Original", image)
chans=cv2.split(image)
colors=("b","g","r")
plt.figure()
plt.title("Flattend color histogram")
plt.xlabel("bins")
plt.ylabel("# of pixels")
#IT IS FOR ONE-DIMENSIONAL RGB HISTOGRAM
for (chan, color) in zip(chans, colors):
	hist=cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256]) #xlim is used for limit
#FOR 2-D HISTOGRAM
fig=plt.figure()
ax=fig.add_subplot(131)# 1 is not of rows, 3 is the number of col, 1 is plot_number
hist=cv2.calcHist([chans[0],chans[1]],[0,1], None, [32, 32],[0, 256, 0,256])
p=ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color histogram of Green and Blue")
plt.colorbar(p)

ax=fig.add_subplot(132) # 1 row and 3 column, plot_number 2
hist=cv2.calcHist([chans[1], chans[2]],[0,1], None, [32, 32],[0, 256, 0, 256])
p=ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color for Blue and red")
plt.colorbar(p)

ax=fig.add_subplot(133) #1 row, 2 col, plot_number 3
hist=cv2.calcHist([chans[0], chans[2]],[0,1], None, [32, 32],[0, 256, 0, 256])
p=ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color for Blue and red")
plt.colorbar(p)

print "2D histogram shape:%s, with %d values" %(hist.shape, hist.flatten().shape[0])

#3D histogram
hist=cv2.calcHist([image],[0, 1, 2], None, [8,8,8],[0, 256, 0, 256, 0, 256])
print "3D Histogram shap:%s, with %d values" %(hist.shape, hist.flatten().shape[0])
plt.show()

