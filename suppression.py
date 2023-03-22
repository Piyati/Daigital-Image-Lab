#import the neccesary packages
from utils import sliding_window,non_max_suppression
import argparse
import time
import numpy as np
import cv2

def correlation_coefficient (patchl, patch2) :
    product = np.mean((patchl-patchl.mean())*(patch2 -patch2.mean ()))
    stds = patchl.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
    return product

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="Path to the image")
ap. add_argument("-s" ,"--simage", required=True, help= "Path to the image")
args = vars(ap.parse_args())
# when run write this command: python sliding window.py --image 1.jpg --simage 2.jpg

# load the image and define the window width and height
image1 = cv2.imread(args["image"])
image2 = cv2 .imread (args [ "simage"])

img1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

# Window size
(winH, winW) = img2.shape
found = []
boundingboxes = []
color= [[0,255,0],[0,0,255],[255,0,0]]
i = 0

for (x,y,window) in sliding_window(img1, stepSize = 16, windowSize = (winW, winH)):
    #if the window does not meet our desired window size, ignore it
    if window. shape[0] != winH or window.shape[1] != winW:
        continue
    patch = window.copy()
    
    # Find correlation coefficient of window with the desired image
    maxV = correlation_coefficient(patch,img2)

    # if we have found a new maximum correlation value, then update the bookkeeping variable
    if maxV > 0.4:
        found.append(maxV)
        boundingboxes.append((x,y,x + winW,y + winH))
        print(found)
    clone = img1.copy()
    cv2. rectangle(clone, (x, y), (x + winW, y + winH), color[0], 2)
    vertical = np.vstack((img2, patch))
    cv2.imshow("Match", vertical)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time. sleep (0.025)

# close all windows
cv2.destroyAllWindows()


clone = image1.copy()
# Draw rectangle on patch
for (startx, starty, endx, endy) in boundingboxes:
    cv2.rectangle (clone, (startx, starty), (endx, endy), color[i], 2)
    i= i+1

# perform non-maximum suppression on the bounding boxes
boundingboxes = np.array(boundingboxes)
print(boundingboxes)
pick = non_max_suppression (boundingboxes, 0.5)
print(pick)


# loop over the picked bounding boxes and draw them
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(image1, (startX, startY), (endX, endY), (0, 255, 0), 2)

#cv2.imshow("Match", vertical)
cv2.imshow("BeforeSuppress",clone)
cv2.imshow("AfterSuppress", image1)

key = cv2.waitKey(0) &0xFF

if key == ord("q") :
    # close all windows
    cv2.destroyAllWindows()