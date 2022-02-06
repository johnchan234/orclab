import numpy as np
import cv2

from os import listdir
from os.path import isfile, join
import imgaug as ia

from imgaug import augmenters as iaa
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sympy import Point, Polygon
import string


imgPath= "./image/genTrainData/images/train/42_1639590365375_JN4U5R_DJ4VBV.jpg"
label = "./image/genTrainData/label/train/42_1639590365375_JN4U5R_DJ4VBV.txt"

img = cv2.imread(imgPath)
print(img.shape)
f = open(label, "r")
txt = f.read()
val = txt.split("\n")

"""
"""

kp = []
for label in val:
    realVal = label.split(" ")
    if len(realVal)>2:
        x = float(realVal[1]) * int(img.shape[0])
        y = float(realVal[2]) * int(img.shape[0])
        print("x: ",x," y : ",y)
        kp.append(ia.Keypoint(x=x, y=y))

kps = ia.KeypointsOnImage(kp, shape=img.shape)


# Augment keypoints and images.

image_before = kps.draw_on_image(img, size=7)
cv2.imshow('tt', image_before)
cv2.waitKey(0)
cv2.destroyAllWindows()
