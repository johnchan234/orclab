import numpy as np
import cv2
import time
import Utils
import os.path

from os import listdir
from os.path import isfile, join
import imgaug as ia
import imageio
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
import matplotlib.pyplot as plt



cardW = 53
cardH = 78
cornerXmin = 2
cornerXmax = 10.5
cornerYmin = 2.5
cornerYmax = 23

# We convert the measures from mm to pixels: multiply by an arbitrary factor 'zoom'
# You shouldn't need to change this
zoom = 4
cardW *= zoom
cardH *= zoom
cornerXmin = int(cornerXmin*zoom)
cornerXmax = int(cornerXmax*zoom)
cornerYmin = int(cornerYmin*zoom)
cornerYmax = int(cornerYmax*zoom)


baseImg = cv2.imread('./image/videoImg/2.jpg')
card = Utils.getCardFomImg(baseImg)

Utils.saveCard(card, "./image/videoCard/", "2.jpg")


topLeftKey,bottomRightKey = Utils.find8PointOfCorner(card, needW=40, needH=70)
#topL = Utils.toKeyPoint(topLeftKey)
#bottomR = Utils.toKeyPoint(bottomRightKey)


bbs = ia.BoundingBoxesOnImage([
    Utils.pointToBoundingBox(topLeftKey),
    Utils.pointToBoundingBox(bottomRightKey),
], shape=card.shape)


seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        scale=(0.5, 0.7),
        rotate=(-80,90)
    )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

# Augment BBs and images.
image_aug, bbs_aug = seq(image=card, bounding_boxes=bbs)

for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )


image_before = bbs.draw_on_image(card, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
ia.imshow(np.hstack([image_after]))



#ia.imshow(np.hstack([card, image_keypoints.draw_on_image(card, size=7)]))

#aug = iaa.Affine(rotate=(-40, 40))
#aug_det = aug.to_deterministic()


# -----------------------------------LOOK


# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
