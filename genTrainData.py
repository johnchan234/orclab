import numpy as np
import cv2
import time
import Utils
import os.path

from os import listdir
from os.path import isfile, join
import imgaug as ia
import random
import imageio
import pickle
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
import matplotlib.pyplot as plt



#cardW = 53
#cardH = 78

imgW = 720
imgH = 720


cardW = 57
cardH = 87
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

decalX = int((imgW-cardW)/2)
decalY = int((imgH-cardH)/2)


refCard = np.array([[0, 0], [cardW, 0], [cardW, cardH],
                    [0, cardH]], dtype=np.float32)
refCardRot = np.array([[cardW, 0], [cardW, cardH], [
                      0, cardH], [0, 0]], dtype=np.float32)


"""---------------------------------------------------------------------"""
genHowMany = 5

""" Load BG"""
bg_pck_fn = "./image/pck/backgrounds.pck"
bg = Utils.loadRandomBg(bg_pck_fn, display=False)


""" Load Card"""
cards_pck_fn = "./image/pck/cards.pck"
cardFromPck, card_name, left, right = Utils.loadRandomCard(cards_pck_fn)
print("cardFromPck : ",cardFromPck.shape,"card_name : ", card_name)


""" Gen """

newSizeCard = np.zeros((imgH, imgW, 4), dtype=np.uint8)
print("tempCard shape :", newSizeCard.shape)

sX = 0
sY = 2
cardW = 200
cardH = 300
newSizeCard[sY:sY+cardH, sX:sX+cardW, :] = cardFromPck


seq = iaa.Sequential([
    iaa.Affine(scale=[0.65]),
    iaa.Affine(rotate=(-180, 180)),
    iaa.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
])

seq.to_deterministic()
pointW = 30
pointH = 40
kps = ia.KeypointsOnImage([
    ia.Keypoint(x=0, y=0),
    ia.Keypoint(x=pointW, y=0),
    ia.Keypoint(x=0, y=pointH),
    ia.Keypoint(x=pointW, y=pointH)
], shape=newSizeCard.shape)


image_aug, kps_aug = seq(image=newSizeCard, keypoints=kps)


for i in range(len(kps.keypoints)):
    before = kps.keypoints[i]
    after = kps_aug.keypoints[i]
    print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
        i, before.x, before.y, after.x, after.y)
    )

b_c, g_c, r_c, a_c = cv2.split(image_aug)
newI  = cv2.merge([b_c, g_c, r_c])
image_before = kps_aug.draw_on_image(newI, size=7)


mask1 = image_aug[:, :, 3]  # alpha層 img1 除了有牌的部分其他的全為0，包括alpha層
mask1 = np.stack([mask1]*3, -1)  # 堆三個alpha層出來w*h*channel(3)


scaleBg = iaa.Resize({"height": imgW, "width": imgH})
bg = scaleBg.augment_image(bg)

final = np.where(mask1, image_aug[:, :, 0:3], bg)

cv2.imwrite("./image/genTrainData/tt.jpg", final)

fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(image_before)

plt.show()
