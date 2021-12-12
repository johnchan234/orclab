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
import matplotlib.pyplot as plt
import matplotlib.patches as patches



imgPath = "./img/"


def changebg(src, output):
   
    img = cv2.imread(imgPath+src)
    print(img.shape)
    b_c,g_c,r_c = cv2.split(img)
    a_c = np.ones(b_c.shape,dtype=b_c.dtype) * 255


    #a_c[20:200,0:200,] = 0
    newImg = cv2.merge([b_c, g_c, r_c, a_c])
    cv2.imwrite(imgPath+output, newImg)


changebg("poker_1.jpg","test.png")

img3 = cv2.imread(imgPath+"test.png", cv2.IMREAD_UNCHANGED)
print(img3.shape)

imgW = 800 
imgH =  800
img2=np.zeros((imgH,imgW,4),dtype=np.uint8)
print("img3 shape :",img3.shape)
sX = 0
sY = 2
cardW = 406
cardH = 607
img2[sY:sY+cardH,sX:sX+cardW,:] =img3 



seq = iaa.Sequential([
    iaa.Affine(scale=[0.65]),
    iaa.Affine(rotate=(-180, 180)),
    iaa.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
])


pointW = 20
pointH = 20
kps = ia.KeypointsOnImage([
    ia.Keypoint(x=0, y=0),
    ia.Keypoint(x=pointW, y=0),
    ia.Keypoint(x=0, y=pointH),
    ia.Keypoint(x=pointW, y=pointH)
], shape=img2.shape)


image_aug,kps_aug = seq(image=img2,keypoints=kps)

for i in range(len(kps.keypoints)):
    before = kps.keypoints[i]
    after = kps_aug.keypoints[i]
    print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
        i, before.x, before.y, after.x, after.y)
    )

# image with keypoints before/after augmentation (shown below)
#image_after = kps_aug.draw_on_image(image_aug, size=7)


mask1=image_aug[:,:,3] #alpha層 img1 除了有牌的部分其他的全為0，包括alpha層
mask1=np.stack([mask1]*3,-1)# 堆三個alpha層出來w*h*channel(3)

bg = cv2.imread(imgPath+"bg.jpeg")
scaleBg = iaa.Resize({"height": imgW, "width": imgH})
bg = scaleBg.augment_image(bg)

final=np.where(mask1,image_aug[:,:,0:3],bg)


cv2.imwrite(imgPath+"finallahihi.jpg", final)


fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(final)


pointW = 20
pointH = 30
newPoint = kps_aug.keypoints[0]

rect=patches.Rectangle((newPoint.x, newPoint.y),pointW,pointH,linewidth=1,edgecolor='b',facecolor='none')
ax.add_patch(rect)# 在ax裏面畫出來
plt.show()


##img2[decalY:decalY+cardH,decalX:decalX+cardW,:]=img1
##img2,self.lkps1,self.bbs1=augment(self.img1,[cardKP,kpsa1,kpsb1],transform_1card)


