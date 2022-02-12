import os
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
import matplotlib.patches as patches
from sympy import Point, Polygon
import string
import sys
from datetime import datetime
"""
clubs (♣) = 0
diamonds (♦) = 1
hearts (♥) = 2
spades (♠) = 3
"""
cardToLabel = {
    '0_a': "0",
    '0_2': "1",
    '0_3': "2",
    '0_4': "3",
    '0_5': "4",
    '0_6': "5",
    '0_7': "6",
    '0_8': "7",
    '0_9': "8",
    '0_10': "9",
    '0_j': "10",
    '0_q': "11",
    '0_k': "12",

    '1_a': "13",
    '1_2': "14",
    '1_3': "15",
    '1_4': "16",
    '1_5': "17",
    '1_6': "18",
    '1_7': "19",
    '1_8': "20",
    '1_9': "21",
    '1_10': "22",
    '1_j': "23",
    '1_q': "24",
    '1_k': "25",

    '2_a': "26",
    '2_2': "27",
    '2_3': "28",
    '2_4': "29",
    '2_5': "30",
    '2_6': "31",
    '2_7': "32",
    '2_8': "33",
    '2_9': "34",
    '2_10': "35",
    '2_j': "36",
    '2_q': "37",
    '2_k': "38",

    '3_a': "39",
    '3_2': "40",
    '3_3': "41",
    '3_4': "42",
    '3_5': "43",
    '3_6': "44",
    '3_7': "45",
    '3_8': "46",
    '3_9': "47",
    '3_10': "48",
    '3_j': "49",
    '3_q': "50",
    '3_k': "51",
}

# cardW = 53
# cardH = 78

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

if not os.path.exists("./image/genTrainData/images/train"):
    os.makedirs("./image/genTrainData/images/train")
if not os.path.exists("./image/genTrainData/images/val"):
    os.makedirs("./image/genTrainData/images/val")

if not os.path.exists("./image/genTrainData/labels/val"):
    os.makedirs("./image/genTrainData/labels/train")
if not os.path.exists("./image/genTrainData/labels/val"):
    os.makedirs("./image/genTrainData/labels/val")


"""---------------------------------------------------------------------"""
genHowMany = 5

""" Load BG"""
bg_pck_fn = "./image/pck/backgrounds.pck"
# bg_pck_fn = "./image/pck/whiteBackgrounds.pck"
bgPck = Utils.loadRandomBg(bg_pck_fn)


""" Load Card"""
cards_pck_fn = "./image/pck/cards.pck"
cardLoaded, _nb_cards_by_value = Utils.loadRandomCard(cards_pck_fn)

# print("cardFromPck : ",cardFromPck.shape,"cardName : ", cardName)
# print("left : ", left)

""" Gen """
sX = 0
sY = 2
cardW = 200
cardH = 300

seq = iaa.Sequential([
    iaa.Affine(scale=[0.65]),
    iaa.Affine(rotate=(-180, 180)),
    iaa.Affine(translate_percent={
        "x": (0, 0.25), "y":  (0, 0.25)}),
])


def findCardName(name):
    t = name.split("_")
    return cardToLabel[t[0]+"_"+t[1]]


def current_milli_time():
    return str(round(time.time() * 1000))


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def createKeyPointList(list):
    key = []
    for p in list:
        key.append(ia.Keypoint(x=p[0], y=p[1]))
    return key


def createFitSizeEmptyCard(cardFromPck):
    newSizeCard = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    newSizeCard[sY:sY+cardH, sX:sX+cardW, :] = cardFromPck
    return newSizeCard


def getKp(left, right, shape):
    return [
        [left[0][0],  left[0][1]],
        [left[1][0],  left[1][1]],
        [left[2][0],  left[2][1]],
        [left[3][0],  left[3][1]],

        [right[0][0],  right[0][1]],
        [right[1][0],  right[1][1]],
        [right[2][0],  right[2][1]],
        [right[3][0],  right[3][1]],

        [0, 0],
        [shape[1], 0],
        [0, shape[0]],
        [shape[1], shape[0]],

    ]


def createCardAndRandomPlace(seqOption, newSizeCard, kplist):
    random.seed(datetime.now())
    seqOptionDefault = [
        iaa.Affine(scale=[0.5, 0.7])
        #iaa.Affine(scale=[0.7])
    ]
    ##iaa.Affine(rotate=(-10, 10)),

    seq = iaa.Sequential(seqOptionDefault+seqOption)

    seq.to_deterministic()
    kps = ia.KeypointsOnImage(createKeyPointList(
        kplist), shape=newSizeCard.shape)

    imageAug, kps_aug = seq(image=newSizeCard, keypoints=kps)
    bb = Utils.afterRotaKeyPointToBB(kps_aug.keypoints[0:4], imgW, imgH)
    bbdown = Utils.afterRotaKeyPointToBB(
        kps_aug.keypoints[4:8], imgW, imgH)

    bbcenter = Utils.afterRotaKeyPointToBB(
        kps_aug.keypoints[8:12], imgW, imgH)

    mask = imageAug[:, :, 3]  # alpha層 img1 除了有牌的部分其他的全為0，包括alpha層
    mask = np.stack([mask]*3, -1)  # 堆三個alpha層出來w*h*channel(3)

    return imageAug, bb, bbdown, bbcenter, mask


def toYolov5LabelFormat(s1, s2, s3, s4, s5):
    txt = s1+" "+s2+" "+s3 + " "+s4+" "+s5+"\n"
    return txt


genMax = 100
if (len(sys.argv) > 1):
    genMax = int(sys.argv[1])

genType = "no"
if (len(sys.argv) == 3):
    genType = sys.argv[2]

for genNum in range(0, genMax):

    bg = Utils.randomGetGB(bgPck)
    """Card 1"""
    cardFromPck, cardName, left, right = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value, genType)
    # print("cardFromPck : ", cardFromPck.shape, "cardName :  ", cardName)

    newSizeCard = createFitSizeEmptyCard(cardFromPck)

    # print("tempCard shape :", newSizeCard.shape)

    while True:
        invalid = False
        seq = [iaa.Affine(rotate=(-15, 15))]
        kp = getKp(left, right, cardFromPck.shape)
        imageAug, bb, bbdown, bbcenter, mask1 = createCardAndRandomPlace(
            seq, newSizeCard, kp)

        strW, strH, strX, strY = Utils.getFinialPointToYolov5(bb, mask1)
        strWDown, strHDown, strXDown, strYDown = Utils.getFinialPointToYolov5(
            bbdown, mask1)

        if float(strX) >= 0.9 or float(strY) >= 0.9:
            invalid = True
        else:
            invalid = False
        if not invalid:
            break
    """Card 1 finish """
    """Card 3"""
    cardFromPck3, cardName3, left3, right3 = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value, genType)
    newSizeCard3 = createFitSizeEmptyCard(cardFromPck3)

    seq3 = [
        iaa.Affine(rotate=(-85, -95)),
        iaa.Affine(translate_px={"x": -60}),
    ]
    kp = getKp(left3, right3, cardFromPck3.shape)

    imageAug3, bb3, bbdown3, bbcenter3, mask3 = createCardAndRandomPlace(
        seq3, newSizeCard3, kp)

    strW3, strH3, strX3, strY3 = Utils.getFinialPointToYolov5(bb3, mask3)





    """Card 2"""
    cardFromPck2, cardName2, left2, right2 = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value, genType)
    newSizeCard2 = createFitSizeEmptyCard(cardFromPck2)

   # print("tempCard shape :", newSizeCard.shape)
    while True:
        invalid = False
        intersect_ratio = 0.8  # 交叉比率閾值

        seq2 = [
            iaa.Affine(rotate=(-15, 15)),
            iaa.Affine(translate_percent={
                "x": 0.4, "y":  (0.05, 0.5)}),
        ]

        kp = getKp(left2, right, cardFromPck2.shape)

        imageAug2, bb2, bbdown2, bbcenter2, mask2 = createCardAndRandomPlace(
            seq2, newSizeCard2, kp)

        strW2, strH2, strX2, strY2 = Utils.getFinialPointToYolov5(bb2, mask2)
        strWDown2, strHDown2, strXDown2, strYDown2 = Utils.getFinialPointToYolov5(
            bbdown2, mask2)

        if float(strX2) >= 0.9 or float(strY2) >= 0.9:

            invalid = True
        else:
            invalid = False

        if not invalid:

            p1, p2, p3, p4 = Utils.kkToSinplePoint(bbcenter)

            mainPoly2 = Polygon(p1, p2, p3, p4)

            p21, p22, p23, p24 = Utils.kkToSinplePoint(bbcenter2)

            smallPoly1 = Polygon(p21, p22, p23, p24)

            if mainPoly2.intersection(smallPoly1):
                invalid = True

            if not invalid:
                break

    scaleBg = iaa.Resize({"height": imgW, "width": imgH})
    bg = scaleBg.augment_image(bg)

    final = np.where(mask1, imageAug[:, :, 0:3], bg)
    final = np.where(mask2, imageAug2[:, :, 0:3], final)
    final = np.where(mask3, imageAug3[:, :, 0:3], final)
  

    tempN1 = findCardName(cardName)
    tempN2 = findCardName(cardName2)
    tempN3 = findCardName(cardName3)
  
    print("genNum :", genNum, " tempN1 : ", tempN1,
          "tempN2 : ", tempN2, " tempN3: ")
    txt = toYolov5LabelFormat(tempN1, strX, strY, strW, strH)
    ##txt += toYolov5LabelFormat(tempN1, strXDown, strYDown, strWDown, strHDown)
    txt += toYolov5LabelFormat(tempN2, strX2, strY2, strW2, strH2)
    ##txt += toYolov5LabelFormat(tempN2, strXDown2,strYDown2, strWDown2, strHDown2)
    txt += toYolov5LabelFormat(tempN3, strX3, strY3, strW3, strH3)
   

    toFiler = "train"
    if genNum > int(genMax*0.9)-3:
        toFiler = "val"

    dataJPGName = str(genNum)+"_"+current_milli_time() + \
        "_"+id_generator()+"_"+id_generator()
    cv2.imwrite("./image/genTrainData/images/" +
                toFiler+"/"+dataJPGName+".jpg", final)

    f = open("./image/genTrainData/labels/" +
             toFiler+"/"+dataJPGName+".txt", "w+")
    f.write(txt)
    f.close()

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(final)

    ax.add_patch(Utils.genRectanglePath(bb))
    # ax.add_patch(Utils.genRectanglePath(bbdown))
    ax.add_patch(Utils.genRectanglePath(bb2))

    # ax.add_patch(Utils.genRectanglePath(bbdown2))
    ax.add_patch(Utils.genRectanglePath(bb3))
   
    ax.add_patch(Utils.genRectanglePath(bbcenter))
    ax.add_patch(Utils.genRectanglePath(bbcenter2))
    ax.add_patch(Utils.genRectanglePath(bbcenter3))


    plt.show()

