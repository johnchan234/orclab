import os
import numpy as np
import cv2
import time
import Utils
import newCardObject
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


cornerXmin = 2
cornerXmax = 10.5
cornerYmin = 2.5
cornerYmax = 23


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
        # iaa.Affine(scale=[0.7])
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
    usingType = "1"

    card1seq = [iaa.Affine(rotate=(-15, 15))]
    card2seq = [
        iaa.Affine(rotate=(-15, 15)),
        iaa.Affine(translate_percent={
            "x": 0.4, "y":  (0.05, 0.5)}),
    ]
    card3seq = iaa.Affine(rotate=(-85, -95)),
    iaa.Affine(translate_px={"x": -60}),
    if genNum % 2 == 0:
        usingType = 2

    bg = Utils.randomGetGB(bgPck)

    """Card 1"""
    cardFromPck, cardName, left, right = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value, genType)

    card1 = newCardObject.MyCard(cardFromPck, cardName, left, right)

    while True:
        invalid = False

        card1.setSeqOption([iaa.Affine(rotate=(-15, 15))])
        card1.createCardAndRandomPlace()

        temp = card1.getBBXY()

        if float(temp["x"]) >= 0.9 or float(temp["y"]) >= 0.9:
            invalid = True
        else:
            invalid = False
        if not invalid:
            break
    """Card 1 finish """
    """Card 3"""
    cardFromPck3, cardName3, left3, right3 = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value, genType)

    card3 = newCardObject.MyCard(cardFromPck3, cardName3, left3, right3)

    seq3 = [
        iaa.Affine(rotate=(-85, -95)),
        iaa.Affine(translate_px={"x": -60}),
    ]
    card3.setSeqOption(seq3)
    card3.createCardAndRandomPlace()

   
    """Card 2"""
    cardFromPck2, cardName2, left2, right2 = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value, genType)
    card2 = newCardObject.MyCard(cardFromPck2, cardName2, left2, right2)


   # print("tempCard shape :", newSizeCard.shape)
    while True:
        invalid = False
        intersect_ratio = 0.8  # 交叉比率閾值

        seq2 = [
            iaa.Affine(rotate=(-15, 15)),
            iaa.Affine(translate_percent={
                "x": 0.4, "y":  (0.05, 0.5)}),
        ]

        #kp = getKp(left2, right, cardFromPck2.shape)
        card2.setSeqOption(seq2)
        card2.createCardAndRandomPlace()

        """
        imageAug2, bb2, bbdown2, bbcenter2, mask2 = createCardAndRandomPlace(
            seq2, newSizeCard2, kp)

        strW2, strH2, strX2, strY2 = Utils.getFinialPointToYolov5(bb2, mask2)
        strWDown2, strHDown2, strXDown2, strYDown2 = Utils.getFinialPointToYolov5(
            bbdown2, mask2)
        """
        temp = card1.getBBXY()
        if float(temp["x"]) >= 0.9 or float(temp["y"]) >= 0.9:

            invalid = True
        else:
            invalid = False

        if not invalid:

            ##p1, p2, p3, p4 = Utils.kkToSinplePoint(bbcenter)
            p1, p2, p3, p4 = card1.getCenterSinglePoint()

            mainPoly2 = Polygon(p1, p2, p3, p4)

            p21, p22, p23, p24 = card2.getCenterSinglePoint()

            smallPoly1 = Polygon(p21, p22, p23, p24)

            if mainPoly2.intersection(smallPoly1):
                invalid = True

            if not invalid:
                break

    scaleBg = iaa.Resize({"height": imgW, "width": imgH})
    bg = scaleBg.augment_image(bg)

    #final = np.where(mask1, imageAug[:, :, 0:3], bg)
    tMask = card1.getImageAug()
    final = np.where(card1.getMask(), tMask[:, :, 0:3], bg)

    tMask = card2.getImageAug()
    final = np.where(card2.getMask(), tMask[:, :, 0:3], final)

    tMask = card3.getImageAug()
    final = np.where(card3.getMask(), tMask[:, :, 0:3], final)

    #tempN1 = findCardName(cardName)
   
    #tempN2 = findCardName(cardName2)
  

    print("genNum :", genNum, " card1 : ", card1.getRealCardName(),
          "card2 : ",  card2.getRealCardName(), " card3: ", card3.getRealCardName())

 

    txt = card1.getBBYolov5LabelString()
    txt += card1.getBBDownYolov5LabelString()


    txt += card2.getBBYolov5LabelString()
    txt += card2.getBBDownYolov5LabelString()
    ##txt += toYolov5LabelFormat(tempN2, strXDown2,strYDown2, strWDown2, strHDown2)

    txt += card3.getBBYolov5LabelString()
    txt += card3.getBBDownYolov5LabelString()


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
    
    """

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(final)

    ax.add_patch(Utils.genRectanglePath(card1.getBB()))
    ax.add_patch(Utils.genRectanglePath(card1.getBBDown()))

    ax.add_patch(Utils.genRectanglePath(card2.getBB()))
    ax.add_patch(Utils.genRectanglePath(card2.getBBDown()))

    ax.add_patch(Utils.genRectanglePath(card3.getBB()))
    ax.add_patch(Utils.genRectanglePath(card3.getBBDown()))

    ax.add_patch(Utils.genRectanglePath(card1.getBBcenter()))
    ax.add_patch(Utils.genRectanglePath(card2.getBBcenter()))
    ax.add_patch(Utils.genRectanglePath(card3.getBBcenter()))

    print(txt)
    plt.show()
"""
