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

""" Gen """
sX = 0
sY = 2
cardW = 200
cardH = 300

imgW = 720
imgH = 720
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
"""-----------------------"""



class MyCard:
    cardFromPck=""
    cardName =""
    leftpoint=""
    rightpoint=""

    def __init__(self, cardFromPck, cardName, leftpoint, rightpoint):

        self.cardFromPck = cardFromPck
        self.cardName = cardName
        self.leftpoint = leftpoint
        self.rightpoint = rightpoint

        self.newSizeCard = self.createFitSizeEmptyCard(cardFromPck)


    def createKeyPointList(self,list):
        key = []
        for p in list:
            key.append(ia.Keypoint(x=p[0], y=p[1]))
        return key

    def createFitSizeEmptyCard(self,cardFromPck):
        newSizeCard = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        newSizeCard[sY:sY+cardH, sX:sX+cardW, :] = cardFromPck
        return newSizeCard

    def getKp(self):
        left =self.leftpoint
        right = self.rightpoint
        shape = self.cardFromPck.shape

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

    def setSeqOption(self, seqOption):
        self.defineSeq = seqOption    
    def createCardAndRandomPlace(self):

        kplist = self.getKp()
      
        seqOptionDefault = [
            iaa.Affine(scale=[0.5, 0.7])
            #iaa.Affine(scale=[0.7])
        ]
        ##iaa.Affine(rotate=(-10, 10)),

        seq = iaa.Sequential(seqOptionDefault + self.defineSeq)

        seq.to_deterministic()
        kps = ia.KeypointsOnImage(self.createKeyPointList(
            kplist), shape=self.newSizeCard.shape)

        imageAug, kps_aug = seq(image=self.newSizeCard, keypoints=kps)
        bb = Utils.afterRotaKeyPointToBB(kps_aug.keypoints[0:4], imgW, imgH)
        bbdown = Utils.afterRotaKeyPointToBB(
            kps_aug.keypoints[4:8], imgW, imgH)

        bbcenter = Utils.afterRotaKeyPointToBB(
            kps_aug.keypoints[8:12], imgW, imgH)

        mask = imageAug[:, :, 3]  # alpha層 img1 除了有牌的部分其他的全為0，包括alpha層
        mask = np.stack([mask]*3, -1)  # 堆三個alpha層出來w*h*channel(3)


        self.imageAug= imageAug
        self.bb = bb
        self.bbdown =bbdown
        self.bbcenter = bbcenter
        self.mask = mask

        self.bbXY = self.getFinialPointToYolov5(bb)
        self.bbDownXY = self.getFinialPointToYolov5(bbdown)

        return imageAug, bb, bbdown, bbcenter, mask


    def getFinialPointToYolov5(self,bb):
        final = self.mask

        bbW = (bb.x2-bb.x1)
        bbH = bb.y2-bb.y1

        centerY = bb.y1+(bbH/2)
        centerX = bb.x1+(bbW/2)

        strW = Utils.roundTo(bbW/final.shape[0])
        strH = Utils.roundTo(bbH/final.shape[0])
        strX = Utils.roundTo(centerX/final.shape[0])
        strY = Utils.roundTo(centerY/final.shape[0])
        return {"w":strW, "h":strH, "x":strX, "y":strY}


    def getCenterSinglePoint(self):
      return Utils.kkToSinplePoint(self.bbcenter)

    def getMask(self):
        return self.mask
    def getImageAug(self):
        return self.imageAug
    def getbbCenter(self):
        return self.bbcenter

    def getBB(self):
        return self.bb
    def getBBDown(self):
        return self.bbdown

    def getBBcenter(self):
        return self.bbcenter
        

    def getBBXY(self):
        return self.bbXY

    def getBBDownXY(self):
        return self.bbDownXY
    def getRealCardName(self):
        t = self.cardName.split("_")
        return cardToLabel[t[0]+"_"+t[1]] 
         

    ##card1Name, bbXy["x"], bbXy["y"], bbXy["w"], bbXy["h"])

    def getBBYolov5LabelString(self):
        bbXy = self.getBBXY()
        txt = self.toYolov5LabelFormat(
            self.getRealCardName(), bbXy["x"], bbXy["y"], bbXy["w"], bbXy["h"])

        return txt

    def getBBDownYolov5LabelString(self):
        bbXy = self.getBBDownXY()

        txt = self.toYolov5LabelFormat(
            self.getRealCardName(), bbXy["x"], bbXy["y"], bbXy["w"], bbXy["h"])
        return txt

    def toYolov5LabelFormat(self,s1, s2, s3, s4, s5):
        txt = s1+" "+s2+" "+s3 + " "+s4+" "+s5+"\n"
        return txt
