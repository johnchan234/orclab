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


"""---------------------------------------------------------------------"""
genHowMany = 5

""" Load BG"""
bg_pck_fn = "./image/pck/backgrounds.pck"
#bg_pck_fn = "./image/pck/whiteBackgrounds.pck"
bgPck = Utils.loadRandomBg(bg_pck_fn)


""" Load Card"""
cards_pck_fn = "./image/pck/cards.pck"
cardLoaded, _nb_cards_by_value = Utils.loadRandomCard(cards_pck_fn)

# print("cardFromPck : ",cardFromPck.shape,"card_name : ", card_name)
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


genMax = 100
if (len(sys.argv)>1):
    genMax = int(sys.argv[1])
    
genType="no"    
if (len(sys.argv)==3):
    genType = sys.argv[2]

for genNum in range(0, genMax):

    bg = Utils.randomGetGB(bgPck)
    """Card 1"""
    cardFromPck, card_name, left, right = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value,genType)
    #print("cardFromPck : ", cardFromPck.shape, "card_name : ", card_name)

    newSizeCard = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    newSizeCard[sY:sY+cardH, sX:sX+cardW, :] = cardFromPck

    # print("tempCard shape :", newSizeCard.shape)

    while True:
        invalid = False
        seq = iaa.Sequential([
            iaa.Affine(scale=[0.5, 0.7]),
            iaa.Affine(rotate=(-10, 10)),
         
        ])
        seq.to_deterministic()

        kps = ia.KeypointsOnImage([

            ia.Keypoint(x=left[0][0], y=left[0][1]),
            ia.Keypoint(x=left[1][0], y=left[1][1]),
            ia.Keypoint(x=left[2][0], y=left[2][1]),
            ia.Keypoint(x=left[3][0], y=left[3][1]),

            ia.Keypoint(x=right[0][0], y=right[0][1]),
            ia.Keypoint(x=right[1][0], y=right[1][1]),
            ia.Keypoint(x=right[2][0], y=right[2][1]),
            ia.Keypoint(x=right[3][0], y=right[3][1]),


            ia.Keypoint(x=0, y=0),
            ia.Keypoint(x=cardFromPck.shape[1], y=0),
            ia.Keypoint(x=0, y=cardFromPck.shape[0]),
            ia.Keypoint(x=cardFromPck.shape[1], y=cardFromPck.shape[0]),

        ], shape=newSizeCard.shape)

        image_aug, kps_aug = seq(image=newSizeCard, keypoints=kps)

        # image_aug, kps_aug = seq(image=newSizeCard , keypoints=[kps,kpsd])
        # kps_aug
        # , keypoints=[kps,kpsd]


        bb = Utils.afterRotaKeyPointToBB(kps_aug.keypoints[0:4], imgW, imgH)
        bbdown = Utils.afterRotaKeyPointToBB(
            kps_aug.keypoints[4:8], imgW, imgH)

        bbcenter = Utils.afterRotaKeyPointToBB(
            kps_aug.keypoints[8:12], imgW, imgH)
        # bb = Utils.afterRotaKeyPointToBB(kps_aug[4:4], imgW, imgH)

        mask1 = image_aug[:, :, 3]  # alpha層 img1 除了有牌的部分其他的全為0，包括alpha層
        mask1 = np.stack([mask1]*3, -1)  # 堆三個alpha層出來w*h*channel(3)

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

    
    """Card 2"""
    cardFromPck2, card_name2, left2, right2 = Utils.randomGetCard(
        cardLoaded, _nb_cards_by_value,genType)

    newSizeCard2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
    newSizeCard2[sY:sY+cardH, sX:sX+cardW, :] = cardFromPck2

    # print("tempCard shape :", newSizeCard.shape)
    while True:
        seq2 = iaa.Sequential([
            iaa.Affine(scale=[0.5,0.7]),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Affine(translate_percent={
                "x": 0.4, "y":  (0.05, 0.5)}),
        ])
        seq2.to_deterministic()

        kps2 = ia.KeypointsOnImage([
            ia.Keypoint(x=left2[0][0], y=left2[0][1]),
            ia.Keypoint(x=left2[1][0], y=left2[1][1]),
            ia.Keypoint(x=left2[2][0], y=left2[2][1]),
            ia.Keypoint(x=left2[3][0], y=left2[3][1]),

            ia.Keypoint(x=right[0][0], y=right[0][1]),
            ia.Keypoint(x=right[1][0], y=right[1][1]),
            ia.Keypoint(x=right[2][0], y=right[2][1]),
            ia.Keypoint(x=right[3][0], y=right[3][1]),

            ia.Keypoint(x=0, y=0),
            ia.Keypoint(x=cardFromPck2.shape[1], y=0),
            ia.Keypoint(x=0, y=cardFromPck2.shape[0]),
            ia.Keypoint(x=cardFromPck2.shape[1], y=cardFromPck2.shape[0]),
        ], shape=newSizeCard2.shape)

        image_aug2, kps_aug2 = seq2(image=newSizeCard2, keypoints=kps2)

      
        invalid=False
        intersect_ratio=0.8#交叉比率閾值
        #bb2 = Utils.afterRotaKeyPointToBB(kps_aug2, imgW, imgH)

        bb2 = Utils.afterRotaKeyPointToBB(kps_aug2.keypoints[0:4], imgW, imgH)
        bbdown2 = Utils.afterRotaKeyPointToBB(
            kps_aug2.keypoints[4:8], imgW, imgH)
        bbcenter2 = Utils.afterRotaKeyPointToBB(kps_aug2.keypoints[8:12], imgW, imgH)

        mask2 = image_aug2[:, :, 3]  # alpha層 img1 除了有牌的部分其他的全為0，包括alpha層
        mask2 = np.stack([mask2]*3, -1)  # 堆三個alpha層出來w*h*channel(3)

        strW2, strH2, strX2, strY2 = Utils.getFinialPointToYolov5(bb2, mask2)
        strWDown2, strHDown2, strXDown2, strYDown2 = Utils.getFinialPointToYolov5(
            bbdown2, mask2)
      
        if float(strX2) >= 0.9 or float(strY2) >= 0.9:

            invalid = True
        else:
            invalid = False

        if not invalid: 

            p1,p2,p3,p4 = Utils.kkToSinplePoint(bbcenter)
            #print(p1, p2, p3, p4)
            mainPoly2 = Polygon(p1, p2, p3, p4)
          
            # 將img1的左上角框定區域變換後轉化為polygon
            # 這裏不取0是因為lkps[0]是cardKp變換後的結果
            
            p21, p22, p23, p24 = Utils.kkToSinplePoint(bbcenter2)
            # kps_to_polygon(kps_aug2.keypoints[8:12])
            smallPoly1 = Polygon(p21, p22, p23, p24)
            #a = smallPoly1.area  # 計算該區域面積

            #smallPoly2 = kps_to_polygon(kps_aug2.keypoints[4:8])
            #a22 = smallPoly2.area  # 計算該區域面積
            
            
           # print("a :", a)
           # print("mainPoly2 :", mainPoly2.area)
        
            # 計算img2的polygon與img1的框定polygon的交集
            isIntersect = mainPoly2.intersection(smallPoly1)
          
           # ai = intersect.area  # 計算交集的面積，表示的是被覆蓋掉的面積
            if isIntersect:
                invalid = True
            
            #print("intersect :", intersect)
            
            
            if not invalid:
                break



    



    """Card 2 finish """
    
    # pts = [(kp.x, kp.y) for kp in kps_aug.keypoints]
    # newPolyPTS = Polygon(pts)

 
    """
    for i in range(len(kps.keypoints)):
        before = kps.keypoints[i]
        after = kps_aug.keypoints[i]
        print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
            i, before.x, before.y, after.x, after.y)
        )
    """
    # b_c, g_c, r_c, a_c = cv2.split(image_aug)
    # newI  = cv2.merge([b_c, g_c, r_c])
    # image_after = kps_aug.draw_on_image(newI, size=7)




    scaleBg = iaa.Resize({"height": imgW, "width": imgH})
    bg = scaleBg.augment_image(bg)

    final = np.where(mask1, image_aug[:, :, 0:3], bg)
    final = np.where(mask2, image_aug2[:, :, 0:3], final)

    

    
  
    
    tempN1 =findCardName(card_name)
    tempN2 = findCardName(card_name2)
    print("genNum :",genNum, " tempN1 : ", tempN1, "tempN2 : ", tempN2)


    txt = tempN1+" "+strX+" "+strY + \
        " "+strW+" "+strH+"\n"
    txt += tempN1+" "+strXDown+" "+strYDown + \
        " "+strWDown+" "+strHDown+"\n"

    txt += tempN2+" "+strX2+" "+strY2 + \
        " "+strW2+" "+strH2+"\n"
    txt += tempN2+" "+strXDown2+" "+strYDown2 + \
        " "+strWDown2+" "+strHDown2+"\n"

 
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

    ax.add_patch(Utils.genRectanglePath(bb))
    ax.add_patch(Utils.genRectanglePath(bbdown))
    ax.add_patch(Utils.genRectanglePath(bb2))
    ax.add_patch(Utils.genRectanglePath(bbdown2))
    ax.add_patch(Utils.genRectanglePath(bbcenter))
    ax.add_patch(Utils.genRectanglePath(bbcenter2))

  
    plt.show()
    """
    

