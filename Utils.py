import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pickle
from glob import glob
import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon
from sympy import Point, Polygon
from datetime import datetime

BKG_THRESH = 60


DEBUG = 0


class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []  # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"


def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0

    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'

        if os.path.isfile(filepath+filename):
            train_ranks[i].img = cv2.imread(
                filepath+filename, cv2.IMREAD_GRAYSCALE)
            i = i + 1
            print("filepath+filename : ", filepath+filename)
    return train_ranks


def change_bg(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # The best threshold level depends on the ambient lighting conditions.
    # For bright lighting, a high threshold must be used to isolate the cards
    # from the background. For dim lighting, a low threshold must be used.
    # To make the card detector independent of lighting conditions, the
    # following adaptive threshold method is used.
    #
    # A background pixel in the center top of the image is sampled to determine
    # its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
    # than that. This allows the threshold to adapt to the lighting conditions.
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh


def card3To4(img):
    b_c, g_c, r_c = cv2.split(img)
    a_c = np.ones(b_c.shape, dtype=b_c.dtype) * 255
    final = cv2.merge([b_c, g_c, r_c, a_c])
    return final


def getCardFomImg(baseImg):
    res = []

    changeBGImg = change_bg(baseImg)
    cv2.imshow('changeBGImg', changeBGImg)
    return
    cnts, outhier = cv2.findContours(
        changeBGImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    outhier = outhier[0]

    for i in range(len(cnts)):
        loopCnts = i
        size = cv2.contourArea(cnts[i])
        peri = cv2.arcLength(cnts[i], True)
        approx = cv2.approxPolyDP(cnts[i], 30, True)
        pts = np.float32(approx)

        # check this conturs is card
        if size > 8000 and (len(approx) == 4) and (outhier[i][3] == -1):
            print("find!!")
            x, y, w, h = cv2.boundingRect(cnts[i])

            """Start image handleWarp card into 200x300 flattened image using perspective transform"""

            warp = flattener(baseImg, pts, w, h)

            if DEBUG == 1:
                cv2.imshow('frame2', warp)

            b_c, g_c, r_c = cv2.split(warp)
            a_c = np.ones(b_c.shape, dtype=b_c.dtype) * 255
            final = cv2.merge([b_c, g_c, r_c, a_c])

            res.append(final)
    return res


def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h:  # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.

    if w > 0.8*h and w < 1.2*h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left

    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1,
                                              maxHeight-1], [0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

    return warp


def saveCard(image, path, name):
    cv2.imwrite(path+name, image)


def find8PointOfCorner(card, needW, needH, debug):
    # cv2.drawContours(img, [hull_in_img], 0, (0, 255, 0), 1)
    # cv2.imshow("Zone", zone)
    """We need 8 point for topleft and bottom right"""

   # needW = 26
    #needH = 76
    # print(card.shape)

    cardH = card.shape[0]
    cardW = card.shape[1]

    marginNeed = 5
    topleftCornerX = marginNeed+2
    topleftCornerY = marginNeed+marginNeed

# cardW - marginNeed -20
    bottomRightCornerX = cardW - marginNeed+2
    bottomRightCornerY = cardH - marginNeed-marginNeed

    topLeftKey = np.array([
        [topleftCornerX, topleftCornerY],
        [topleftCornerX + needW, topleftCornerY],
        [topleftCornerX, topleftCornerY + needH],
        [topleftCornerX + needW, topleftCornerY + needH],

    ])
    bottomRightKey = np.array([
        [bottomRightCornerX - needW, bottomRightCornerY - needH],
        [bottomRightCornerX, bottomRightCornerY - needH],
        [bottomRightCornerX - needW, bottomRightCornerY],
        [bottomRightCornerX, bottomRightCornerY],
    ]
    )
    finalTopLK = np.array([
        [topLeftKey[0][0] + (needW/2), topLeftKey[0][1] + (needH/2)]
    ])

    finalBRK = np.array([
        [bottomRightKey[0][0] + (needW/2), bottomRightKey[0][1] + (needH/2)]
    ])

    topLeftbb = keyPointToBB(finalTopLK, needW, needH)

    bottomRightbb = keyPointToBB(finalBRK, needW, needH)

    if debug == 1:

        arr = np.concatenate((topLeftbb, bottomRightbb), axis=0)
        b_c, g_c, r_c, a_c = cv2.split(card)
        newI = cv2.merge([b_c, g_c, r_c])

        bbs = ia.BoundingBoxesOnImage([
            pointToBoundingBox(topLeftbb),
            pointToBoundingBox(bottomRightbb)],
            shape=newI.shape)
        ia.imshow(bbs.draw_on_image(newI, size=2))
        plt.show()

    return topLeftKey, bottomRightKey, topLeftbb, bottomRightbb

    # return finalTopLK, finalBRK


def afterRotaKeyPointToBB(kps_aug, imgW, imgH):

    extend = 3  # To make the bounding box a little bit bigger
    kpsx = [kp.x for kp in kps_aug]
    minx = max(0, int(min(kpsx)-extend))
    maxx = min(imgW, int(max(kpsx)+extend))
    kpsy = [kp.y for kp in kps_aug]
    miny = max(0, int(min(kpsy)-extend))
    maxy = min(imgH, int(max(kpsy)+extend))
    #print("x1 : ", minx, "y1 : ", miny, "x2 ： ", maxx, "y2 :", maxy)
    return ia.BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy)


def toKeyPoint(point):
    kps = []
    for val in point:
        kps.append(ia.Keypoint(x=val[0], y=val[1]))
    return kps


def keyPointToBB(point, needW, needH):
    #print("point : ", point[0])
    x = point[0][0]
    y = point[0][1]

    return [x - (needW/2), y-(needH/2), x + (needW/2), y + (needH/2)]


def pointToBoundingBox(point):
    return ia.BoundingBox(x1=point[0], y1=point[1], x2=point[2], y2=point[3])


def genRectanglePath(bb):
    return patches.Rectangle((bb.x1, bb.y1), bb.x2-bb.x1,
                             bb.y2-bb.y1, linewidth=1, edgecolor='b', facecolor='none')


def kkToSinplePoint(bbcenter):
    p1, p2, p3, p4 = map(Point, [(bbcenter.x1, bbcenter.y1), (bbcenter.x2,
                                                              bbcenter.y1), (bbcenter.x1, bbcenter.y2), (bbcenter.x2, bbcenter.y2)])
    return p1, p2, p3, p4


def boundingBox(point):
    print(point)
    bb = []
    for val in point:

        bb.append(ia.BoundingBox(x1=val[0], y1=val[1], x2=val[2], y2=val[3]))
    return bb


def display_img(img, polygons=[], channels="bgr", size=9):
    """
        Function to display an inline image, and draw optional polygons (bounding boxes, convex hulls) on it.
        Use the param 'channels' to specify the order of the channels ("bgr" for an image coming from OpenCV world)
    """
    if not isinstance(polygons, list):
        polygons = [polygons]
    if channels == "bgr":  # bgr (cv2 image)
        nb_channels = img.shape[2]
        if nb_channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_facecolor((0, 0, 0))
    ax.imshow(img)
    for polygon in polygons:
        # An polygon has either shape (n,2),
        # either (n,1,2) if it is a cv2 contour (like convex hull).
        # In the latter case, reshape in (n,2)
        if len(polygon.shape) == 3:
            polygon = polygon.reshape(-1, 2)
        patch = patches.Polygon(polygon, linewidth=1,
                                edgecolor='g', facecolor='none')
        ax.add_patch(patch)


def loadRandomCard(cards_pck_fn):
    cardLoaded = pickle.load(open(cards_pck_fn, 'rb'))
    # self._cards is a dictionary where keys are card names (ex:'Kc') and values are lists of (img,hullHL,hullLR)

    _nb_cards_by_value = {k: len(cardLoaded[k]) for k in cardLoaded}
    return cardLoaded, _nb_cards_by_value


def randomGetCard(cardLoaded, _nb_cards_by_value, cardType):
  

    list0 = []

    list1 = []

    list2 = []

    list3 = []

  

    for x in list(cardLoaded.keys()):

        if x[0] == '0':
            list0.append(x)
        if x[0] == '1':
            list1.append(x)
        if x[0] == '2':
            list2.append(x)
        if x[0] == '3':
            list3.append(x)

    sys_random = random.SystemRandom()
    listA = list1 + list2+list0 + list3
    if cardType == "red":
        listA = list1 + list2
    if cardType == "black":
        listA = list0 + list3
    if cardType == "3only":
        listA = list3
    if cardType == "0only":
        listA = list0
    if  cardType == "2only":
        listA = list2
    if  cardType == "1only":
        listA = list1

    card_name = sys_random.choice(listA)
    #print("card_name ", cardLoaded.keys())
    ##card_name = "1_j_1"
    card33, left, right = cardLoaded[card_name][random.randint(
        0, _nb_cards_by_value[card_name]-1)]
    return card33, card_name, left, right


def loadRandomBg(backgrounds_pck_fn):
    return pickle.load(open(backgrounds_pck_fn, 'rb'))


def randomGetGB(_images):
    num = len(_images)
    return _images[random.randint(0, num-1)]


def roundTo(num):
    return str(round(num, 6))


def getFinialPointToYolov5(bb, final):

    bbW = (bb.x2-bb.x1)
    bbH = bb.y2-bb.y1

    centerY = bb.y1+(bbH/2)
    centerX = bb.x1+(bbW/2)

    strW = roundTo(bbW/final.shape[0])
    strH = roundTo(bbH/final.shape[0])
    strX = roundTo(centerX/final.shape[0])
    strY = roundTo(centerY/final.shape[0])
    return strW, strH, strX, strY
