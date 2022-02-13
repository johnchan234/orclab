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


def find8PointOfCorner(card, needW, needH,debug):
    # cv2.drawContours(img, [hull_in_img], 0, (0, 255, 0), 1)
    # cv2.imshow("Zone", zone)
    """We need 8 point for topleft and bottom right"""

   # needW = 26
    #needH = 76
    #print(card.shape)

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
        [topLeftKey[0][0] + (needW/2), topLeftKey[0][1]+ (needH/2)]
        ])

    finalBRK = np.array([
        [bottomRightKey[0][0] + (needW/2), bottomRightKey[0][1]+ (needH/2)]
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

    #return finalTopLK, finalBRK


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

def keyPointToBB(point,needW,needH):
    #print("point : ", point[0])
    x = point[0][0]
    y=  point[0][1]

    return [x - (needW/2), y-(needH/2), x + (needW/2), y +(needH/2)]

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
    return cardLoaded,_nb_cards_by_value
   

def randomGetCard(cardLoaded, _nb_cards_by_value,cardType):
    #listA = ['1_10_1', '1_10_1_1642606209972_AH2WY6', '1_10_2', '1_10_2_1642606209974_GWLJ8W', '1_10_3_1642606209976_2GRSYF', '1_10_4_1642606209977_M39OJS', '1_2_1', '1_2_11_1639499483841_UD8L68', '1_2_1_1639499483840_08O33L', '1_2_1_1639670954009_TDGNNH', '1_2_1_1642606209979_SLM9VV', '1_2_2', '1_2_2_1639670954009_GMVG1S', '1_2_2_1642606209981_EF380P', '1_2_3_1639499483841_DAOQ3A', '1_2_3_1639670954010_D75ZM3', '1_2_3_1642606209982_FXDRP9', '1_2_4_1639499483842_RBE25B', '1_2_4_1639670954011_WDTFOQ', '1_2_4_1642606209984_A2S726', '1_3_1', '1_3_1_1639670954011_P9GLIB', '1_3_1_1642606209985_5VXXOK', '1_3_2', '1_3_2_1639670954012_VP9RHK', '1_3_2_1642606209987_WT3KGA', '1_3_3_1642606209988_4M7TGU', '1_3_4_1642606209990_6TBRT8', '1_4_1', '1_4_1_1639499483842_RAIUEN', '1_4_1_1639670954012_0L2MWV', '1_4_1_1642606209991_KNDTUJ', '1_4_2', '1_4_2_1639499483845_W7GK7E', '1_4_2_1639670954013_CLACV6', '1_4_2_1642606209993_YR6O11', '1_4_3_1639499483848_1XS4PE', '1_4_3_1642606209994_IZ7UGM', '1_4_4_1639499483849_GLSIA3', '1_5_1', '1_5_1_1639670954014_HI24CY', '1_5_1_1642606209995_0NIQYT', '1_5_2', '1_5_2_1639670954014_1R15CE', '1_5_2_1642606209997_3RL91O', '1_5_3_1642606209998_86C34X', '1_5_4_1642606210000_UWV41T', '1_6_1', '1_6_1_1642606210006_ODSIX5', '1_6_2', '1_6_2_1642606210008_H2TG61', '1_6_3_1642606210009_APEYW1', '1_7_1', '1_7_1_1639670954015_5RKCA5', '1_7_1_1642606210011_X4LBDK', '1_7_2', '1_7_2_1642606210012_KWBHU6', '1_8_1', '1_8_1_1639670954016_2253E6', '1_8_1_1642606210014_YE6HKD', '1_8_2', '1_8_2_1639670954016_935W53', '1_8_2_1642606210015_4B8YBP', '1_8_3_1639670954017_EVSO1R', '1_8_4_1639670954017_G2ZS7R', '1_9_1', '1_9_1_1639670954019_MIG0FI', '1_9_1_1642606210017_T79PMX', '1_9_2', '1_9_2_1642606210019_7RGAS6', '1_9_3_1642606210020_4D9DNX', '1_9_4_1642606210022_L4UT3N', '1_a_1', '1_a_1_1639499483850_15KDTW', '1_a_1_1642606210023_Y5BF1J', '1_a_2', '1_a_2_1639499483851_0BX7MA', '1_a_2_1642606210024_IS2X2P', '1_a_3_1639499483852_LRK5VD', '1_a_3_1642606210025_GY29K2', '1_a_4_1639499483852_64GBIK', '1_a_4_1642606210027_R5ORMQ', '1_j_1', '1_j_1_1639670954019_66DLYX', '1_j_1_1642606210028_8Z1P6S', '1_j_2', '1_j_2_1639670954020_NNK3ZS', '1_j_2_1642606210029_4WNQYL', '1_k_1', '1_k_1_1639670954021_F7Q85U', '1_k_1_1642606210031_7NIM3R', '1_k_2', '1_k_2_1639670954021_ANWI4G', '1_k_3_1639670954022_FWSBMO', '1_k_3_1642606210032_96PLNT', '1_k_4_1642606210034_HOGDEF', '1_k_5_1642606210035_62YOOG', '1_k_6_1642606210036_TPK64T', '1_q_1', '1_q_1_1639499483853_2TZWCM', '1_q_1_1639670954023_VJHK8O', '1_q_1_1642606210037_RINZPY', '1_q_2', '1_q_2_1639499483854_01FIEZ', '1_q_2_1639670954023_BODPBS', '1_q_2_1642606210038_JGCUIM', '1_q_3_1639499483854_DYGZ3R', '1_q_3_1642606210040_BSBFKT', '1_q_4_1639499483855_Y5XGPL', '1_q_4_1642606210041_NWYC8B', '1_q_5_1642606210042_S0ODWA']
   # listA = ['0_4_1', '0_4_1_1639592890472_BQS46213', '0_4_1_1639592890472_BQS46332']
   #Card is red
  

    list0 =['0_10_1', '0_10_1_1639592890471_MAFT23', '0_10_1_1639592890471_MAFTU2334', '0_10_1_1639592890471_MAFTUK', 
'0_10_1_1639592890471_MAFTUK346', '0_2_3', '0_2_4', '0_2_5_1639499483831_UJIVK44', '0_2_5_1639499483831_UJIVKN', 
'0_2_5_1639499483831_UJIVKN123', '0_2_7_1639499483832_DJGKKR', '0_2_8_1639499483832_2ADXR2', '0_2_9_1639499483833_AVJY2Q', 
'0_3_1', '0_3_1_1639592890471_UTEXW3', '0_3_1_1639592890471_UTEXWa32', '0_3_1_1639592890471_UTEXWL', '0_3_2', '0_3_3', '0_3_4', 
'0_4_1', '0_4_1_1639592890472_BQS46213', '0_4_1_1639592890472_BQS46332', '0_4_1_1639592890472_BQS46E', '0_4_2', '0_5_1',
 '0_5_1_1639499483834_FXNX332', '0_5_1_1639499483834_FXNXS233', '0_5_1_1639499483834_FXNXS8', '0_5_2', '0_5_2_1639499483834_IVXRQN',
 '0_5_3_1639499483835_09UGZM', '0_5_4_1639499483835_GJIF1I', '0_6_1', '0_6_1_1639499483836_IJW3O53', '0_6_1_1639499483836_IJW3OG',
 '0_6_1_1639499483836_IJW3OG123', '0_6_2', '0_6_2_1639499483837_DKXHKU', '0_6_3_1639499483838_46YEE8', '0_6_4_1639499483839_5UAUAF',
 '0_7_1', '0_7_1_1639592890473_NMLIG213', '0_7_1_1639592890473_NMLIG43', '0_7_1_1639592890473_NMLIGX', '0_7_1_1639592890473_NMLIGX23',
 '0_8_1', '0_8_1_1639592890473_SAER3', '0_8_1_1639592890473_SAER3412', '0_8_1_1639592890473_SAER3T', '0_8_1_1639592890473_SAER3T23', 
 '0_9_1', '0_9_1_1639592890474_820Q44', '0_9_1_1639592890474_820QJS', '0_9_1_1639592890474_820QJS3', '0_a_1', '0_a_1_12325ffe_34', 
 '0_a_1_1639592890474_MH72AB', '0_a_1_32323_rgrg', '0_a_2', '0_a_3', '0_j_1', '0_j_1_1639592890475_FNURZ23', 
 '0_j_1_1639592890475_FNURZ4323', '0_j_1_1639592890475_FNURZ4f23', '0_j_1_1639592890475_FNURZ4ff', '0_j_1_1639592890475_FNURZA', 
 '0_j_2', '0_k_1', '0_k_1_1639592890475_WTTP23', '0_k_1_1639592890475_WTTP3', '0_k_1_1639592890475_WTTPRP', '0_k_2', '0_q_1', 
 '0_q_1_1639592890476_JKEV0V', '0_q_1_1639592890476_JKEV0wae', '0_q_1_1639592890476_JKEV3', '0_q_2']


    list1 = ['1_10_1', '1_10_1_1642606209972_AH2WY6', '1_10_2', '1_10_2_1642606209974_GWLJ8W', '1_10_3_1642606209976_2GRSYF', '1_10_4_1642606209977_M39OJS', '1_2_1', '1_2_11_1639499483841_UD8L68', '1_2_1_1639499483840_08O33L', '1_2_1_1639670954009_TDGNNH', '1_2_1_1642606209979_SLM9VV', '1_2_2', '1_2_2_1639670954009_GMVG1S', '1_2_2_1642606209981_EF380P', '1_2_3_1639499483841_DAOQ3A', '1_2_3_1639670954010_D75ZM3', '1_2_3_1642606209982_FXDRP9', '1_2_4_1639499483842_RBE25B', '1_2_4_1639670954011_WDTFOQ', '1_2_4_1642606209984_A2S726', '1_3_1', '1_3_1_1639670954011_P9GLIB', '1_3_1_1642606209985_5VXXOK', '1_3_2', '1_3_2_1639670954012_VP9RHK', '1_3_2_1642606209987_WT3KGA', '1_3_3_1642606209988_4M7TGU', '1_3_4_1642606209990_6TBRT8', '1_4_1', '1_4_1_1639499483842_RAIUEN', '1_4_1_1639670954012_0L2MWV', '1_4_1_1642606209991_KNDTUJ', '1_4_2', '1_4_2_1639499483845_W7GK7E', '1_4_2_1639670954013_CLACV6', '1_4_2_1642606209993_YR6O11', '1_4_3_1639499483848_1XS4PE', '1_4_3_1642606209994_IZ7UGM', '1_4_4_1639499483849_GLSIA3', '1_5_1', '1_5_1_1639670954014_HI24CY', '1_5_1_1642606209995_0NIQYT', '1_5_2', '1_5_2_1639670954014_1R15CE', '1_5_2_1642606209997_3RL91O', '1_5_3_1642606209998_86C34X', '1_5_4_1642606210000_UWV41T', '1_6_1', '1_6_1_1642606210006_ODSIX5', '1_6_2', '1_6_2_1642606210008_H2TG61', '1_6_3_1642606210009_APEYW1', 
'1_7_1', '1_7_1_1639670954015_5RKCA5', '1_7_1_1642606210011_X4LBDK', '1_7_2', '1_7_2_1642606210012_KWBHU6', '1_8_1', '1_8_1_1639670954016_2253E6', '1_8_1_1642606210014_YE6HKD', '1_8_2', '1_8_2_1639670954016_935W53', '1_8_2_1642606210015_4B8YBP', '1_8_3_1639670954017_EVSO1R', '1_8_4_1639670954017_G2ZS7R', '1_9_1', '1_9_1_1639670954019_MIG0FI', '1_9_1_1642606210017_T79PMX', '1_9_2', '1_9_2_1642606210019_7RGAS6', '1_9_3_1642606210020_4D9DNX', '1_9_4_1642606210022_L4UT3N', '1_a_1', '1_a_1_1639499483850_15KDTW', '1_a_1_1642606210023_Y5BF1J', '1_a_2', '1_a_2_1639499483851_0BX7MA', '1_a_2_1642606210024_IS2X2P', '1_a_3_1639499483852_LRK5VD', '1_a_3_1642606210025_GY29K2', '1_a_4_1639499483852_64GBIK', '1_a_4_1642606210027_R5ORMQ', '1_j_1', '1_j_1_1639670954019_66DLYX', '1_j_1_1642606210028_8Z1P6S', '1_j_2', '1_j_2_1639670954020_NNK3ZS', '1_j_2_1642606210029_4WNQYL', '1_k_1', '1_k_1_1639670954021_F7Q85U', '1_k_1_1642606210031_7NIM3R', '1_k_2', '1_k_2_1639670954021_ANWI4G', '1_k_3_1639670954022_FWSBMO', '1_k_3_1642606210032_96PLNT', '1_k_4_1642606210034_HOGDEF', '1_k_5_1642606210035_62YOOG', '1_k_6_1642606210036_TPK64T', '1_q_1', '1_q_1_1639499483853_2TZWCM', '1_q_1_1639670954023_VJHK8O', '1_q_1_1642606210037_RINZPY', '1_q_2', '1_q_2_1639499483854_01FIEZ', '1_q_2_1639670954023_BODPBS', '1_q_2_1642606210038_JGCUIM', '1_q_3_1639499483854_DYGZ3R', '1_q_3_1642606210040_BSBFKT', '1_q_4_1639499483855_Y5XGPL', '1_q_4_1642606210041_NWYC8B', '1_q_5_1642606210042_S0ODWA']


    list2=['2_10_1_1639669744695_DAA455', '2_10_1_1639669744695_DAA45546', '2_10_1_1639669744695_DAA455dd', '2_10_1_1639669744695_DAA4M23', '2_10_1_1639669744695_DAA4MB', '2_2_1_1639669744696_HXRU123', '2_2_1_1639669744696_HXRUP34', '2_2_1_1639669744696_HXRUPD', '2_2_1_1639670954024_R67YLS', '2_2_2_1639670954025_7FIZDP', '2_2_3_1639670954025_A8DW8Z', '2_2_4_1639670954026_83MFSE', '2_3_1_1639669744697_DQ5T6123', '2_3_1_1639669744697_DQ5T63', '2_3_1_1639669744697_DQ5T6V', '2_3_1_1639670954026_YCFZLX', '2_3_2_1639670954027_I7DBTR', '2_3_3_1639670954028_THL9DZ', '2_3_4_1639670954028_ZB6UNA', '2_4_1_1639669744698_BD1R4', '2_4_1_1639669744698_BD1RI123', '2_4_1_1639669744698_BD1RI5', '2_4_2_1639669744698_C9VM6Y', '2_4_3_1639669744699_QH7293', '2_5_1_1639669744700_NGC443', '2_5_1_1639669744700_NGCL123', '2_5_1_1639669744700_NGCL3', '2_5_1_1639669744700_NGCLJD', '2_6_1_1639669744700_ZAUH16', '2_6_2_1639669744701_6DO3H4', '2_6_3_1639669744702_LRSL143', '2_6_3_1639669744702_LRSL1E', '2_6_3_1639669744702_LRSL1E33', '2_6_4_1639669744702_ZO7SKZ', '2_7_1_1639669744703_N0PC442', '2_7_1_1639669744703_N0PCJ552', '2_7_1_1639669744703_N0PCJO', '2_7_2_1639669744703_4896Y4', '2_7_3_1639669744704_656JGX', '2_8_1_1639669744705_A3EQ323', '2_8_1_1639669744705_A3EQN', '2_8_1_1639669744705_A3EQNL', '2_8_2_1639669744706_I2HDSD', '2_8_3_1639669744706_UUB4GF', 
'2_9_1_1639669744707_4HKSTY - Copy (2)', '2_9_1_1639669744707_4HKSTY - Copy (3)', '2_9_1_1639669744707_4HKSTY - Copy', '2_9_1_1639669744707_4HKSTY', '2_a_1_1639669744708_60IJ6D', '2_a_2_1639669744709_LMFLQR', '2_a_3_1639669744709_UWDY523', '2_a_3_1639669744709_UWDY8123', '2_a_3_1639669744709_UWDY84', '2_a_4_1639669744710_7E4HPB', '2_a_5_1639669744710_MFUNTF', '2_a_6_1639669744711_BFVXWA', '2_j_1', '2_j_1_1639669744712_1LEE6O', '2_j_2_1639669744713_XL5FYQ', '2_j_3_1639669744713_CCTD623d', '2_j_3_1639669744713_CCTD6Y', '2_j_3_1639669744713_CCTdd2', '2_j_4_1639669744714_17OHKK', '2_k_1_1639669744715_DSHBF123', '2_k_1_1639669744715_DSHBF324', '2_k_1_1639669744715_DSHBFK', '2_k_2_1639669744715_HGYI1T', '2_k_3_1639669744717_7I3123', '2_k_3_1639669744717_7I31K5', '2_k_4_1639669744718_3TBXNB', '2_q_1_1639669744719_X3BNR2', '2_q_2_1639669744719_IAMSO4', '2_q_3_1639669744720_BVW22awd', '2_q_3_1639669744720_BVW22C - Copy', '2_q_3_1639669744720_BVW22C', '2_q_3_1639669744720_BVW22dd']

    list3=['3_10_1_1639669744722_TFVX123', '3_10_1_1639669744722_TFVXS45', '3_10_1_1639669744722_TFVXS6',
    '3_10_2_1639669744723_LS4T0P', '3_10_3_1639669744723_BE022X', '3_2_1_1639669744724_RZW0N123',
     '3_2_1_1639669744724_RZW0N6', '3_2_2_1639669744724_8L1BK123', '3_2_2_1639669744724_8L1BK2',
      '3_2_3_1639669744725_UR57C5', '3_3_1_1639669744726_5ACW', '3_3_1_1639669744726_5ACWA123',
       '3_3_1_1639669744726_5ACWA8', '3_3_2_1639669744727_91D9DI', '3_3_3_1639669744727_AAWXDG',
        '3_4_1_1639669744728_QLG6132', '3_4_1_1639669744728_QLG6K123', '3_4_1_1639669744728_QLG6KB', 
        '3_4_2_1639669744729_W53HWY', '3_4_3_1639669744729_O0NUNC', '3_5_1_1639669744730_04V123',
         '3_5_1_1639669744730_04V6T2', '3_5_1_1639669744730_04V6T24', '3_5_2_1639669744730_HE1QJQ',
          '3_5_3_1639669744731_4B1Z1B', '3_6_1_1639669744731_AFIY123', '3_6_1_1639669744731_AFIYV123', 
          '3_6_1_1639669744731_AFIYV8', '3_6_2_1639669744732_6TEI6N', '3_7_1_1639669744733_2P23', '3_7_1_1639669744733_2P8K123', 
          '3_7_1_1639669744733_2P8KST', '3_7_2_1639669744733_7WCCZW', '3_7_3', '3_7_3_1639669744734_ZE951C', '3_8_1_1639669744734_6WLX2K',
           '3_8_2_1639669744735_REZU123', '3_8_2_1639669744735_REZUQ23', '3_8_2_1639669744735_REZUQ4', '3_9_1_1639669744735_46NR23',
           '3_9_1_1639669744735_46NRX123', '3_9_1_1639669744735_46NRXY', '3_9_2_1639669744736_TKMJVU', '3_9_3_1639669744738_DZXU6A', 
           '3_a_1_1639669744739_BFDL7E', '3_a_2_1639669744740_3VU3IK', '3_a_3_1639669744740_EB9Z3123', '3_a_3_1639669744740_EB9Z335',
            '3_a_3_1639669744740_EB9Z37', '3_j_1_1639669744741_AM5CM123', '3_j_1_1639669744741_AM5CM443', '3_j_1_1639669744741_AM5CMZ', 
            '3_j_2_1639669744742_CG3D2J', '3_j_3_1639669744743_AAN4IR', '3_k_1_1639669744744_9943', '3_k_1_1639669744744_994W323',
             '3_k_1_1639669744744_994W3O', '3_k_2_1639669744745_C68327', '3_k_3', '3_q_1_1639669744746_X1X123', '3_q_1_1639669744746_X1X966',
              '3_q_1_1639669744746_X1X966344', '3_q_2_1639669744747_VE4N97', '3_q_3_1639669744747_NGHKKO']


    sys_random = random.SystemRandom()
    listA= list1 + list2+list0 + list3
    if cardType=="red":
        listA= list1 + list2
    if cardType=="black":
        listA= list0 + list3
    if cardType=="3only":
        listA= list3
    if cardType=="0only":
        listA =list0   
    #print(list(cardLoaded.keys()))
    
    card_name = sys_random.choice(listA)
    #print("card_name ", cardLoaded.keys())
    ##card_name = "1_j_1"
    card33, left, right = cardLoaded[card_name][random.randint(0, _nb_cards_by_value[card_name]-1)]
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
    return strW,strH,strX,strY

