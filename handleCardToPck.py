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
import Utils


"""Save card to pck ----------------"""
cards_pck_fn = "./image/pck/cards.pck"
srcFile = "./image/videoCard/"
cards = {}
for filename in os.listdir(srcFile):
    print(filename)
    needW = 40
    needH = 70
    cardName = filename.split(".")[0]
    cards[cardName] = []
    img = cv2.imread(srcFile+filename, cv2.IMREAD_UNCHANGED)
    topLeftKey, bottomRightKey, topLeftKeyBB, bottomRightKeyBB = Utils.find8PointOfCorner(
        img, needW=needW, needH=needH)
    print("img : ", img.shape)
    cards[cardName].append((img, topLeftKey, bottomRightKey))

pickle.dump(cards, open(cards_pck_fn, 'wb'))
