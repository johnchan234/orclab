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
i  = 0
for filename in os.listdir(srcFile):
    if ".png" in filename:
        print(str(i)+":"+filename)
        needW = 26
        needH = 76
        cardName = filename.split(".")[0]
        cards[cardName] = []
        img = cv2.imread(srcFile+filename, cv2.IMREAD_UNCHANGED)
        
        topLeftKey, bottomRightKey, topLeftKeyBB, bottomRightKeyBB = Utils.find8PointOfCorner(img, needW=needW, needH=needH,debug=0)
        #topLeftKey, bottomRightKey = Utils.find8PointOfCorner(img, needW=needW, needH=needH)
        cards[cardName].append((img, topLeftKey, bottomRightKey))
        i+=1

pickle.dump(cards, open(cards_pck_fn, 'wb'))
