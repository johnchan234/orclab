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
cards_pck_fn = "./image/pck/newCards.pck"
srcFile = "./image/videoCard/topck/"
cards = {}
i  = 0
for filename in os.listdir(srcFile):
    if ".png" in filename:
       # print(str(i)+":"+filename)
        
        needW = 23
        needH = 70
        cardName = filename.split(".")[0]
        cards[cardName] = []
        img = cv2.imread(srcFile+filename, cv2.IMREAD_UNCHANGED)
       # print("cardFromPck : ", img.shape[2])
        
        if img.shape[2] == 3 :
           
            img = Utils.card3To4(img)
           #$ resizeimg = cv2.resize(Nimg, (200, 300), interpolation=cv2.INTER_AREA)
            #print(img.shape)
           # print(Nimg.shape)
          #  print(resizeimg.shape)
        
        if img.shape[0] != 300 or img.shape[1] != 200:
            img = cv2.resize(
                img, (200, 300), interpolation=cv2.INTER_AREA)
            print(str(i)+":"+filename)

            print(img.shape)
            #print(cv2.split(img))
       
        topLeftKey, bottomRightKey, topLeftKeyBB, bottomRightKeyBB = Utils.find8PointOfCorner(img, needW=needW, needH=needH,debug=0)
        #topLeftKey, bottomRightKey = Utils.find8PointOfCorner(img, needW=needW, needH=needH)
        cards[cardName].append((img, topLeftKey, bottomRightKey))
        i+=1

pickle.dump(cards, open(cards_pck_fn, 'wb'))

