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
"""
clubs (♣) = 0
diamonds (♦) = 1
hearts (♥) = 2 
spades (♠) = 3
"""


srcFile = "./image/videoImg/"
outputFile = "./image/videoCard/new/"

"""LOOP jpg / video cap, find the card and save as PNG"""

print(srcFile)
  
for i, filename in enumerate(glob(srcFile+"*.jpg")):
 
    baseImg = cv2.imread(filename)
    print("handle filename  : ", filename)
    card = Utils.getCardFomImg(baseImg)
  

    for num, x in enumerate(card):
        Utils.saveCard(x, outputFile, str(i)+"_"+str(num)+".png")
    
   
#cv2.waitKey(0)
#cv2.destroyAllWindows()
