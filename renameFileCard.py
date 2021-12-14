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
import string
import random
import time



"""Save card to pck ----------------"""
"""
clubs (♣) = 0
diamonds (♦) = 1
hearts (♥) = 2 
spades (♠) = 3
"""


srcFile = "./image/videoImg/"
outputFile = "./image/videoCard/"

"""LOOP jpg / video cap, find the card and save as PNG"""


def current_milli_time():
    return str(round(time.time() * 1000))
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

for folderename in os.listdir(outputFile):
    renamePath = ""
    if folderename == "clubs":
        renamePath= "0"
    elif folderename == "diamonds":
        renamePath = "1"
    elif folderename == "hearts":
        renamePath = "2"
    elif folderename == "spades":
        renamePath = "3"
    if folderename!="temp":
        for cardPng in os.listdir(outputFile+folderename+"/"):
            x = cardPng.split(".")
            y = x[0].split("_")
            realFileName = renamePath+"_" + \
                y[0]+"_"+y[1]+"_"+current_milli_time()+"_"+id_generator()+"."+x[1]
            
            print(realFileName)
            os.rename(outputFile+folderename+"/"+cardPng,
                      outputFile + "/"+realFileName)




