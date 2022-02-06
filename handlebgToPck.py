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



#backgrounds_pck_fn = "./image/pck/backgrounds.pck"
#dtd_dir = "./image/dtd-r1.0.1/dtd/images/"
dtd_dir = "./image/whitebg/"
backgrounds_pck_fn = "./image/pck/whiteBackgrounds.pck"
bg_images = []
for subdir in glob(dtd_dir+"/*"):
    for f in glob(subdir+"/*.jpg"):
        bg_images.append(mpimg.imread(f))
print("Nb of images loaded :", len(bg_images))
print("Saved in :", backgrounds_pck_fn)
pickle.dump(bg_images, open(backgrounds_pck_fn, 'wb'))
