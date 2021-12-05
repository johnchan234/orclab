import numpy as np
import cv2
import time
import Utils
import os.path

from os import listdir
from os.path import isfile, join


logFile = open("res.txt", "a")



CORNER_WIDTH = 40
CORNER_HEIGHT = 90


# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

print('hello world')
debugPath = "./debug/7/"


best_rank_match_diff = 200000
best_rank_match_name = "Unknown"


print("stream start")

url = "http://streampull.vincifung.com/live/005.m3u8"


vcap = cv2.VideoCapture(url)
fps = vcap.get(cv2.CAP_PROP_FPS)
wt = 30 / fps


cam_quit = 0  # Loop control variable
i = 0
while cam_quit >-1  :
    logFile.write("CEHCK =================\n")
    start_time = time.time()
    # Capture frame-by-frame
    ret, frame = vcap.read()

    if frame is not None:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        """FRAME IS my image need to check"""

        baseImg = frame

        """Start image handle"""
        changeBGImg = Utils.change_bg(baseImg)
        dummy, cnts, outhier = cv2.findContours(
        changeBGImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imwrite(debugPath+str(cam_quit)+"_output.jpg", baseImg)

        outhier = outhier[0]

        for i in range(len(cnts)):
            loopCnts = i
            size = cv2.contourArea(cnts[i])
            peri = cv2.arcLength(cnts[i], True)
            approx = cv2.approxPolyDP(cnts[i], 30, True)
            pts = np.float32(approx)

            # check this conturs is card
            if size > 10000 and (len(approx) == 4) and (outhier[i][3] == -1):

                x, y, w, h = cv2.boundingRect(cnts[i])

                """Start image handleWarp card into 200x300 flattened image using perspective transform"""

                warp = Utils.flattener(baseImg, pts, w, h)
                #cv2.imwrite(debugPath+"wap_"+str(i)+"output.jpg", warp)

                Qcorner = warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
                Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)
                cv2.imwrite(debugPath+str(i)+"_check.jpg", warp)

                white_level = Qcorner_zoom[15, int((CORNER_WIDTH*4)/2)]
                thresh_level = white_level - CARD_THRESH
                if (thresh_level <= 0):
                    thresh_level = 1
                retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)

                # Split in to top and bottom half (top shows rank, bottom shows suit)
                Qrank = query_thresh[50:200, 10:110]
                # Qsuit = query_thresh[186:336, 0:128]

                dummy, Qrank_cnts, hier = cv2.findContours(
                    Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

                ResQrank = ""
                if len(Qrank_cnts) != 0:
                    x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
                    Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
                    Qrank_sized = cv2.resize(
                        Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
                    cv2.imwrite(debugPath+str(i)+"_Qrank_res.jpg", Qrank_sized)
                    ResQrank = Qrank_sized

                if len(Qrank_cnts) != 0:
                    best_rank_name = "NOT_FIND"
                    best_rank_match_diff = 6000

                    mypath = "./train/7/"
                    onlyfiles = [f for f in listdir(
                        mypath) if isfile(join(mypath, f))]
                    #print(onlyfiles)
                    for Rank in onlyfiles:
                        #
                        filename = Rank
                        filepath = "train/7/"

                        if os.path.isfile(filepath+filename):
                            train_ranksImg = cv2.imread(
                                filepath+filename, cv2.IMREAD_GRAYSCALE)
                    
                            diff_img = cv2.absdiff(warp, train_ranksImg)

                            rank_diff = int(np.sum(diff_img)/255)
                            # print(rank_diff)
                            if rank_diff < best_rank_match_diff:
                                best_rank_diff_img = diff_img
                                best_rank_match_diff = rank_diff
                                best_rank_name = Rank

                    if best_rank_name != "NOT_FIND":
                    ## cv2.imwrite(debugPath+'RES_'+str(loopCnts)+best_rank_name, warp)
                        logFile.write("Result : "+best_rank_name+" \n")
                        print("i Result :", best_rank_name)
                    ##else:
                    ##cv2.imwrite(debugPath+'RES_NOT_FIND_'+str(loopCnts) + ".jpg", warp)

        # Press q to close the video windows before it ends if you want
        if cv2.waitKey(22) & 0xFF == ord('q'):
            cv2.imwrite('output_stream.jpg', frame)
          
            break
        dt = time.time() - start_time
        if wt - dt > 0:
            time.sleep(wt - dt)
    else:
        print("Frame is None")
        break

    cam_quit += 1
    logFile.write("CEHCK FINISH =================\n")
vcap.release()
cv2.destroyAllWindows()
print("Video stop")
