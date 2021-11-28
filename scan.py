import json
import numpy as np
import cv2
import time
import logUtils

from os import listdir
from os.path import isfile, join

logFile = logUtils.logOpen("./log")
errorLogFile = logUtils.errorOpen("./log")


## Get confih from json
try:
    with open('config.json', "r") as f:
        #stuff goes here
        data = json.load(f)
      
        f.close()
except BaseException as err:
    #do what you want if there is an error with the file opening
    logUtils.writeLog(errorLogFile, "json error , please check : "+str(err))
    quit()


streamUrl = data['streamLink']
version = data["version"]
logUtils.writeLog(logFile, "Start OCR , version : "+ version)
logUtils.writeLog(logFile, "stream : "+streamUrl)

## Start handel Video
vcap = cv2.VideoCapture(streamUrl)
fps = vcap.get(cv2.CAP_PROP_FPS)
wt = 30 / fps

cam_quit = 0  # Loop control variable
i = 0
while cam_quit > 5:
    logUtils.writeLog("CEHCK =================\n")
    start_time = time.time()
    # Capture frame-by-frame
    ret, frame = vcap.read()
    """---------------------------------------------------------------------------------Start handle Image Logic---------------------------------------------------------------------------------"""
    cv2.imshow('frame', frame)

    """---------------------------------------------------------------------------------End Image Logic---------------------------------------------------------------------------------"""
    if cv2.waitKey(22) & 0xFF == ord('q'):
        ##cv2.imwrite('output_stream.jpg', frame)
        break
    dt = time.time() - start_time
    if wt - dt > 0:
        time.sleep(wt - dt)
    else:
        print("Frame is None")
        break

    cam_quit += 1
    logUtils.writeLog("CEHCK FINISH =================\n")
vcap.release()
cv2.destroyAllWindows()
print("Video stop")


